''' Audio Diffusion model based on flow-matching
    Part of the code is based on MuCodec
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import random
import inspect
import numpy as np
from tqdm import tqdm
import typing as tp
from abc import ABC
import os
import torchaudio
from einops import repeat
import diffusers
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDPMScheduler
from tools.tokenizer.ReasoningCodec_film.models.processor import Feature2DProcessor 
from tools.tokenizer.ReasoningCodec_film.models.transformer_1d_flow import Transformer1DModel, ProjectLayer
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel,HubertModel
from torch.cuda.amp import autocast
from tools.tokenizer.ReasoningCodec_film.modules.our_MERT_BESTRQ.test import load_best_rq_model
from tools.tokenizer.ReasoningCodec_film.models.PretrainedModel import BESTRQ_Model
from vector_quantize_pytorch import ResidualVQ
from whisper.audio import log_mel_spectrogram
import whisper
from tools.tokenizer.ReasoningCodec_film.models.modeling_whisper import WhisperModel
from typing import Dict, Iterable, Optional, List
from transformers import WhisperFeatureExtractor
from tools.tokenizer.ReasoningCodec_film.models.vocos import VocosBackbone
from transformers import LlamaTokenizer, StoppingCriteriaList, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from tools.tokenizer.ReasoningCodec_film.models.model_utils import load_state, verify_state, StoppingCriteriaSub
from tools.tokenizer.ReasoningCodec_film.modules.transformer import TransformerBlock
from transformers import LlamaTokenizer, StoppingCriteriaList, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaForCausalLM # Llama-3.2-3B-Instruct
import logging
import json
import contextlib
import random
from transformers import AutoModel
from tools.tokenizer.ReasoningCodec_film.models.semantic_decoder import Decoder, FiLMEncoder

def _chk(x, name):
    if not torch.is_tensor(x): return
    if not torch.isfinite(x).all():
        bad = ~torch.isfinite(x)
        n_nan = bad.sum().item()
        mx = torch.nanmax(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)).item()
        mn = torch.nanmin(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)).item()
        raise RuntimeError(f"[NaN DETECT] {name}: shape={tuple(x.shape)} nan/inf={n_nan}, min={mn:.3e}, max={mx:.3e}")

def _safe(x, name):
    _chk(x, name)                # 发现就抛，方便二分定位
    return x


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        estimator,
    ):
        super().__init__()
        self.sigma_min = 1e-4
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, n_timesteps, temperature=1.0):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span)

    def solve_euler(self, x, incontext_x, incontext_length, t_span, mu, added_cond_kwargs, guidance_scale):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        noise = x.clone()

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        for step in tqdm(range(1, len(t_span))):
            x[:,0:incontext_length,:] = (1 - (1 - self.sigma_min) * t) * noise[:,0:incontext_length,:] + t * incontext_x[:,0:incontext_length,:]
            if(guidance_scale > 1.0):
                dphi_dt = self.estimator( \
                    torch.cat([ \
                        torch.cat([x, x], 0), \
                        torch.cat([incontext_x, incontext_x], 0), \
                        torch.cat([torch.zeros_like(mu), mu], 0), \
                        ], 2), \
                timestep = t.unsqueeze(-1).repeat(2), \
                added_cond_kwargs={k:torch.cat([v,v],0) for k,v in added_cond_kwargs.items()}).sample
                dphi_dt_uncond, dhpi_dt_cond = dphi_dt.chunk(2,0)
                dphi_dt = dphi_dt_uncond + guidance_scale * (dhpi_dt_cond - dphi_dt_uncond)
            else:
                dphi_dt = self.estimator(torch.cat([x, incontext_x, mu], 1), \
                timestep = t.unsqueeze(-1),
                added_cond_kwargs=added_cond_kwargs).sample

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mu, added_cond_kwargs, latent_masks, validation_mode=False):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats , T)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, T)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, T)
        """
        b = mu[0].shape[0]

        # random timestep
        if(validation_mode):
            t = torch.ones([b, 1, 1], device=mu[0].device, dtype=mu[0].dtype) * 0.5
        else:
            t = torch.rand([b, 1, 1], device=mu[0].device, dtype=mu[0].dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        # print('y ', y.shape)
        # print('t ', t.shape)
        out = self.estimator(
            torch.cat([y, *mu],2), 
            timestep = t.squeeze(-1).squeeze(-1),
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        weight = (latent_masks > 1.5).unsqueeze(-1).repeat(1, 1, out.shape[-1]).float() + (latent_masks < 0.5).unsqueeze(-1).repeat(1, 1, out.shape[-1]).float() * 0.01
        loss = F.mse_loss(out * weight, u * weight, reduction="sum") / weight.sum()
        return loss

class AudioThinking(nn.Module):
    def __init__(self, dim, interval, encoder_depth, whisper_fea_dim, llm_path, prompt_path, use_detokenizer = False, is_train=False):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, dim), requires_grad=True) # init the query token
        self.interval = interval
        self.whisper_fea_dim = whisper_fea_dim
        en_transformers = []
        power_normalized = True
        for _ in range(encoder_depth):
            en_transformers.append(TransformerBlock(dim, dim_heads = 128, causal = False, zero_init_branch_outputs = False if power_normalized else True, 
                                                 remove_norms = False, power_normalized = power_normalized, conformer = False, layer_scale = True, 
                                                 add_rope = True, attn_kwargs={'qk_norm': True},  ff_kwargs={'mult': 4, 'no_bias': False}, norm_kwargs = {'eps': 1e-2}))
        self.encoder_transformers = nn.Sequential(*en_transformers)
        self.semantic_merge_proj = nn.Linear(whisper_fea_dim + 1024, dim) # simple linear projection
        self.reasoning_vq = ResidualVQ(dim = dim,
                    codebook_size = 4096, decay = 0.9, # the exponential moving average decay, lower means the dictionary will change faster
                    commitment_weight = 1., threshold_ema_dead_code = 0.1,
                    use_cosine_sim = False, codebook_dim = 64, implicit_neural_codebook = False,
                    rotation_trick = True, num_quantizers= 8)
        self.down_sampling_layer_whisper = nn.Conv1d(in_channels=whisper_fea_dim, out_channels=whisper_fea_dim, kernel_size=2, stride=2, padding=0, bias=True)

class AudioDiffusion1D(nn.Module):
    def __init__(
        self,
        num_channels,
        pre_trained_model_name = 'whisper&bestrq',
        features_type = 'continuous',
        prompt_path = '',
        llm_path = '',
        best_rq_ckpt = '',
        reason_lm_path = '',
        vq_training = True,
        unet_model_name=None,
        unet_model_config_path=None,
        whisper_path=None,
        uncondition=True,
        fine_decoder=False,
        is_train = False,
        use_detokenizer = True,
        wav_lm_path = None,
        music_ssl_folder = None,
        device = None    
    ):
        super().__init__()
        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.uncondition = uncondition
        self.num_channels = num_channels
        self.features_type = features_type
        self.max_t_len = 30*50 # set the max audio seqence as 30 seconds
        self.sample_rate = 24000 # the sample rate for mu_encoder is 24k hz
        self.sq_codec_latent = 136
        self.is_train = is_train
        '''load the whisper, wavlm part, and BESQRQ model'''
        self.whisper_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        self.whisper_fea_dim = self.whisper_encoder.config.d_model
        self.llm_path = llm_path
        self.prompt_path = prompt_path
        self.use_detokenizer = use_detokenizer
        self.codec_dim = 768 # fixed value
        self.wavlm_fea_dim = 768
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False
        self.whisper_encoder = WhisperModel.from_pretrained(whisper_path).encoder.to(device)
        self.wavlm_encoder = AutoModel.from_pretrained(wav_lm_path).to(device) # 
        self.wavlm_transfer = torchaudio.transforms.Resample(24000, 16000)
        for param in self.wavlm_encoder.parameters():
            param.requires_grad = False
        self.pretrained_model = BESTRQ_Model(model_dir = music_ssl_folder, 
                                checkpoint_dir = best_rq_ckpt, output_features = features_type, layers = [4, 11]).to(device)
    
        for v in self.pretrained_model.parameters():
            v.requires_grad = False 
        
        '''Define the down-samping layers and semantic decoders'''
        self.d_conv_whisper = nn.Conv1d(in_channels=self.whisper_fea_dim, out_channels=self.whisper_fea_dim, kernel_size=4,
                                        stride=4, padding=0, bias=True)
        self.d_conv_wavlm = nn.Conv1d(in_channels=self.wavlm_fea_dim, out_channels=self.wavlm_fea_dim, kernel_size=4,
                                        stride=4, padding=0, bias=True)
        self.d_conv_embedding_semantic = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=2,
                                        stride=2, padding=0, bias=True)
        self.d_conv_embedding_acoustic = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=2,
                                        stride=2, padding=0, bias=True)
        self.structure_semantic_decoder = Decoder(code_dim=self.codec_dim, output_channels=1024, decode_channels=1024, strides=[1, 2])
        self.pronunciation_decoder = Decoder(code_dim=self.codec_dim, output_channels=self.wavlm_fea_dim, decode_channels=self.wavlm_fea_dim, strides=[2, 2])
        
        '''Define three level VQ module'''
        self.vq_acoustic = ResidualVQ(
                dim = self.codec_dim, codebook_size = 8192, decay = 0.9, commitment_weight = 1.,   
                threshold_ema_dead_code = 2, use_cosine_sim = False, codebook_dim = 32, num_quantizers= 6)
        self.vq_structure_semantic = ResidualVQ(
            dim = self.codec_dim, codebook_size = 8192, decay = 0.9,  commitment_weight = 1.,   
            threshold_ema_dead_code = 2, use_cosine_sim = False, codebook_dim = 32, num_quantizers= 1)
        self.vq_pronunciation_semantic = ResidualVQ(
            dim = self.codec_dim, codebook_size = 8192, decay = 0.9, commitment_weight = 1.,   
            threshold_ema_dead_code = 2, use_cosine_sim = False, codebook_dim = 32, num_quantizers= 1)
        if fine_decoder:
            self.fix_module(self.d_conv_whisper)
            self.fix_module(self.d_conv_wavlm)
            self.fix_module(self.d_conv_embedding_semantic)
            self.fix_module(self.d_conv_embedding_acoustic)
            self.fix_module(self.structure_semantic_decoder)
            self.fix_module(self.pronunciation_decoder)
            self.fix_module(self.vq_acoustic)
            self.fix_module(self.vq_structure_semantic)
            self.fix_module(self.vq_pronunciation_semantic)

        '''Load the 1D flow-matching decoder'''
        if unet_model_config_path:
            self.cond_fusion_layer_semantic = nn.Linear(1024, self.codec_dim)
            self.cond_fusion_layer_acoustic = nn.Linear(1024+self.whisper_fea_dim, self.codec_dim)
            self.cond_fusion_layer_phone = nn.Linear(self.wavlm_fea_dim, self.codec_dim)

            self.time_film_phone    = self.make_time_film_head(768, self.codec_dim)
            self.time_film_semantic = self.make_time_film_head(768, self.codec_dim)
            self.time_film_acoustic = self.make_time_film_head(768, self.codec_dim)
            self.gamma = 0.1

            if fine_decoder:
                self.fix_module(self.cond_fusion_layer_semantic)
                self.fix_module(self.cond_fusion_layer_acoustic)
                self.fix_module(self.cond_fusion_layer_phone)
            
            self.reason_adaptor = nn.Linear(self.codec_dim, self.codec_dim) # new add
            self.cond_feature_emb = nn.Linear(self.codec_dim, self.codec_dim)
            self.zero_cond_embedding1 = nn.Parameter(torch.randn(self.codec_dim))
            unet = Transformer1DModel.from_config(unet_model_config_path)
            self.set_from = "random"
            self.cfm_wrapper = BASECFM(unet)
            if not use_detokenizer:
                print('delete detokenizer during the audio tokenization stage')
                del self.cfm_wrapper

        '''For the audio thinking part'''
        self.audio_thinking = AudioThinking(dim=self.codec_dim, interval=5, encoder_depth=5, whisper_fea_dim=self.whisper_fea_dim, llm_path=llm_path, prompt_path=prompt_path, use_detokenizer=use_detokenizer, is_train=is_train)
        # self.load_parameter_audio_thinking(reason_lm_path) # load pre-trained model
        for param in self.audio_thinking.parameters(): # fix the audio thinker
            param.requires_grad = False
        self.end_sym = '<|end_of_text|>'
        self.max_txt_len = 512 # we set the max txt length as 512
    
    def make_time_film_head(self, D, C):
        return nn.Linear(D, 2 * C)

    def fix_module(self, model_name):
        for v in model_name.parameters():
            v.requires_grad = False 

    def load_parameter_audio_thinking(self, reason_lm_path):
        '''load the audio reasoning model from the ckpt'''
        load_state(self.audio_thinking, reason_lm_path)
        # verify_state(self.audio_thinking, reason_lm_path, device="cpu")
    
    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.cfm_wrapper.estimator.transformer_blocks) 

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def get_whisper_feature(self, mels, n_len, len_semantic):
        with torch.no_grad():
            # print('mel ', mel.shape)
            n_len = int((n_len/24000)*50)
            n_len = max(n_len, len_semantic*2)
            # features = self.whisper_encoder.encoder(mel)
            whisper_embeds = self.whisper_encoder(mels, return_dict=True).last_hidden_state # get whisper features
            # print('whisper_embeds ', whisper_embeds.shape)
            # print('n_len ', n_len)
            return whisper_embeds[:,:n_len,:].transpose(1,2) # b, t, 512

    def L2_loss(self, feature, target_feature):
        """
        feature: B, T, D
        target_feature: B, T ,D
        return the mse loss
        """
        n = min(feature.size(1), target_feature.size(1))
        return F.mse_loss(target_feature[:,:n], feature[:,:n])


    def get_wavlm_feature(self, wav_24k, len_semantic):
        ''' wav: (B, 1, T)
            return: B, T//320, self.wavlm_dim*2
        '''
        wav_16k = self.wavlm_transfer(wav_24k).squeeze(1)
        wav_16k = torch.cat([wav_16k, torch.zeros(wav_16k.shape[0], 160).to(wav_16k.device)], dim=-1) # to make sure the frame problem
        target = self.wavlm_encoder(wav_16k, output_hidden_states=True).hidden_states
        target = torch.stack(target, dim=1)
        target = target[:,6:10,:].mean(1).transpose(1,2) # we choose the 6~9 layers as the conidtional phone-level information
        n_len = target.shape[-1]
        n_len = min(n_len, len_semantic*2) # 
        return target[:,:,:n_len]

    def get_reasoning_prompt(self, tasks):
        prompt = [self.audio_thinking.prompt_dict[task][-1] for task in tasks]
        return prompt
    
    def encode_reasoning_part(self, whisper_embeds, muencoder_embeds):
        # print('whisper_embeds ', whisper_embeds.shape)
        '''whisper_embeds (B, D, T)'''
        # print('whisper_embeds 0', whisper_embeds.shape, muencoder_embeds.shape)
        whisper_embeds = self.audio_thinking.down_sampling_layer_whisper(whisper_embeds) # down-samping using meaning pooling
        whisper_embeds = whisper_embeds.transpose(1,2)
        muencoder_embeds = muencoder_embeds.transpose(1,2)
        min_len = min(whisper_embeds.shape[1], muencoder_embeds.shape[1])
        embeds = torch.cat((whisper_embeds[:,:min_len,:], muencoder_embeds[:,:min_len,:]), dim=-1) # concatenate the two features
        # print('whisper_embeds ', whisper_embeds.shape)
        # print('embeds ', embeds.shape)
        embeds_llm = self.audio_thinking.semantic_merge_proj(embeds) # merge two features
        query_embeds_llm = self.set_masking(embeds_llm) # add the query token
        query_embeds_llm = self.audio_thinking.encoder_transformers(query_embeds_llm)
        query_tokens = self.extract_mask_positions(query_embeds_llm)
        # we should first mapping, then we quantize it
        quantized_features, indices, commitment_loss = self.audio_thinking.reasoning_vq(query_tokens)

        return quantized_features, indices, commitment_loss
    

    def reasoning_loss(self, lm_embeds, atts, texts, device):
        text = [t + self.end_sym for t in texts]
        to_regress_tokens = self.audio_thinking.llama_tokenizer(
            text, return_tensors="pt", padding="longest", truncation=True,
            max_length=self.max_txt_len, add_special_tokens=False ).to(device)
        to_regress_embeds = self.audio_thinking.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(to_regress_tokens.input_ids == self.audio_thinking.llama_tokenizer.pad_token_id, -100)
        empty_targets = (
            torch.ones(
                [atts.shape[0], atts.shape[1] + 1],
                dtype=torch.long
            ).to(device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        batch_size = lm_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.audio_thinking.llama_tokenizer.bos_token_id
        bos_embeds = self.audio_thinking.llama_model.model.model.embed_tokens(bos)
        atts_bos = atts[:, :1]
        inputs_embeds = torch.cat([bos_embeds, lm_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts, to_regress_tokens.attention_mask], dim=1)

        # calulate loss
        # with self.maybe_autocast():
        outputs = self.audio_thinking.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        return outputs.loss

    def time_film(self, cond_seq, features_BTC, layer):
        B, T, C = features_BTC.shape
        params = layer(cond_seq)                      # (B, T, 2C)
        delta_gamma, beta = params.chunk(2, dim=-1)
        
        gamma = 1.0 + self.gamma * delta_gamma.tanh()      # 幅度再小一点更稳
        # === 20% 概率 zero-condition ===
        mask = (torch.rand(B, 1, 1, device=cond_seq.device) < 0.2).float()
        gamma = gamma * (1 - mask) + 1.0 * mask
        beta  = beta  * (1 - mask) + 0.0 * mask
        return gamma * features_BTC + beta

    def feature_combine(self, reasoning_feaure, rec_feature):
        '''
        reasoning_feaure: (B, T_q, D)
        rec_feature: (B, T, D)
        Goal: upsample query_quantized to T, then combine with feature_quantized
        '''
        B, T, D = rec_feature.shape
        _, T_q, _ = reasoning_feaure.shape
        reasoning_feaure = self.reason_adaptor(reasoning_feaure)
        # repeat query token to match frame-level resolution
        reasoning_feaure = F.interpolate(reasoning_feaure.permute(0, 2, 1), scale_factor=2.5, mode='nearest').permute(0, 2, 1)
        # handle boundary if T is not divisible by T_q
        if reasoning_feaure.shape[1] != T:
            reasoning_feaure = reasoning_feaure[:, :T, :]  # or pad if needed
        # combine — e.g., element-wise addition or concat
        fused = rec_feature + reasoning_feaure  # or torch.cat([...], dim=-1)
        return fused

    def set_masking(self, x):
        """
        Perform per-sample random masking 
        x: [N, T, D], sequence, we put mask token 
        we set the mask rate as 0.8, so we put a mask every 5 frames. And we sure the total frame is 5 times
        """
        B, T, D = x.shape  # batch, length, dim
        # print('x ', x.shape, self.audio_thinking.interval)
        cls_token = self.audio_thinking.cls_token.repeat(1, (T//self.audio_thinking.interval), 1) # repeat 
        mask = cls_token.repeat(B, 1, 1)  # 扩展mask token以匹配batch size
        new_T = T + (T // self.audio_thinking.interval)
        x_reshaped = x.reshape(B, T // self.audio_thinking.interval, self.audio_thinking.interval, D)
        
        mask_tokens = mask.unsqueeze(2) #.repeat(1, T // self.interval, 1, 1) # B, a,1,D

        x_with_masks = torch.cat([x_reshaped, mask_tokens], dim=2)

        new_x = x_with_masks.reshape(B, -1, D)

        return new_x

    def extract_mask_positions(self, new_x):
        B, new_T, D = new_x.shape
        original_T = new_T - new_T // (self.audio_thinking.interval + 1)
        
        mask_indices = [(i + 1) * (self.audio_thinking.interval + 1) - 1 for i in range(original_T // self.audio_thinking.interval)]
        mask_positions = new_x[:, mask_indices, :]

        return mask_positions

    def init_device_dtype(self, device, dtype):
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def fetch_codes_batch(self, input_audios, spectrograms, additional_feats=None, return_reasoning_text=False):
        ''' if return_reasoning_text is False: additional_feats is None
            if return_reasoning_text is true, additional_feats should includes text and task?
            But for our final version, the text and task will be fixed.
        '''
        input_audio = input_audios[:,0,:]
        self.pretrained_model.eval()
        self.wavlm_encoder.eval()
        self.whisper_encoder.eval()
        with torch.cuda.amp.autocast(enabled=False):
            bestrq_emb_acoustic, bestrq_emb_semantic = self.pretrained_model.extract_continous_embeds_multiple(input_audios.clone())
        bestrq_emb_acoustic = bestrq_emb_acoustic.detach()
        bestrq_emb_semantic = bestrq_emb_semantic.detach()
        semantic_target = bestrq_emb_acoustic.clone()
        whisper_embeds = self.get_whisper_feature(spectrograms, input_audios.shape[-1], bestrq_emb_semantic.shape[2])
        whisper_embeds = whisper_embeds.detach()
        wavlm_embeds = self.get_wavlm_feature(input_audios, bestrq_emb_semantic.shape[2]) # 
        wavlm_embeds = wavlm_embeds.detach()

        '''For the semantic-rich reconstruction branch'''

        wavlm_target = wavlm_embeds.clone()
        whisper_embeds_rec = self.d_conv_whisper(whisper_embeds.clone()) #.transpose(1,2)
        wavlm_encoder_features = self.d_conv_wavlm(wavlm_embeds) 
        features_emb_semantic_rec = self.d_conv_embedding_semantic(bestrq_emb_semantic.clone())
        features_emb_acoustic = self.d_conv_embedding_acoustic(bestrq_emb_acoustic)

        #quantized_reasoning, reasoning_codes, lm_embeds, atts, commitment_loss_reason = self.encode_reasoning_part(whisper_embeds, bestrq_emb_semantic)
        quantized_reasoning, reasoning_codes, lm_embeds = self.encode_reasoning_part(whisper_embeds.clone(), bestrq_emb_semantic.clone())  
        reasoning_feaure = self.reason_adaptor(quantized_reasoning)
        reasoning_feaure = F.interpolate(reasoning_feaure.permute(0, 2, 1), scale_factor=2.5, mode='nearest').permute(0, 2, 1)

        '''The pronunciation semantic branch module'''
        features_emb_phone = self.cond_fusion_layer_phone(wavlm_encoder_features.transpose(1,2)).transpose(1,2)
        self.vq_pronunciation_semantic.eval()
        features_emb_phone = self.time_film(reasoning_feaure.clone(), features_emb_phone.transpose(1,2), self.time_film_phone).transpose(1,2)
        quantized_feature_emb_phone, codes_phone, commitment_loss_feature_emb_phone = self.vq_pronunciation_semantic(features_emb_phone.transpose(1, 2))

        '''The structure semantic branch module'''
        features_emb_semantic_rec = self.cond_fusion_layer_semantic(features_emb_semantic_rec.transpose(1, 2)).transpose(1, 2)
        self.vq_structure_semantic.eval()
        features_emb_semantic_rec = self.time_film(reasoning_feaure.clone(), features_emb_semantic_rec.transpose(1, 2), self.time_film_semantic).transpose(1,2)
        quantized_feature_emb_semantic, codes_semantic, commitment_loss_feature_emb_semantic = self.vq_structure_semantic(features_emb_semantic_rec.transpose(1, 2))  #
        
        
        '''The acoustic branch module: concat the whisper feature and muencoder feature'''
        min_f_len = min(features_emb_acoustic.shape[-1], whisper_embeds_rec.shape[-1])
        features_emb_acoustic = torch.cat([features_emb_acoustic[:, :, :min_f_len], whisper_embeds_rec[:, :, :min_f_len]], dim=1)  # concat 
        features_emb_acoustic = self.cond_fusion_layer_acoustic(features_emb_acoustic.transpose(1, 2)).transpose(1, 2)
        self.vq_acoustic.eval()
        features_emb_acoustic = self.time_film(reasoning_feaure, features_emb_acoustic.transpose(1, 2), self.time_film_acoustic).transpose(1,2)
        quantized_feature_emb_acoustic, codes_acoustic, commitment_loss_feature_emb_acoustic = self.vq_acoustic(features_emb_acoustic.transpose(1, 2))  #
        quantized_feature_emb = quantized_feature_emb_phone + quantized_feature_emb_semantic + quantized_feature_emb_acoustic

        merge_features = self.cond_feature_emb(quantized_feature_emb) # b t 512
        # print('codes_acoustic ', codes_acoustic.shape)
        # assert 1==2
        merge_codes = torch.cat([codes_phone, codes_semantic, codes_acoustic], dim=-1)
        return [reasoning_codes], [merge_codes], [merge_features]

    @torch.no_grad()
    def inference_codes(self, codes, spk_embeds, true_latents, latent_length, incontext_length, additional_feats, 
                  guidance_scale=2, num_steps=20, disable_progress=True, scenario='start_seg'):
        classifier_free_guidance = guidance_scale > 1.0
        device = self.device
        dtype = self.dtype
        # print('codes ', codes)
        if len(codes) == 2:
            codes_reasoning = codes[0] # reasoning tokens
            # print('codes[1] ', codes[1].shape)
            # assert 1==2
            codes_phone = codes[1][:,0:1,:] # phone tokens
            codes_semantic = codes[1][:,1:2,:] # semantic tokens
            codes_acoustic = codes[1][:,2:,:] # acoustic
        else:
            codes_reasoning = None
            codes_phone = codes[0][:,0:1,:] # phone tokens
            codes_semantic = codes[0][:,1:2,:] # semantic tokens
            codes_acoustic = codes[0][:,2:,:] # acoustic
        
        batch_size = codes_phone.shape[0]
        self.vq_pronunciation_semantic.eval()
        self.vq_structure_semantic.eval()
        self.vq_acoustic.eval()
        quantized_feature_emb_phone = self.vq_pronunciation_semantic.get_output_from_indices(codes_phone.transpose(1, 2))
        quantized_feature_emb_semantic = self.vq_structure_semantic.get_output_from_indices(codes_semantic.transpose(1, 2))
        quantized_feature_emb_acoustic = self.vq_acoustic.get_output_from_indices(codes_acoustic.transpose(1, 2))
        quantized_feature_emb = quantized_feature_emb_phone + quantized_feature_emb_semantic + quantized_feature_emb_acoustic
        
        if codes_reasoning is not None:
            quantized_feature_reasoning = self.audio_thinking.reasoning_vq.get_output_from_indices(codes_reasoning.transpose(1, 2))
            merge_features = self.feature_combine(quantized_feature_reasoning, quantized_feature_emb)
        else:
            merge_features = quantized_feature_emb
        
        merge_features = self.cond_feature_emb(merge_features) # b t 512
        merge_features = F.interpolate(merge_features.permute(0, 2, 1), scale_factor=2, mode='nearest').permute(0, 2, 1)
        # latents: B, 2*T, 64:  b, 2*T, 512/2 
        B, T, D = merge_features.shape
        num_frames = merge_features.shape[1] # 

        latents = self.prepare_latents(batch_size, num_frames, dtype, device) # prepapre the latent shap
        bsz, T, dim = latents.shape
        resolution = torch.tensor([T, 1]).repeat(bsz, 1) # ?
        aspect_ratio = torch.tensor([float(T/self.max_t_len)]).repeat(bsz, 1)
        resolution = resolution.to(dtype=merge_features.dtype, device=device)
        aspect_ratio = aspect_ratio.to(dtype=merge_features.dtype, device=device)
        if classifier_free_guidance:
            resolution = torch.cat([resolution, resolution], 0)
            aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], 0)
        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        latent_masks = torch.zeros(latents.shape[0], latents.shape[1], dtype=torch.int64, device=latents.device)
        latent_masks[:,0:latent_length] = 2
        if(scenario=='other_seg'):
            latent_masks[:,0:incontext_length] = 1

        merge_features = (latent_masks > 0.5).unsqueeze(-1) * merge_features \
            + (latent_masks < 0.5).unsqueeze(-1) * self.zero_cond_embedding1.unsqueeze(0)
        
        incontext_latents = true_latents * ((latent_masks > 0.5) * (latent_masks < 1.5)).unsqueeze(-1).float()
        incontext_length = ((latent_masks > 0.5) * (latent_masks < 1.5)).sum(-1)[0]

        if('spk' in additional_feats):
            additional_model_input = torch.cat([merge_features, spk_embeds],1)
        else:
            additional_model_input = torch.cat([merge_features],1)
        temperature = 1.0
        t_span = torch.linspace(0, 1, num_steps + 1, device=merge_features.device)
        latents = self.cfm_wrapper.solve_euler(latents * temperature, incontext_latents, incontext_length, t_span, additional_model_input, added_cond_kwargs, guidance_scale)
        latents[:,0:incontext_length,:] = incontext_latents[:,0:incontext_length,:] # B, T, dim
        return latents

    def infer_llm(self, lm_embeds, atts, prompts, generate_cfg):
        if prompts is not None:
            lm_embeds, atts = self.prompt_wrap(lm_embeds, atts, prompts)
        bos = torch.ones([lm_embeds.shape[0], 1], dtype=torch.int32, device=lm_embeds.device) * self.audio_thinking.llama_tokenizer.bos_token_id
        bos_embeds = self.audio_thinking.llama_model.model.model.embed_tokens(bos)
        atts_bos = atts[:, :1]
        embeds = torch.cat([bos_embeds, lm_embeds], dim=1)
        attns = torch.cat([atts_bos, atts], dim=1)
        stop_words_ids = [torch.tensor([128001]).to(embeds.device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.audio_thinking.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
            pad_token_id = self.audio_thinking.llama_tokenizer.pad_token_id,
        )
        text = self.audio_thinking.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return text
    
    def prepare_latents(self, batch_size, num_frames, dtype, device):
        shape = (batch_size, num_frames, self.sq_codec_latent)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        return latents

