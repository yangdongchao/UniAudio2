import os
import sys
from omegaconf import OmegaConf
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from tools.tokenizer.abs_tokenizer import AbsTokenizer
import torchaudio
from tools.tokenizer.ReasoningCodec_film.models.AudioDiffusion1D import AudioDiffusion1D, AudioThinking
from tools.tokenizer.ReasoningCodec_film.models.model_utils import load_state
from tools.tokenizer.common import VolumeNorm
from tools.tokenizer.ReasoningCodec_film.models.scalar24k import ScalarAE
from transformers import WhisperFeatureExtractor
import yaml
import argparse
import os 
import glob
import math
import numpy as np 

class ReasoningTokenizer(AbsTokenizer):
    def __init__(self, train_config, model_path, music_ssl_folder, device=torch.device('cpu')):
        super(ReasoningTokenizer, self).__init__()
        with open(train_config, "r", encoding="utf-8") as f:
            train_args = yaml.safe_load(f)
            train_args = argparse.Namespace(**train_args)
        self.sample_rate = 24000
        self.device = device
        self.MAX_DURATION = 360
        self.n_codebook = 8
        self.sq_codec_hz = 25 # the frame-rate of SQCodec
        self.rec_frame_rate = 12.5 # 
        self.reason_frame_rate = 5 
        self.SQCodec = ScalarAE(scalar_config=train_args.sq_config, resume_path=train_args.sq_resume)
        self.SQCodec = self.SQCodec.eval().to(device)
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(train_args.whisper_path)
        self.transfer16k = torchaudio.transforms.Resample(24000, 16000).to(device)
        self.model = AudioDiffusion1D(
            num_channels= train_args.num_channels,
            pre_trained_model_name = 'whisper&bestrq',
            features_type = 'continuous',
            vq_training = True,
            unet_model_name = train_args.unet_model_name,
            unet_model_config_path = train_args.transformer_diffusion_config,
            whisper_path = train_args.whisper_path,
            reason_lm_path = train_args.reason_lm_path,
            best_rq_ckpt = train_args.best_rq_ckpt,
            llm_path = train_args.llm_path ,
            prompt_path = train_args.prompt_path,
            uncondition = True,
            use_detokenizer = True,
            wav_lm_path = train_args.wav_lm_path,
            music_ssl_folder = music_ssl_folder,
            device = self.device
        )
        if model_path is not None:
            state_dict = torch.load(model_path, map_location='cpu')['model']
            state_dict = { k.split("module.")[-1] if k.startswith("module.") else k: v
                        for k, v in state_dict.items() }
            self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device)
        print ("Successfully loaded checkpoint from:", model_path)
        self.model.eval()
        self.model.init_device_dtype(torch.device(device), torch.float32)
        self.volume_norm = VolumeNorm(params=[-16, 3], sample_rate=24000, energy_threshold=1e-6)
    
    def get_whisper_features(self, audio, sr):
        if sr != 16000:
            audio = self.transfer16k(audio)
            sr = 16000
        spectrogram = self.wav_processor(audio.detach().cpu().numpy(), sampling_rate=16000, return_tensors="pt")["input_features"] # B, 80, t
        return spectrogram

    def find_length(self, x):
        '''x: (RVQ_num, T)'''
        return x.shape[1]

    def tokenize2(self, token):
        '''Token: '''
        if isinstance(token, torch.Tensor):
            return token.to(torch.int64).transpose(0, 1) # transfer to (T, 8)
        else:
            raise NotImplementedError
    
    @torch.no_grad()
    def audio2token(self, orig_samples, sr, return_reasoning_text=False, 
                            task_name='speech_reasoning', min_duration=30, batch_size=6):
        if(orig_samples.ndim == 2):
            audios = orig_samples.unsqueeze(0).to(self.device)
        elif(orig_samples.ndim == 3):
            audios = orig_samples.to(self.device)
        else:
            assert orig_samples.ndim in (2,3), orig_samples.shape
        
        audios = audios.squeeze(0) # change to 2-dim. (channel, time)
        orig_length = audios.shape[-1]
        min_samples = int(min_duration * self.sample_rate) # the min_segment

        output_len = int(orig_length / float(self.sample_rate) * self.rec_frame_rate) + 1 # get the number of token
        output_len_reason = int(orig_length / float(self.sample_rate) * self.reason_frame_rate) + 1
        while(audios.shape[-1] < min_samples + 240): # add 240, to make sure the last frame is completed
            audios = torch.cat([audios, audios], -1)

        int_max_len = audios.shape[-1]//min_samples+1 # the max segment number
        audios = torch.cat([audios, audios], -1)
        audios=audios[:,:int(int_max_len*(min_samples+240))] 
        reason_codes_list=[]
        rec_codes_list=[]
        audio_input = audios.reshape(1, -1, min_samples+240).permute(1, 0, 2).reshape(-1, 1, min_samples+240)
        for audio_inx in range(0, audio_input.shape[0], batch_size):
            mels = self.get_whisper_features(audio_input[audio_inx:audio_inx+batch_size,0,:], 24000).to(self.device)
            if return_reasoning_text:
                additional_feats = [task_name]*audio_input.shape[0]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    reasoning_text, reasoning_codes, rec_codes, _ = self.model.fetch_codes_batch((audio_input[audio_inx:audio_inx+batch_size]), mels, additional_feats=additional_feats, return_reasoning_text=return_reasoning_text)
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    reasoning_codes, rec_codes, _ = self.model.fetch_codes_batch((audio_input[audio_inx:audio_inx+batch_size]), mels, additional_feats=[], return_reasoning_text=return_reasoning_text)
                reasoning_text = None
            reason_codes_list.append(torch.cat(reasoning_codes, 1))
            rec_codes_list.append(torch.cat(rec_codes, 1))

        reason_codec = torch.cat(reason_codes_list, 0)
        rec_codec = torch.cat(rec_codes_list, 0)
        reason_codec = reason_codec.reshape(-1, 8).unsqueeze(0)
        rec_codec = rec_codec.reshape(-1, 8).unsqueeze(0)
        rec_codec = rec_codec[:,:output_len,:].transpose(1, 2)
        reason_codec = reason_codec[:,:output_len_reason,:].transpose(1, 2)
        return reason_codec, rec_codec # (1, T, B)
    
    @torch.no_grad()
    def token2audio(self, reason_codec, rec_codec, return_reasoning_text, 
                          duration=30, guidance_scale=1.5, num_steps=10, disable_progress=False):
        ''' reason_codec: (B, 8, T)
            rec_codec: (B, 8, T2)
            first_latent: the latent representation for SQCodec
        '''
        rec_codec = rec_codec.to(self.device)
        reason_codec = reason_codec.to(self.device)
        first_latent = torch.randn(rec_codec.shape[0], int(duration*25), 136).to(self.device) # B, T, 64
        first_latent_length = 0
        first_latent_codes_length = 0
        
        min_samples = int(duration*self.rec_frame_rate) # get the min 
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        hop_frames = hop_samples // 2
        ovlp_frames = ovlp_samples // 2
        
        min_samples_q  = int(duration * self.reason_frame_rate)
        hop_samples_q   = min_samples_q // 4 * 3  
        ovlp_samples_q  = min_samples_q - hop_samples_q

        rec_codes_len= rec_codec.shape[-1] #
        codec_q_len = reason_codec.shape[-1]
        target_len = int((rec_codes_len - first_latent_codes_length) / 12.5 * self.sample_rate)
        if(rec_codes_len < min_samples):
            while(rec_codec.shape[-1] < min_samples):
                rec_codec = torch.cat([rec_codec, rec_codec], -1)
            rec_codec = rec_codec[:,:,0:min_samples]
        if (codec_q_len < min_samples_q):
            while(reason_codec.shape[-1] < min_samples_q):
                reason_codec = torch.cat([reason_codec, reason_codec], -1)
            reason_codec = reason_codec[:,:,0:min_samples_q]
        rec_codes_len = rec_codec.shape[-1]
        codec_q_len = reason_codec.shape[-1]

        if((rec_codes_len - ovlp_samples) % hop_samples > 0):
            len_codes=math.ceil((rec_codes_len - ovlp_samples) / float(hop_samples)) * hop_samples + ovlp_samples
            while(rec_codec.shape[-1] < len_codes):
                rec_codec = torch.cat([rec_codec, rec_codec], -1)
            rec_codec = rec_codec[:,:,0:len_codes]
        
        if((codec_q_len - ovlp_samples_q) % (hop_samples_q) > 0):
            len_codes_q = math.ceil((codec_q_len - ovlp_samples_q) / float(hop_samples_q)) * hop_samples_q + ovlp_samples_q
            while(reason_codec.shape[-1] < len_codes_q):
                reason_codec = torch.cat([reason_codec, reason_codec], -1)
            reason_codec = reason_codec[:,:,0:len_codes_q]

        latent_length = int(duration*self.sq_codec_hz)
        latent_list = []
        spk_embeds = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cnt = 0
            for sinx in range(0, rec_codec.shape[-1]-hop_samples, hop_samples):
                codes_input = []
                sinx_q = hop_samples_q * cnt
                codes_input.append(reason_codec[:,:,sinx_q:sinx_q+min_samples_q])
                codes_input.append(rec_codec[:,:,sinx:sinx+min_samples])

                if(sinx == 0):
                    incontext_length = first_latent_length
                    latents = self.model.inference_codes(codes_input, spk_embeds, first_latent, latent_length, incontext_length, 
                                  additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
                    latent_list.append(latents)
                else:
                    true_latent = latent_list[-1][:,-ovlp_frames:,:]
                    len_add_to_latent = latent_length - true_latent.shape[1] # 
                    incontext_length = true_latent.shape[1]
                    true_latent = torch.cat([true_latent, torch.randn(true_latent.shape[0], len_add_to_latent, true_latent.shape[-1]).to(self.device)], 1)
                    latents = self.model.inference_codes(codes_input, spk_embeds, true_latent, latent_length, incontext_length, return_reasoning_text=return_reasoning_text,
                                   additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
                    latent_list.append(latents)
        
        latent_list = [l.float() for l in latent_list]
        latent_list[0] = latent_list[0][:,first_latent_length:,:]
        min_samples =  int(duration * self.sample_rate)
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        with torch.no_grad():
            output = None
            for i in range(len(latent_list)):
                latent = latent_list[i]
                bsz , t, f = latent.shape
                cur_output = self.SQCodec.decode(latent.transpose(1,2)).squeeze(0)
                cur_output = cur_output[:, 0:min_samples].detach().cpu() # B, T

                if output is None:
                    output = cur_output
                else:
                    ov_win = torch.from_numpy(np.linspace(0, 1, ovlp_samples)[None, :])
                    ov_win = torch.cat([ov_win, 1 - ov_win], -1)
                    output[:, -ovlp_samples:] = output[:, -ovlp_samples:] * ov_win[:, -ovlp_samples:] + cur_output[:, 0:ovlp_samples] * ov_win[:, 0:ovlp_samples]
                    output = torch.cat([output, cur_output[:, ovlp_samples:]], -1)
            output = output[:, 0:target_len]
        return output

    @torch.no_grad()
    def token2audio_no_reason(self, rec_codec, return_reasoning_text, duration=20, guidance_scale=1.5, num_steps=20, disable_progress=False):
        ''' 
            rec_codec: (B, 8, T2)
            first_latent: the latent representation for SQCodec
        '''
        rec_codec = rec_codec.to(self.device)
        first_latent = torch.randn(rec_codec.shape[0], int(duration*25), 136).to(self.device) # B, T, 64
        first_latent_length = 0
        first_latent_codes_length = 0
        
        min_samples = int(duration*self.rec_frame_rate) # get the min 
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        hop_frames = hop_samples // 2
        ovlp_frames = ovlp_samples // 2
        
        min_samples_q  = int(duration * self.reason_frame_rate)
        hop_samples_q   = min_samples_q // 4 * 3  
        ovlp_samples_q  = min_samples_q - hop_samples_q

        rec_codes_len= rec_codec.shape[-1] #
        target_len = int((rec_codes_len - first_latent_codes_length) / 12.5 * self.sample_rate)
        if(rec_codes_len < min_samples):
            while(rec_codec.shape[-1] < min_samples):
                rec_codec = torch.cat([rec_codec, rec_codec], -1)
            rec_codec = rec_codec[:,:,0:min_samples]
        rec_codes_len = rec_codec.shape[-1]
        if((rec_codes_len - ovlp_samples) % hop_samples > 0):
            len_codes=math.ceil((rec_codes_len - ovlp_samples) / float(hop_samples)) * hop_samples + ovlp_samples
            while(rec_codec.shape[-1] < len_codes):
                rec_codec = torch.cat([rec_codec, rec_codec], -1)
            rec_codec = rec_codec[:,:,0:len_codes]
        
        latent_length = int(duration*self.sq_codec_hz)
        latent_list = []
        spk_embeds = None
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            cnt = 0
            for sinx in range(0, rec_codec.shape[-1]-hop_samples, hop_samples):
                codes_input = []
                codes_input.append(rec_codec[:,:,sinx:sinx+min_samples])

                if(sinx == 0):
                    incontext_length = first_latent_length
                    latents = self.model.inference_codes(codes_input, spk_embeds, first_latent, latent_length, incontext_length, 
                                  additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
                    latent_list.append(latents)
                else:
                    true_latent = latent_list[-1][:,-ovlp_frames:,:]
                    len_add_to_latent = latent_length - true_latent.shape[1] # 
                    incontext_length = true_latent.shape[1]
                    true_latent = torch.cat([true_latent, torch.randn(true_latent.shape[0], len_add_to_latent, true_latent.shape[-1]).to(self.device)], 1)
                    latents = self.model.inference_codes(codes_input, spk_embeds, true_latent, latent_length, incontext_length,
                                   additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
                    latent_list.append(latents)
        
        latent_list = [l.float() for l in latent_list]
        latent_list[0] = latent_list[0][:,first_latent_length:,:]
        min_samples =  int(duration * self.sample_rate)
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        with torch.no_grad():
            output = None
            for i in range(len(latent_list)):
                latent = latent_list[i]
                bsz , t, f = latent.shape
                cur_output = self.SQCodec.decode(latent.transpose(1,2)).squeeze(0)
                cur_output = cur_output[:, 0:min_samples].detach().cpu() # B, T

                if output is None:
                    output = cur_output
                else:
                    ov_win = torch.from_numpy(np.linspace(0, 1, ovlp_samples)[None, :])
                    ov_win = torch.cat([ov_win, 1 - ov_win], -1)
                    output[:, -ovlp_samples:] = output[:, -ovlp_samples:] * ov_win[:, -ovlp_samples:] + cur_output[:, 0:ovlp_samples] * ov_win[:, 0:ovlp_samples]
                    output = torch.cat([output, cur_output[:, ovlp_samples:]], -1)
            output = output[:, 0:target_len]
        return output

    def wave_pad(self, wav):
        seq_len = wav.shape[-1]
        if seq_len % 9600 == 0:
            return wav
        pad_size = 9600 - (seq_len % 9600)
        padding_shape = list(wav.shape)
        padding_shape[-1] = pad_size
        padding = torch.zeros(*padding_shape, dtype=wav.dtype, device=wav.device)
        return torch.cat([wav, padding], dim=-1)

    def encode_segment(self, orig_samples, sr, return_reasoning_text=False, task_name='asr'):
        if(orig_samples.ndim == 2):
            audios = orig_samples.unsqueeze(0).to(self.device)
        elif(orig_samples.ndim == 3):
            audios = orig_samples.to(self.device)
        else:
            assert orig_samples.ndim in (2,3), orig_samples.shape
        orig_length = audios.shape[-1]
        audio_input = self.wave_pad(audios)
        output_len = int(orig_length / float(self.sample_rate) * self.rec_frame_rate) + 1
        output_len_reason = int(orig_length / float(self.sample_rate) * self.reason_frame_rate) + 1
        mels = self.get_whisper_features(audio_input.squeeze(0), 24000).to(self.device)
        if return_reasoning_text:
            additional_feats = [task_name]*audio_input.shape[0]
            reasoning_text, reasoning_codes, rec_codes, _  = self.model.fetch_codes_batch((audio_input), mels, additional_feats=additional_feats, return_reasoning_text=return_reasoning_text)
        else:
            reasoning_codes, rec_codes, _  = self.model.fetch_codes_batch((audio_input), mels, additional_feats=[], return_reasoning_text=return_reasoning_text)
            reasoning_text = None
        reasoning_codes = torch.cat(reasoning_codes, 1)
        rec_codes = torch.cat(rec_codes, 1)
        rec_codes = rec_codes[:,:output_len,:].transpose(1, 2)
        reasoning_codes = reasoning_codes[:,:output_len_reason,:].transpose(1, 2)
        # print('rec_codes ', rec_codes.shape)
        # print('reasoning_codes ', reasoning_codes.shape)
        # assert 1==2
        return reasoning_codes, rec_codes, reasoning_text
        

    def decode_segment(self, reason_codec, rec_codec, return_reasoning_text, 
                       guidance_scale=1.5, num_steps=20, disable_progress=False):
        ''' reason_codec: (B, 8, T); rec_codec: (B, 8, T2)
            first_latent: the latent representation for SQCodec
        '''
        pre_duration = (rec_codec.shape[-1]*1920) / self.sample_rate
        rec_codec = rec_codec.to(self.device)
        reason_codec = reason_codec.to(self.device)
        first_latent = torch.randn(rec_codec.shape[0], int(pre_duration*25), 136).to(self.device) # B, T, 64
        first_latent_length = 0
        first_latent_codes_length = 0

        spk_embeds = None
        latent_length = int(pre_duration*self.sq_codec_hz)
        incontext_length = 0 # no incontext
        codes_input = []
        codes_input.append(reason_codec)
        codes_input.append(rec_codec)

        latents = self.model.inference_codes(codes_input, spk_embeds, first_latent, latent_length, incontext_length, 
                                            additional_feats=[], guidance_scale=1.5, 
                                            num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
        cur_output = self.SQCodec.decode(latents.transpose(1,2)).squeeze(0)
        cur_output = cur_output.detach().cpu() # B, T
        return cur_output


    @property
    def is_discrete(self):
        return True
    
    def tokenize(self, wav, return_reasoning_text=False, task_name='asr', min_duration=30):
        ''' tokenize 时, 只支持wav_path output: (8,T)'''
        if isinstance(wav, str):
            prompt_audio, fs = torchaudio.load(wav) # read the file
            if prompt_audio.shape[0] == 2:
                prompt_audio = prompt_audio.mean(0, keepdim=True)
            if(fs!=self.sample_rate):
                prompt_audio = torchaudio.functional.resample(prompt_audio, fs, self.sample_rate)
                fs = self.sample_rate
            reason_codec, rec_codec = self.audio2token(prompt_audio, fs, return_reasoning_text, task_name=task_name) # [1, 4, T]
            return reason_codec.squeeze(0), rec_codec.squeeze(0) 
        elif isinstance(wav, torch.Tensor):
            return wav
        else:
            raise NotImplementedError
    
    def detokenize(self, reason_codec, rec_codec, return_reasoning_text, min_duration=30, steps=50, guidance_scale=1.5, disable_progress=False):
        '''reason_codec: 8,T1, rec_codec: (8, T2)'''
        wave = self.token2audio(reason_codec.unsqueeze(0), rec_codec.unsqueeze(0), return_reasoning_text=return_reasoning_text, 
                                duration=min_duration, guidance_scale=guidance_scale, num_steps=steps, disable_progress=disable_progress)
        return wave 

    def detokenize_no_reason(self, rec_codec, return_reasoning_text, min_duration=30, steps=50, guidance_scale=1.5, disable_progress=False):
        '''reason_codec: 8,T1, rec_codec: (8, T2)'''
        wave = self.token2audio_no_reason(rec_codec.unsqueeze(0), 
                   return_reasoning_text=return_reasoning_text, guidance_scale=guidance_scale, 
                   num_steps=steps, disable_progress=disable_progress)
        return wave 


if __name__ == '__main__':
    pass

