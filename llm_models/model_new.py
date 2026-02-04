''' Multi-scale audio language model using LLAMA3_2 as the backbone
Author: Dongchao Yang. 2025
Code based on:
https://github.com/yangdongchao/UniAudio
https://github.com/pytorch/torchtune/tree/main
https://github.com/SesameAILabs/csm

The key features:
(1) It is a causal model for both global and local transformer. So it can easily combine with any LLM
(2) 
for text sequence, the audio place is set as 0.
for audio sequence, the text place is set as 0.
We use mask token to solve the influence caused by these tokens
'''
from dataclasses import dataclass
import torch
import torch.nn as nn
import torchtune
from huggingface_hub import PyTorchModelHubMixin
from typing import List, Tuple
from torch.nn import functional as F
from llm_models.lit_model import GPT
from llm_models.config import Config as gpt_config
from llm_models.semantic_decoder import Decoder, FiLMEncoder

def select_with_fixed_mask(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """
    Args:
        x: (B, T) input tensor
        mask: (B, T) mask tensor (exactly k ones per row)
        k: number of selected elements per row
    Returns:
        (B, k) tensor of selected elements
    """
    indices = torch.nonzero(mask, as_tuple=True)[1].reshape(-1, k)
    return x.gather(1, indices)

def CrossEntropyAndAccuracy_zero(logits, y, mask, ignore_id=0):
    ''' 
    The zero layer loss.
    logits: [B, T, K]
    y: (B, T), mask (B, T)
    '''
    y, mask = y.to(logits.device), mask.to(logits.device)
    #print('ignore_id ', ignore_id, y, y.shape)
    # print('logits ', logits)
    # assert 1==2
    loss = F.cross_entropy(logits.transpose(1, 2).contiguous(), y.contiguous(), ignore_index=ignore_id, reduction='none')
    # print('loss ', loss)
    # assert 1==2
    pred = logits.argmax(2)
    num_all_tokens = mask.int().sum()
    
    acc = torch.logical_and(pred.eq(y), mask).int().sum() / num_all_tokens
    loss = (loss*mask).sum() / num_all_tokens
    metrics = {'acc_0': acc, 'loss_0': loss.clone().detach()}
    return loss, metrics

def CrossEntropyAndAccuracy_text(logits, y, mask, ignore_id=-1):
    ''' 
    logits: [B, T, K]
    y: (B, T), mask (B, T)
    '''
    y, mask = y.to(logits.device), mask.to(logits.device)
    loss = F.cross_entropy(logits.transpose(1, 2).contiguous(), y.contiguous(), ignore_index=ignore_id, reduction='none')
    
    pred = logits.argmax(2)
    num_all_tokens = mask.int().sum()
    
    acc = torch.logical_and(pred.eq(y), mask).int().sum() / num_all_tokens
    loss = (loss*mask).sum() / num_all_tokens
    metrics = {'acc_text': acc, 'loss_text': loss.clone().detach()}
    return loss, metrics

def CrossEntropyAndAccuracy_residual(logits, y, loss_mask, loss_weights=[1, 1, 1], ignore_id=None):
    """
       The residual layer loss. 
       loss_mask: (B, T, N)
       logits: (B, N, k)
       reserved_mask: (B, T)
       y: (B, T, N)
    """
    y = y.to(logits.device)
    loss_dict = {}
    acc = {}
    loss_avg = 0
    for idx, w in enumerate(loss_weights):

        tmp_logit = logits[:,idx,:].contiguous()

        tmp_y = y[:,idx].contiguous()
        tmp_loss = F.cross_entropy(tmp_logit, tmp_y, ignore_index=ignore_id, reduction='none')

        tmp_loss = tmp_loss*loss_mask[:,idx] # add loss mask
        
        tmp_pred = tmp_logit.argmax(1)
        tmp_num_all_tokens = tmp_y.shape[0]  # we only calculate the non-mask part
    
        tmp_acc_tk = tmp_pred.eq(tmp_y).int().sum()
        acc[f'acc_{idx+1}'] = tmp_acc_tk/tmp_num_all_tokens
        tmp_loss = tmp_loss.sum()/tmp_num_all_tokens
        loss_avg += tmp_loss*loss_weights[idx]
        loss_dict[f'loss_{idx+1}'] = tmp_loss.clone().detach()
        
    loss_avg = loss_avg/len(loss_weights)
    metrics = {}
    metrics.update(acc)
    metrics.update(loss_dict)
    return loss_avg, metrics

def _prepare_transformer(model):
    embed_dim = model.config.n_embd
    model.transformer.wte = nn.Identity()
    model.lm_head = nn.Identity()
    return model, embed_dim

def _prepare_llm_transformer(model):
    '''preserve all of the llm --> we need to load the pre-trained LM'''
    embed_dim = model.config.n_embd
    return model, embed_dim

def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

def _create_causal_mask_train(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(1, seq_len, seq_len, dtype=torch.bool, device=device))

def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """ get the expected causal mask based on the position input
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(probs):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token

def audio_sample_topk(logits: torch.Tensor, topk: int, temperature: float, forbid_prefix: int = 0):
    """
    logits: (batch, vocab) or (..., vocab)
    topk: int
    temperature: float > 0
    forbid_prefix: int, number of initial token indices [0, forbid_prefix-1] that are forbidden
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if forbid_prefix < 0:
        raise ValueError("forbid_prefix must be >= 0")
    logits = logits.clone() / temperature  # clone to avoid in-place changes outside
    vocab_size = logits.size(-1)
    if forbid_prefix >= vocab_size:
        raise ValueError("forbid_prefix must be smaller than vocab size")
    # Set first forbid_prefix positions to -inf (forbidden)
    if forbid_prefix > 0:
        # Works with arbitrary batch dimensions
        logits[..., :forbid_prefix] = float("-inf")
    # Ensure topk does not exceed number of valid tokens
    effective_vocab = vocab_size - forbid_prefix
    if topk <= 0 or topk > effective_vocab:
        raise ValueError(f"topk must be in 1..{effective_vocab} given forbid_prefix={forbid_prefix}")
    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)
    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class ModelArgs:
    llm_name: str
    decoder_name: str
    llm_pretrained_model: str
    audio_embeddings_path: str 
    audio_understanding_expert_path: str
    audio_semantic_vocab_size: int
    audio_reason_vocab_size: int
    audio_num_codebooks: int


class Model(
    nn.Module,
    PyTorchModelHubMixin,
    pipeline_tag="Text-Audio Foundation Models",
):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        llm_config = gpt_config.from_name(config.llm_name)
        self.backbone, backbone_dim = _prepare_llm_transformer(GPT(llm_config))
        self.backbone.load_state_dict(torch.load(config.llm_pretrained_model, 'cpu'))
        decoder_config = gpt_config.from_name(config.decoder_name)
        self.decoder, decoder_dim = _prepare_transformer(GPT(decoder_config))
        self.audio_embeddings = nn.Embedding((config.audio_semantic_vocab_size + config.audio_reason_vocab_size )* config.audio_num_codebooks, backbone_dim)
        
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks, decoder_dim, 
                               config.audio_semantic_vocab_size + config.audio_reason_vocab_size))
    def forward(self, tokens: torch.Tensor, labels: torch.Tensor, 
                      tokens_mask: torch.Tensor, loss_mask: torch.Tensor, input_pos=None):
        '''
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            labels: (batch_size, seq_len, audio_num_codebooks)
            input_pos: (batch_size, seq_len) positions for each token
            mask: (batch_size, seq_len, max_seq_len
        '''
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask[:,:-1,:].unsqueeze(-1)
        h = masked_embeds.sum(dim=2)  # merge streams
        
        h = self.backbone(h, input_pos=input_pos)
        text_logits = self.backbone.lm_head(h)

        audio_local_embed = self._embed_local_audio(labels[:,:,:-1])  # remove text stream and last VQ layer
        curr_h = torch.cat([h.unsqueeze(2), audio_local_embed], dim=2)  # (B, seq_len, audio_num_codebooks, D)
        curr_h = curr_h[tokens_mask[:,1:,0].bool()]  # (N, audio_num_codebooks, D); only audio steps
        choosed_label = labels[tokens_mask[:,1:,0].bool()]
        choosed_mask = loss_mask[:,1:,:][tokens_mask[:,1:,0].bool()]
        decoder_h = self.decoder(self.projection(curr_h)) # B, N, D
        ci_logits = torch.einsum("bsd,sdv->bsv", decoder_h, self.audio_head)
        return text_logits, ci_logits, choosed_label, choosed_mask

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """Setup KV caches and return a causal mask."""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        with device:
            self.backbone.set_kv_cache(max_batch_size, device=device, dtype=dtype)
            self.decoder.set_kv_cache(max_batch_size, max_seq_length=self.config.audio_num_codebooks, device=device, dtype=dtype)

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        input_pos_maxp1: torch.Tensor,
        temperature: float,
        topk: int,
        forbid_prefix: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len) positions for each token
            mask: (batch_size, seq_len, max_seq_len

        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)  # merge streams
        h = self.backbone(h, input_pos=input_pos, input_pos_maxp1=input_pos_maxp1).to(dtype=dtype)
        last_h = h[:, -1, :] # the last frame
        text_logits = self.backbone.lm_head(last_h)
        text_sample = sample_topk(text_logits, topk, temperature)
        curr_sample = text_sample
        curr_h = last_h.unsqueeze(1)
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)
        # Decoder caches must be reset every frame.
        self.decoder.reset_kv_cache() # !!!!
        for i in range(self.config.audio_num_codebooks):
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos).to(dtype=dtype)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i])
            ci_sample = audio_sample_topk(ci_logits, topk, temperature, forbid_prefix)
            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def reset_caches(self):
        self.backbone.reset_kv_cache()
        self.decoder.reset_kv_cache()

    def _embed_local_audio(self, tokens):
        ''' the token from 0-30
        '''
        audio_tokens = tokens + ((self.config.audio_semantic_vocab_size + self.config.audio_reason_vocab_size) * torch.arange(self.config.audio_num_codebooks-1, device=tokens.device))
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks-1, -1
        )
        return audio_embeds

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * (self.config.audio_semantic_vocab_size + self.config.audio_reason_vocab_size))

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.backbone.transformer.wte(tokens[:, :, -1]).unsqueeze(-2) # the last layer is text token

        audio_tokens = tokens[:, :, :-1] + (
            (self.config.audio_semantic_vocab_size + self.config.audio_reason_vocab_size) * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )
        return torch.cat([audio_embeds, text_embeds], dim=-2)

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.backbone.transformer.h)


class Model_stage3(
    nn.Module,
    PyTorchModelHubMixin,
    pipeline_tag="Text-Audio Foundation Models",
):
    """Stage 3: text-audio joint pre-training."""
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        llm_config = gpt_config.from_name(config.llm_name)
        self.backbone, backbone_dim = _prepare_llm_transformer(GPT(llm_config))
        decoder_config = gpt_config.from_name(config.decoder_name)
        self.decoder, decoder_dim = _prepare_transformer(GPT(decoder_config))
        self.audio_embeddings = nn.Embedding((config.audio_semantic_vocab_size + config.audio_reason_vocab_size )* config.audio_num_codebooks, backbone_dim)
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks, decoder_dim, 
                               config.audio_semantic_vocab_size + config.audio_reason_vocab_size))
        audio_understanding_expert_config = gpt_config.from_name('meta-llama/Llama-3.2-Understanding') # load three causal transformer layers
        self.audio_understanding_expert, _ = _prepare_transformer(GPT(audio_understanding_expert_config))
        # we should load the audio understanding expert
        audio_generation_expert_config = gpt_config.from_name('meta-llama/Llama-3.2-Generation')
        self.audio_generation_expert, _ = _prepare_transformer(GPT(audio_generation_expert_config))
    
    def load_from_stage2_checkpoint(self, checkpoint_path):
        """Load weights from a Stage2 checkpoint into this Stage3 model."""
        print("=" * 80)
        print("Initializing Stage3 from Stage2 checkpoint")
        print("=" * 80)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in checkpoint:
                stage2_state_dict = checkpoint['model']
                stage2_state_dict = { k.split("module.")[-1] if k.startswith("module.") else k: v
                        for k, v in stage2_state_dict.items()}
            elif 'state_dict' in checkpoint:
                stage2_state_dict = checkpoint['state_dict']
            else:
                stage2_state_dict = checkpoint
            print(f"Stage2 checkpoint loaded: {len(stage2_state_dict)} parameters")
            current_state_dict = self.state_dict()
            loaded_count = 0
            skipped_count = 0
            shape_mismatch_count = 0
            print("\nInitializing parameters...")
            for name, param in current_state_dict.items():
                if name in stage2_state_dict:
                    stage1_param = stage2_state_dict[name]
                    if stage1_param.shape == param.shape:
                        param.data.copy_(stage1_param)
                        loaded_count += 1
                    else:
                        shape_mismatch_count += 1
                else:
                    skipped_count += 1
            print("\n" + "=" * 80)
            print("Parameter initialization summary:")
            print(f"  Loaded: {loaded_count}")
            print(f"  Shape mismatch: {shape_mismatch_count}")
            print(f"  Skipped (new): {skipped_count}")
            print(f"  Total: {len(current_state_dict)}")
            print(f"  Load ratio: {loaded_count/len(current_state_dict)*100:.1f}%")
            print("=" * 80)
            del checkpoint

        except Exception as e:
            print(f"Failed to load Stage2 checkpoint: {e}")
            raise
    
    def forward(self, tokens: torch.Tensor, labels: torch.Tensor, 
            tokens_mask: torch.Tensor, loss_mask: torch.Tensor, input_pos=None, input_pos_maxp1=None):
        """
        Args:
            tokens: (B, S, audio_num_codebooks+1)
            tokens_mask: (B, S, audio_num_codebooks+1)
            labels: (B, S, audio_num_codebooks)
            input_pos: (B, S)
        Returns:
            text_logits: (B, S, vocab_text)
            ci_logits:   (N, audio_num_codebooks, audio_vocab)   # N = total selected audio steps
            choosed_label: (N, audio_num_codebooks)
            choosed_mask:  (N, audio_num_codebooks)
        """
        dtype = next(self.parameters()).dtype
        device = tokens.device
        B, S, _ = tokens.size()

        # 1) Embed each stream: last stream is text, rest are audio codebooks
        audio_stream_embeds = self._embed_audio_tokens(tokens)  # (B, S, audio_num_codebooks, D)

        # Mask setup: 1 = audio step, 1 = text step
        audio_step_mask = tokens_mask[:, :-1, 0].unsqueeze(-1).to(dtype=dtype)   # (B, S, 1)
        text_step_mask = tokens_mask[:, :-1, -1].unsqueeze(-1).to(dtype=dtype)  # (B, S, 1)

        # 2) Audio expert input: fuse codebooks per timestep (text positions are masked in data)
        audio_stream_mask = tokens_mask[:, :-1, :-1].unsqueeze(-1).to(dtype=dtype)
        audio_input = (audio_stream_embeds * audio_stream_mask).sum(dim=2)  # (B, S, D)

        h_audio = self.audio_understanding_expert(audio_input)  # (B, S, D)

        # 3) Text embeddings (not fed to audio expert)
        text_embeds = self.backbone.transformer.wte(tokens[:, :, -1])  # (B, S, D_backbone)

        # 4) Build unified backbone input: audio steps = expert output, text steps = text embed, padding = 0
        backbone_input = h_audio * audio_step_mask + text_embeds * text_step_mask  # (B, S, D)

        # 5) Unified LLM backbone
        h = self.backbone(backbone_input, input_pos=input_pos)  # (B, S, D)

        generation_input = h * audio_step_mask
        h_audio = self.audio_generation_expert(generation_input)
        h_final = h_audio * audio_step_mask + h * text_step_mask  # recover text features
        text_logits = self.backbone.lm_head(h_final)                            # (B, S, V_text)

        audio_local_embed = self._embed_local_audio(labels[:,:,:-1])  # drop text stream and last VQ layer
        curr_h = torch.cat([h_final.unsqueeze(2), audio_local_embed], dim=2)  # (B, S, audio_num_codebooks, D)
        curr_h = curr_h[tokens_mask[:,1:,0].bool()]  # (N, audio_num_codebooks, D); only audio steps
        choosed_label = labels[tokens_mask[:,1:,0].bool()]
        choosed_mask = loss_mask[:,1:,:][tokens_mask[:,1:,0].bool()]
        decoder_h = self.decoder(self.projection(curr_h))  # (B, N, D)
        ci_logits = torch.einsum("bsd,sdv->bsv", decoder_h, self.audio_head)
        return text_logits, ci_logits, choosed_label, choosed_mask

    def forward_prefix(self, tokens: torch.Tensor, labels: torch.Tensor, 
            tokens_mask: torch.Tensor, loss_mask: torch.Tensor, input_pos=None, input_pos_maxp1=None):
        """
        Args:
            tokens: (B, S, audio_num_codebooks+1)
            tokens_mask: (B, S, audio_num_codebooks+1)
            labels: (B, S, audio_num_codebooks)
            input_pos: (B, S)
        Returns:
            text_logits: (B, S, vocab_text)
            ci_logits:   (N, audio_num_codebooks, audio_vocab)   # N = total selected audio steps
            choosed_label: (N, audio_num_codebooks)
            choosed_mask:  (N, audio_num_codebooks)
        """
        dtype = next(self.parameters()).dtype
        device = tokens.device
        B, S, _ = tokens.size()

        # 1) Embed each stream: last stream is text, rest are audio codebooks
        audio_stream_embeds = self._embed_audio_tokens(tokens)  # (B, S, audio_num_codebooks, D)

        audio_step_mask = tokens_mask[:, :-1, 0].unsqueeze(-1).to(dtype=dtype)   # (B, S, 1)
        text_step_mask = tokens_mask[:, :-1, -1].unsqueeze(-1).to(dtype=dtype)  # (B, S, 1)

        # 2) Audio expert: fuse codebooks per timestep (text positions masked in data)
        audio_stream_mask = tokens_mask[:, :-1, :-1].unsqueeze(-1).to(dtype=dtype)
        audio_input = (audio_stream_embeds * audio_stream_mask).sum(dim=2)  # (B, S, D)

        h_audio = self.audio_understanding_expert(audio_input, input_pos=input_pos, input_pos_maxp1=input_pos_maxp1)  # (B, S, D)

        # 3) Text embeddings (not fed to audio expert)
        text_embeds = self.backbone.transformer.wte(tokens[:, :, -1])  # (B, S, D_backbone)

        # 4) Unified backbone input: audio steps = expert output, text = embed, padding = 0
        backbone_input = h_audio * audio_step_mask + text_embeds * text_step_mask  # (B, S, D)

        # 5) LLM backbone
        h = self.backbone(backbone_input, input_pos=input_pos, input_pos_maxp1=input_pos_maxp1)  # (B, S, D)

        generation_input = h * audio_step_mask
        h_audio = self.audio_generation_expert(generation_input, input_pos=input_pos, input_pos_maxp1=input_pos_maxp1)
        h_final = h_audio * audio_step_mask + h * text_step_mask  # recover text features
        text_logits = self.backbone.lm_head(h_final)                            # (B, S, V_text)

        audio_local_embed = self._embed_local_audio(labels[:,:,:-1])  # drop text stream and last VQ layer
        curr_h = torch.cat([h_final.unsqueeze(2), audio_local_embed], dim=2)  # (B, S, audio_num_codebooks, D)
        curr_h = curr_h[tokens_mask[:,1:,0].bool()]  # (N, audio_num_codebooks, D); only audio steps
        choosed_label = labels[tokens_mask[:,1:,0].bool()]
        choosed_mask = loss_mask[:,1:,:][tokens_mask[:,1:,0].bool()]
        decoder_h = self.decoder(self.projection(curr_h))  # (B, N, D)
        ci_logits = torch.einsum("bsd,sdv->bsv", decoder_h, self.audio_head)
        return text_logits, ci_logits, choosed_label, choosed_mask

    def forward_text(self, tokens: torch.Tensor, labels: torch.Tensor, 
            tokens_mask: torch.Tensor, loss_mask: torch.Tensor, input_pos=None):
        """
        Args:
            tokens: (B, S, audio_num_codebooks+1)
            tokens_mask: (B, S, audio_num_codebooks+1)
            labels: (B, S, audio_num_codebooks)
            input_pos: (B, S)
        Returns:
            text_logits: (B, S, vocab_text)
            ci_logits:   (N, audio_num_codebooks, audio_vocab)   # N = total selected audio steps
            choosed_label: (N, audio_num_codebooks)
            choosed_mask:  (N, audio_num_codebooks)
        """
        dtype = next(self.parameters()).dtype
        device = tokens.device
        B, S, _ = tokens.size()

        # 1) Embed each stream: last stream is text, rest are audio codebooks
        audio_stream_embeds = self._embed_audio_tokens(tokens)  # (B, S, audio_num_codebooks, D)

        audio_step_mask = tokens_mask[:, :, 0].unsqueeze(-1).to(dtype=dtype)   # (B, S, 1)
        text_step_mask = tokens_mask[:, :, -1].unsqueeze(-1).to(dtype=dtype)  # (B, S, 1)

        # 2) Audio expert input (text positions masked in data)
        audio_stream_mask = tokens_mask[:, :, :-1].unsqueeze(-1).to(dtype=dtype)
        audio_input = (audio_stream_embeds * audio_stream_mask).sum(dim=2)  # (B, S, D)

        h_audio = self.audio_understanding_expert(audio_input)  # (B, S, D)

        # 3) Text embeddings
        text_embeds = self.backbone.transformer.wte(tokens[:, :, -1])  # (B, S, D_backbone)

        # 4) Unified backbone input: audio = expert output, text = embed, padding = 0
        backbone_input = h_audio * audio_step_mask + text_embeds * text_step_mask  # (B, S, D)

        # 5) LLM backbone
        h = self.backbone(backbone_input)  # (B, S, D)

        generation_input = h * audio_step_mask
        h_audio = self.audio_generation_expert(generation_input)
        h_final = h_audio * audio_step_mask + h * text_step_mask  # recover text features
        text_logits = self.backbone.lm_head(h_final)  # (B, S, V_text)
        return text_logits 

    def setup_caches(self, max_batch_size: int) -> torch.Tensor:
        """Setup KV caches and return a causal mask."""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        with device:
            self.backbone.set_kv_cache(max_batch_size, max_seq_length=2048, device=device, dtype=dtype)
            self.decoder.set_kv_cache(max_batch_size, max_seq_length=self.config.audio_num_codebooks, device=device, dtype=dtype)
            # Audio understanding expert
            self.audio_understanding_expert.set_kv_cache(max_batch_size, max_seq_length=2048, device=device, dtype=dtype)
            # Audio generation expert
            self.audio_generation_expert.set_kv_cache(max_batch_size, max_seq_length=2048, device=device, dtype=dtype)

    
    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        input_pos_maxp1: torch.Tensor,
        temperature: float,
        topk: int,
        forbid_prefix: int = 0,
        cfg_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Args:
            tokens:      (B, S, audio_num_codebooks+1)
            tokens_mask: (B, S, audio_num_codebooks+1)
            input_pos:   (B, S)
        Returns:
            (B, 1 + audio_num_codebooks) sampled tokens for the next step
        """
        dtype = next(self.parameters()).dtype
        device = tokens.device
        B, S, C1 = tokens.size()
        num_cb = self.config.audio_num_codebooks
        assert C1 == num_cb + 1, "last stream must be text"

        # 1) Audio-step / text-step masks
        audio_step_mask = tokens_mask[:, :, 0].unsqueeze(-1).to(dtype=dtype)   # (B,S,1)
        text_step_mask  = tokens_mask[:, :, -1].unsqueeze(-1).to(dtype=dtype)  # (B,S,1)

        # 2) Fuse codebooks per timestep for audio expert; non-audio steps masked next
        audio_embeds = self._embed_audio_tokens(tokens)
        audio_stream_mask = tokens_mask[:, :, :-1].unsqueeze(-1).to(dtype=dtype)
        audio_input = (audio_embeds * audio_stream_mask).sum(dim=2)  # (B,S,D)

        h_audio = self.audio_understanding_expert(audio_input, input_pos=input_pos, input_pos_maxp1=input_pos_maxp1).to(dtype=dtype)  # (B,S,D)
        # 3) Text embeddings from backbone
        text_embeds = self.backbone.transformer.wte(tokens[:, :, -1])  # (B,S,D)

        # 4) Unified backbone input
        backbone_input = h_audio * audio_step_mask + text_embeds * text_step_mask  # (B,S,D)
        # 5) LLM backbone incremental forward; take last timestep
        h = self.backbone(backbone_input, input_pos=input_pos, input_pos_maxp1=input_pos_maxp1).to(dtype=dtype)  # (B,S,D)
        generation_input = h * audio_step_mask
        h_audio = self.audio_generation_expert(generation_input, input_pos=input_pos, input_pos_maxp1=input_pos_maxp1)

        h_final = h_audio * audio_step_mask + h * text_step_mask  # recover text features

        last_h = h_final[:, -1, :]  # (B,D)
        # Text head: sample one text token per frame first
        text_logits  = self.backbone.lm_head(last_h)  # (B,V_text)
        if cfg_scale > 1.0 and B > 1:
            logits_c0 = text_logits[1:,:] + (text_logits[0:1,:]-text_logits[1:,:])*cfg_scale
            text_sample = sample_topk(logits_c0, topk, temperature)
            text_sample = text_sample.repeat(2, 1) # repeat to uncondition
        else:
            text_sample  = sample_topk(text_logits, topk, temperature)  # (B,1)
        curr_sample = text_sample

        curr_h = last_h.unsqueeze(1)
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)
        # Decoder caches must be reset every frame.
        self.decoder.reset_kv_cache() # !!!!
        for i in range(self.config.audio_num_codebooks):
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos).to(dtype=dtype)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i])

            if cfg_scale > 1.0 and B > 1:
                logits_ci = ci_logits[1:,:] + (ci_logits[0:1,:]-ci_logits[1:,:])*cfg_scale
                ci_sample = audio_sample_topk(logits_ci, topk, temperature, forbid_prefix)
                ci_sample = ci_sample.repeat(2, 1)
            else:
                ci_sample = audio_sample_topk(ci_logits, topk, temperature, forbid_prefix)
            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def reset_caches(self):
        self.backbone.reset_kv_cache()
        self.decoder.reset_kv_cache()
        self.audio_generation_expert.reset_kv_cache()
        self.audio_understanding_expert.reset_kv_cache()

    def _embed_local_audio(self, tokens):
        ''' the token from 0-30
        '''
        audio_tokens = tokens + ((self.config.audio_semantic_vocab_size + self.config.audio_reason_vocab_size) * torch.arange(self.config.audio_num_codebooks-1, device=tokens.device))
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks-1, -1
        )
        return audio_embeds

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * (self.config.audio_semantic_vocab_size + self.config.audio_reason_vocab_size))

    def _embed_audio_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        
        audio_tokens = tokens[:, :, :-1] + (
            (self.config.audio_semantic_vocab_size + self.config.audio_reason_vocab_size) * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )
        return audio_embeds

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.backbone.transformer.wte(tokens[:, :, -1]).unsqueeze(-2) # the last layer is text token

        audio_tokens = tokens[:, :, :-1] + (
            (self.config.audio_semantic_vocab_size + self.config.audio_reason_vocab_size) * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )
        return torch.cat([audio_embeds, text_embeds], dim=-2)

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.backbone.transformer.h) + list(self.audio_understanding_expert.transformer.h) + list(self.audio_generation_expert.transformer.h)
