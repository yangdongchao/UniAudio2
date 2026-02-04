# build the TTS task evlaution set
# (1) without the audio timbre prompt
# (2) using the audio timbre prompt


from dataclasses import dataclass
from typing import List, Tuple
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from llm_models.model_new import Model, ModelArgs, Model_stage3
from tools.tokenizer.Text2ID.text_tokenizer import TextTokenizer
from llm_utils.arguments import str2bool
import sys
import os
from llm_utils.train_utils import resume_for_inference
import argparse
from pathlib import Path
import yaml
import os 
import random
from tools.tokenizer.ReasoningCodec_film.reason_tokenizer import ReasoningTokenizer
import json

@dataclass
class Segment:
    segment_id: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_text_tokenizer(tokenizer_checkpoint_path):
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer = TextTokenizer(checkpoint_dir=tokenizer_checkpoint_path)
    # bos = tokenizer.bos_id
    # eos = tokenizer.eos_id
    return tokenizer


def load_audio_tokenizer(train_config, model_path, device):
    #tokenizer = ReasoningTokenizer(train_config, model_path, device)
    #tokenizer = ReasoningTokenizer(model_path=model_path, train_config=train_config, device=device)
    tokenizer = None 
    return tokenizer

def read_json(json_path):
    # read the json file
    with open(json_path, 'r', encoding='utf-8') as f:
        config_content = json.load(f)
    return config_content

import re
from typing import List

def post_process_text(text: str, max_repeat_length: int = 3) -> str:
    """
    后处理文本，移除尾部重复
    
    Args:
        text: 原始文本
        max_repeat_length: 最大允许的重复长度
    
    Returns:
        处理后的文本
    """
    if not text:
        return text
    
    # 方法1: 检测并移除尾部重复模式
    processed_text = remove_tail_repetition(text, max_repeat_length)
    
    # 方法2: 如果方法1效果不好，使用更激进的策略
    if has_excessive_repetition(processed_text):
        processed_text = aggressive_repetition_removal(processed_text)
    
    # 方法3: 清理多余空格和标点
    processed_text = clean_text(processed_text)
    
    return processed_text

def remove_tail_repetition(text: str, max_repeat: int = 3) -> str:
    """移除尾部重复"""
    words = text.strip().split()
    if len(words) <= 1:
        return text
    
    # 从尾部开始检查重复
    i = len(words) - 1
    while i > 0:
        # 检查当前词是否在前面出现过
        current_word = words[i].lower().strip('.,!?;')
        if not current_word:
            i -= 1
            continue
            
        # 统计当前词在最近窗口中的出现次数
        window_size = min(10, len(words))
        start_idx = max(0, i - window_size)
        count = 0
        
        for j in range(start_idx, i):
            compare_word = words[j].lower().strip('.,!?;')
            if current_word == compare_word:
                count += 1
        
        # 如果重复次数过多，移除尾部重复部分
        if count >= max_repeat:
            # 找到第一个重复的位置
            for k in range(i-1, -1, -1):
                if words[k].lower().strip('.,!?;') == current_word:
                    # 保留到第一个重复位置
                    return ' '.join(words[:k+1])
        
        i -= 1
    
    return text

def has_excessive_repetition(text: str, threshold: int = 2) -> bool:
    """检测是否存在过度重复"""
    words = text.strip().split()
    if len(words) < 5:
        return False
    
    word_counts = {}
    for word in words:
        clean_word = word.lower().strip('.,!?;')
        if len(clean_word) > 2:  # 只考虑长度大于2的词
            word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
    
    # 检查是否有词重复次数超过阈值
    max_count = max(word_counts.values()) if word_counts else 0
    return max_count > threshold and max_count > len(words) * 0.3

def aggressive_repetition_removal(text: str) -> str:
    """激进的重复移除策略"""
    words = text.strip().split()
    if len(words) <= 3:
        return text
    
    # 构建词频统计
    word_sequence = []
    seen_phrases = set()
    
    for word in words:
        clean_word = word.lower().strip('.,!?;')
        
        # 检查是否形成重复短语
        if len(word_sequence) >= 2:
            recent_phrase = ' '.join(word_sequence[-2:]).lower()
            current_phrase = recent_phrase + ' ' + clean_word if len(recent_phrase) > 0 else clean_word
            
            if current_phrase in seen_phrases:
                # 发现重复短语，停止添加新词
                break
            
            seen_phrases.add(current_phrase)
        
        word_sequence.append(word)
    
    return ' '.join(word_sequence)

def clean_text(text: str) -> str:
    """清理文本"""
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 修复标点符号周围的空格
    text = re.sub(r'\s*([.,!?;])\s*', r'\1 ', text)
    text = re.sub(r'\s+$', '', text)  # 移除尾部空格
    return text.strip()


class Generator:
    def __init__(
        self,
        model: Model,
        train_args,
        audio_tokenizer_config,
        audio_model_path,
        text_tokenizer_path,
        is_cfg = False
    ):
        self._model = model
        if is_cfg:
            self._model.setup_caches(2) # 1 for no cfg. 2 for with cfg
        else:
            self._model.setup_caches(1)
        self.is_cfg = is_cfg
        self._text_tokenizer = load_text_tokenizer(text_tokenizer_path)
        device = next(model.parameters()).device
        self._audio_tokenizer = load_audio_tokenizer(train_config=audio_tokenizer_config, model_path=audio_model_path, device=device)
        self.sample_rate = 24000
        self.device = device
        
        self.empty_token = 0 # all of the emtpy token can be set as 0. It cannot produce any influence
        self.text_pad_token = train_args.text_pad_token 
        self.semantic_pad_token = train_args.semantic_pad_token
        self.semantic_eos = train_args.semantic_eos
        self.semantic_bos = train_args.semantic_bos
        self.reason_eos = train_args.reason_eos
        self.reason_bos = train_args.reason_bos
        self.reason_pad_token = train_args.reason_pad_token
        self.parallel_number = train_args.parallel_number # how many parallel tokens
        self.audio_reason_card = train_args.audio_reason_card
        self.special_token_dict = self.get_special_token()

    def get_special_token(self):
        return {'<think>': 128002, '</think>': 128003, '</answer>': 128005,
                '<transcription>': 128011, '</transcription>': 128012, '<lyric>': 128013,
                '</lyric>': 128014, '<caption>': 128015, '</caption>': 128016, '<answer>': 128017,
                '<reason_token>': 128018, '<semantic_token>': 128019}
    
    def _tokenize_text_segment(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []
        text_tokens = self._text_tokenizer.tokenize(text)
        text_frame = torch.zeros(len(text_tokens), self.parallel_number).long()

        text_frame_mask = torch.zeros(len(text_tokens), self.parallel_number).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        
        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.ones(audio_tokens.size(0), 1).to(self.device)*self.semantic_eos
        
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)
        
        audio_frame = torch.zeros(audio_tokens.size(1), self.parallel_number).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), self.parallel_number).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True # true denotes can be used

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    def text_pad(self, x):
        '''input 1-dimension sequence. add empty token for semantic streaming.
        '''
        sequences = torch.ones((len(x), self.parallel_number)).to(torch.int64)
        sequences[:, -1] = x # the text tokens
        sequences[:,:-1] = sequences[:,:-1]*self.empty_token # we will set as 0
        return sequences

    def audio_pad(self, x):
        '''input audio (T, 4) sequence. Add empty token for text.
        '''
        sequences = torch.ones((x.shape[0], self.parallel_number)).to(torch.int64)*self.empty_token
        sequences[:,:-1] = x 
        return sequences

    def add_offset_semantic(self, x, offset_value):
        '''Add offset for semantic tokens'''
        x =  x + offset_value 
        return x 

    def add_special_token(self, key, this_data):
        if key == 'text_seq':
            return this_data
        key = key.replace('_seq','')
        tmp_start = self.special_token_dict['<' + key + '>']
        tmp_end = self.special_token_dict['</' + key + '>']
        bos_frame = torch.ones(1)*tmp_start # add bos tokens
        eos_frame = torch.ones(1)*tmp_end # add eos tokens
        this_data = torch.cat([bos_frame, this_data, eos_frame], dim=0)
        return this_data

    def add_special_start_token(self, key):
        key = key.replace('_seq','')
        tmp_start = self.special_token_dict['<' + key + '>']
        bos_frame = torch.ones(1)*tmp_start # add bos tokens
        return bos_frame

    def prepare_asr_task(self, task_prompt, this_reason_data, this_semantic_data):
        ''' we add the start of transcirption as the start?
        '''
        this_text_data = self.text_pad(task_prompt)
        this_text_mask = torch.zeros((this_text_data.shape[0], self.parallel_number))
        this_text_mask[:,-1] = True

        reason_bos_frame = torch.ones(1, this_reason_data.shape[1])*self.reason_bos # add bos tokens
        reason_eos_frame = torch.ones(1, this_reason_data.shape[1])*self.reason_eos # add eos tokens
        this_reason_data = torch.cat([reason_bos_frame, this_reason_data, reason_eos_frame], dim=0) # (T+2, 8)

        semantic_bos_frame = torch.ones(1, this_semantic_data.shape[1])*self.semantic_bos # add bos tokens
        semantic_eos_frame = torch.ones(1, this_semantic_data.shape[1])*self.semantic_eos # add eos tokens
        this_semantic_data = torch.cat([semantic_bos_frame, this_semantic_data, semantic_eos_frame], dim=0) # (T+2, 8)
        this_semantic_data = self.add_offset_semantic(this_semantic_data, self.audio_reason_card)

        this_audio_data = torch.cat([this_reason_data, this_semantic_data], dim=0)

        this_audio_data = self.audio_pad(this_audio_data)
        this_audio_mask = torch.zeros((this_audio_data.shape[0], self.parallel_number))
        this_audio_mask[:,:-1] = True 

        # add special start token
        # transcirption_start = self.add_special_start_token('transcription_seq')
        # transcirption_start_mask = torch.zeros((transcirption_start.shape[0], self.parallel_number))
        this_data = torch.cat([this_text_data, this_audio_data], dim=0)
        this_mask = torch.cat([this_text_mask, this_audio_mask], dim=0)
        return this_data, this_mask

    @torch.inference_mode()
    def generate_asr_with_ngram_sampling(
        self,
        task_prompt,
        task_name, 
        text_token = None,
        semantic_token = None,
        reason_token = None,
        temperature: float = 0.9,
        topk: int = 200,
        cfg_scale = 1.0,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        ''' 使用n-gram采样的ASR生成 '''
        self._model.reset_caches()
        max_audio_frames = 500
        tokens, tokens_mask = self.prepare_asr_task(task_prompt, reason_token, semantic_token)
        
        prompt_tokens = tokens.to(self.device)
        prompt_tokens_mask = tokens_mask.bool().to(self.device)
        
        bs_size = 1
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        
        # pre-filling
        _ = self._model(curr_tokens[:,:-1], labels=curr_tokens[:,1:,:-1], tokens_mask=curr_tokens_mask, loss_mask=curr_tokens_mask, input_pos=curr_pos[:,:-1])
        curr_pos = torch.tensor([prompt_tokens.size(0)-1], device=self.device, dtype=torch.int64)
        input_pos_maxp1 = prompt_tokens.size(0)
        text_samples = []
        
        # 用于避免重复的数据结构
        generated_tokens = []
        generated_ngrams = {}
        
        for _ in range(max_audio_frames):
            # 使用原始的generate_frame方法
            sample = self._model.generate_frame(
                curr_tokens[:,-1:], 
                curr_tokens_mask[:,-1:], 
                input_pos=curr_pos, 
                input_pos_maxp1=input_pos_maxp1, 
                temperature=temperature, 
                topk=topk, 
                forbid_prefix=0
            )
            
            text_token_candidate = sample[:, 0:1]
            
            # 检查是否应该接受这个token（基于n-gram约束）
            if no_repeat_ngram_size > 0 and self._should_reject_ngram(text_token_candidate.item(), generated_tokens, generated_ngrams, no_repeat_ngram_size):
                # 如果被拒绝，使用回退策略
                text_token_candidate = self._get_fallback_token(generated_tokens, repetition_penalty)
            
            text_token = text_token_candidate
            audio_tokens = torch.zeros(1, 8).to(self.device).long()
            
            if text_token.item() == 128001:
                break
                
            text_samples.append(text_token[0,0])
            generated_tokens.append(text_token.item())
            
            # 更新n-gram记录
            if no_repeat_ngram_size > 0:
                self._update_ngrams_simple(generated_tokens, generated_ngrams, no_repeat_ngram_size)
            
            curr_tokens = torch.cat([audio_tokens, text_token], dim=-1)
            curr_tokens = curr_tokens.unsqueeze(1).to(self.device)
            curr_tokens_mask = torch.cat(
                [torch.zeros_like(audio_tokens).bool(), torch.ones(bs_size, 1).bool().to(self.device)], dim=1 ).unsqueeze(1)
            curr_pos.add_(1)
            input_pos_maxp1 += 1
        
        text_content = self._text_tokenizer.decode(torch.stack(text_samples, dim=-1))
        return text_content

    def _should_reject_ngram(self, candidate_token, generated_tokens, generated_ngrams, ngram_size):
        """检查候选token是否会导致n-gram重复"""
        if len(generated_tokens) < ngram_size - 1:
            return False
        
        current_prefix = tuple(generated_tokens[-(ngram_size-1):])
        banned_tokens = generated_ngrams.get(current_prefix, set())
        
        return candidate_token in banned_tokens

    def _get_fallback_token(self, generated_tokens, repetition_penalty):
        """获取回退token（简单的策略：选择与最近token不同的token）"""
        # 这里可以实现更复杂的回退策略
        # 当前实现：返回一个固定的安全token（需要根据您的tokenizer调整）
        return torch.tensor([[128000]]).to(self.device)  # 使用一个安全的token

    def _update_ngrams_simple(self, generated_tokens, generated_ngrams, ngram_size):
        """简单更新n-gram记录"""
        if len(generated_tokens) < ngram_size:
            return
        
        for i in range(len(generated_tokens) - ngram_size + 1):
            ngram = tuple(generated_tokens[i:i+ngram_size])
            prefix = ngram[:-1]
            next_token = ngram[-1]
            
            if prefix not in generated_ngrams:
                generated_ngrams[prefix] = set()
            generated_ngrams[prefix].add(next_token)

    @torch.inference_mode()
    def generate_asr_beam_search(
        self,
        task_prompt,
        task_name, 
        text_token = None,
        semantic_token = None,
        reason_token = None,
        beam_width: int = 5,
        length_penalty: float = 0.6,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        max_length: int = 500
    ) -> torch.Tensor:
        ''' 使用束搜索的ASR生成 '''
        self._model.reset_caches()
        tokens, tokens_mask = self.prepare_asr_task(task_prompt, reason_token, semantic_token)
        
        prompt_tokens = tokens.to(self.device)
        prompt_tokens_mask = tokens_mask.bool().to(self.device)
        
        
        
        # 预填充（所有beam共享相同的初始状态）
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        # 初始化束搜索
        beams = [{
            'tokens': prompt_tokens.clone(),
            'mask': prompt_tokens_mask.clone(),
            'score': 0.0,
            'finished': False,
            'text_tokens': [],  # 只记录文本token
            'pos': torch.tensor([prompt_tokens.size(0)-1], device=self.device, dtype=torch.int64),
            'input_pos_maxp1': prompt_tokens.size(0)
        }]
        # print('curr_pos ', curr_pos.shape)
        # print('curr_pos 3', curr_pos)
        _ = self._model(curr_tokens[:,:-1], labels=curr_tokens[:,1:,:-1], 
                    tokens_mask=curr_tokens_mask, loss_mask=curr_tokens_mask, 
                    input_pos=curr_pos[:,:-1])
        
        for step in range(max_length):
            # 收集所有活跃beam的下一个token候选
            all_candidates = []
            
            for beam in beams:
                if beam['finished']:
                    # 对于已完成的beam，直接保留
                    all_candidates.append(beam)
                    continue
                
                # 准备当前beam的输入
                curr_tokens_beam = beam['tokens'].unsqueeze(0)
                curr_tokens_mask_beam = beam['mask'].unsqueeze(0)
                curr_pos_beam = beam['pos']
                
                # 获取下一个token的logits
                logits = self._get_next_logits(
                    curr_tokens_beam[:, -1:], 
                    curr_tokens_mask_beam[:, -1:], 
                    curr_pos_beam,
                    beam['input_pos_maxp1']
                )
                
                # 应用logits处理
                logits = self._process_logits_beam_search(
                    logits, beam['text_tokens'], no_repeat_ngram_size
                )
                
                # 获取top-k候选
                probs = torch.softmax(logits, dim=-1)
                topk_probs, topk_tokens = torch.topk(probs, beam_width, dim=-1)
                
                # 为每个候选创建新的beam
                for i in range(beam_width):
                    candidate_token = topk_tokens[0, i].item()
                    candidate_prob = topk_probs[0, i].item()
                    
                    # 跳过EOS token的特殊处理
                    if candidate_token == 128001:  # EOS
                        new_beam = beam.copy()
                        new_beam['finished'] = True
                        # 应用长度惩罚
                        length = len(new_beam['text_tokens'])
                        length_penalty_score = ((5 + length) / (5 + 1)) ** length_penalty
                        new_beam['score'] += torch.log(torch.tensor(candidate_prob)) / length_penalty_score
                        all_candidates.append(new_beam)
                        continue
                    
                    # 创建新的beam
                    new_beam = beam.copy()
                    
                    # 更新token序列
                    new_text_token = torch.tensor([[candidate_token]]).to(self.device)
                    new_audio_tokens = torch.zeros(1, 8).to(self.device).long()
                    
                    new_token = torch.cat([new_audio_tokens, new_text_token], dim=-1)
                    new_beam['tokens'] = torch.cat([new_beam['tokens'], new_token], dim=0)
                    
                    # 更新mask
                    new_mask = torch.cat([
                        torch.zeros_like(new_audio_tokens).bool(), 
                        torch.ones(1, 1).bool().to(self.device)
                    ], dim=1)
                    new_beam['mask'] = torch.cat([new_beam['mask'], new_mask], dim=0)
                    
                    # 更新位置信息
                    new_beam['pos'] = new_beam['pos'].add_(1)
                    new_beam['input_pos_maxp1'] += 1
                    
                    # 更新文本token记录
                    new_beam['text_tokens'].append(candidate_token)
                    
                    # 更新分数（应用长度惩罚）
                    length = len(new_beam['text_tokens'])
                    length_penalty_score = ((5 + length) / (5 + 1)) ** length_penalty
                    new_beam['score'] += torch.log(torch.tensor(candidate_prob)) / length_penalty_score
                    
                    all_candidates.append(new_beam)
            
            # 选择分数最高的beam_width个候选
            beams = sorted(all_candidates, key=lambda x: x['score'], reverse=True)[:beam_width]
            
            # 检查是否所有beam都已完成
            if early_stopping and all(beam['finished'] for beam in beams):
                break
            
            # 检查是否达到最大长度
            if step == max_length - 1:
                for beam in beams:
                    if not beam['finished']:
                        beam['finished'] = True
        
        # 选择最佳beam
        best_beam = max(beams, key=lambda x: x['score'])
        text_content = self._text_tokenizer.decode(torch.tensor(best_beam['text_tokens']))
        
        return text_content

    def _get_next_logits(self, curr_tokens, curr_tokens_mask, curr_pos, input_pos_maxp1):
        """获取下一个token的logits"""
        # 这里需要根据您的模型实现来获取logits
        # 如果您的模型没有直接提供logits的方法，可以使用以下替代方案
        
        # 方法1: 使用generate_frame但返回logits（需要修改模型）
        # 方法2: 使用模型的前向传播获取logits
        
        # 临时实现：使用现有的generate_frame，但需要修改模型以返回logits
        # 这里假设您的模型可以返回logits
        # try:
        # print('curr_tokens ', curr_tokens.shape)
        # print('curr_tokens_mask ', curr_tokens_mask.shape)
        # print('curr_pos ', curr_pos.shape)
        # print('input_pos_maxp1 ', input_pos_maxp1)
        logits = self._model.get_next_logits(
            curr_tokens, curr_tokens_mask, curr_pos, input_pos_maxp1
        )
        return logits
        # except:
        #     # 如果模型不支持直接获取logits，使用替代方法
        #     print('cannot get the logits')

    
    def _process_logits_beam_search(self, logits, generated_tokens, no_repeat_ngram_size):
        """处理logits，应用各种约束"""
        # 应用n-gram重复抑制
        if no_repeat_ngram_size > 0 and len(generated_tokens) >= no_repeat_ngram_size - 1:
            logits = self._apply_ngram_constraint(logits, generated_tokens, no_repeat_ngram_size)
        
        return logits

    def _apply_ngram_constraint(self, logits, generated_tokens, ngram_size):
        """应用n-gram约束"""
        if len(generated_tokens) < ngram_size - 1:
            return logits
        
        # 获取当前的前缀
        current_prefix = tuple(generated_tokens[-(ngram_size-1):])
        
        # 这里需要维护一个n-gram的禁止列表
        # 在实际实现中，您需要在beam搜索过程中维护这个信息
        banned_tokens = set()
        
        # 检查当前前缀是否会导致重复的n-gram
        # 这里简化实现，实际需要更复杂的n-gram跟踪
        
        for token in banned_tokens:
            logits[0, token] = -float('inf')
        
        return logits

    @torch.inference_mode()
    def generate_asr(
        self,
        task_prompt,
        task_name, 
        text_token = None,
        semantic_token = None,
        reason_token = None,
        temperature: float = 0.9,
        topk: int = 200,
        cfg_scale = 1.0,
    ) -> torch.Tensor:
        ''' for pretraining version generation. Without any task description or Lable'''
        self._model.reset_caches() # we reset the whole cache every times
        max_audio_frames = 500 # the frame number of max prediction
        tokens, tokens_mask = self.prepare_asr_task(task_prompt, reason_token, semantic_token)
        
        prompt_tokens = tokens.to(self.device)
        prompt_tokens_mask = tokens_mask.bool().to(self.device)
        samples = []
        
        bs_size = 1
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        
        # pre-filling
        #_ = self._model.forward_prefix(curr_tokens[:,:-1], labels=curr_tokens[:,1:,:-1], tokens_mask=curr_tokens_mask, loss_mask=curr_tokens_mask, input_pos=curr_pos[:,:-1])
        _ = self._model.forward_prefix(curr_tokens[:,:-1], labels=curr_tokens[:,1:,:-1], tokens_mask=curr_tokens_mask, loss_mask=curr_tokens_mask, input_pos=curr_pos[:,:-1])
        curr_pos = torch.tensor([prompt_tokens.size(0)-1], device=self.device, dtype=torch.int64)
        input_pos_maxp1 = prompt_tokens.size(0)
        text_samples = []
        audio_samples = []
        is_reason = True # 后续需要加上是否需要推理的代码
        save_flag = True
        forbid_prefix = 0
        pre_reason_token, pre_semantic_token = [], []
        for _ in range(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens[:,-1:], curr_tokens_mask[:,-1:], input_pos=curr_pos, 
                            input_pos_maxp1=input_pos_maxp1, temperature=temperature, topk=topk, forbid_prefix=forbid_prefix)
            text_token = sample[:, 0:1]
            audio_tokens  = sample[:, 1:]
            text_token = sample[:, 0:1]
            audio_tokens  = torch.zeros(1, 8).to(self.device).long()
            if text_token == 128001:
                break
            text_samples.append(text_token[0,0])
            curr_tokens = torch.cat([audio_tokens, text_token], dim=-1)
            curr_tokens = curr_tokens.unsqueeze(1).to(self.device)
            curr_tokens_mask = torch.cat(
                [torch.zeros_like(audio_tokens).bool(), torch.ones(bs_size, 1).bool().to(self.device)], dim=1 ).unsqueeze(1)
            curr_pos.add_(1)
            input_pos_maxp1 += 1
        
        #print('torch.stack(text_samples, dim=-1) ', torch.stack(text_samples, dim=-1))
        text_content = self._text_tokenizer.decode(torch.stack(text_samples, dim=-1))
        # print('text_content ', text_content)
        # assert 1==2
        return text_content
    
def get_parser():
    parser = argparse.ArgumentParser()

    # model related: use the resume model if provided; otherwise use the latest in exp_dir
    parser.add_argument('--resume', type=str, default=None, help='model to resume. If None, use the latest checkpoint in exp_dir')
    parser.add_argument('--llm_train_config', type=str, default=None, help='the config file')
    parser.add_argument('--exp_dir', type=str, default=None, help='experiment directory')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='the path of tokenizer')
    parser.add_argument('--text_tokenizer_path', type=str, default=None, help='the path of text tokenizer')
    parser.add_argument('--audio_tokenizer_config', type=str, default=None, help='the audio tokenizer version')
    parser.add_argument('--audio_model_path', type=str, default=None, help='the audio detokenizer model path')
    parser.add_argument('--test_data_json', type=str, default=None, help='the json path')
    parser.add_argument('--use_cfg', type=str2bool, default=False, help="whether to use the cfg guidance")
    parser.add_argument('--cfg_scale', type=float, default=1.0, help="whether to use the value of cfg scale")
    parser.add_argument('--temperature', type=float, default=0.9, help="the sampling temperature")
    parser.add_argument('--topk', type=int, default=200, help="the top k value")
    parser.add_argument('--prompt_tokens', type=str, default=None, help="the task prompt tokens")
    parser.add_argument('--results', type=str, default=None, help='the txt path of results')

    # inference related: 
    parser.add_argument('--seed', type=int, default=888, help='random seed')

    # device related
    parser.add_argument('--rank', type=int, default=-1, help='GPU rank. -1 means CPU')
    parser.add_argument('--decode_type', type=str, default='beamsearch', help='whether use n gram')
    # data related
    parser.add_argument('--output_dir', type=str, help="tag for decoding")
    return parser 


if __name__ == '__main__':
    # from jiwer import compute_measures 
    parser = get_parser()
    args = parser.parse_args()
    llm_train_config = args.llm_train_config
    with open(llm_train_config, "r", encoding="utf-8") as f:
        train_args = yaml.safe_load(f)
        train_args = argparse.Namespace(**train_args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    args.rank = args.rank - 1
    if args.rank >= 0:
        args.rank = args.rank % torch.cuda.device_count()
        device = torch.device(f'cuda:{args.rank}') 
    else:
        device = torch.device('cpu')
    print('args ', args)
    print('device ', device)
    config = ModelArgs(
        decoder_name=train_args.local_model,
        llm_pretrained_model = train_args.llm_pretrained_model,
        llm_name = train_args.llm_name,
        audio_semantic_vocab_size = train_args.audio_semantic_card,
        audio_reason_vocab_size = train_args.audio_reason_card,
        audio_num_codebooks=train_args.parallel_number-1,
        audio_embeddings_path = train_args.audio_embeddings_path,
        audio_understanding_expert_path = train_args.audio_understanding_expert_path,
    )
    model = Model_stage3(config)
    model.to(device=device) 
    resume_for_inference(args.resume, args.exp_dir, model, device) # init the model
    generator = Generator(model, train_args, audio_model_path=args.audio_model_path,
                  audio_tokenizer_config=args.audio_tokenizer_config,
                  text_tokenizer_path=args.text_tokenizer_path)
    os.makedirs(args.output_dir, exist_ok=True)
    ### tmp code   ###
    data_content = read_json(args.test_data_json)
    task_prompts = torch.load(args.prompt_tokens, map_location='cpu')
    semantic_dict = torch.load(data_content['keys']['semantic_seq'], map_location='cpu')
    reason_dict = torch.load(data_content['keys']['reason_seq'], map_location='cpu')
    text_dict = torch.load(data_content['keys']['transcription_seq'], map_location='cpu')
    task_seqs  = task_prompts[data_content['task']]
    cnt = 0
    f_out = open(args.results, 'w')
    for key in semantic_dict.keys():
        cnt += 1
        if key not in text_dict.keys():
            continue 
        tmp_task_prompt = task_seqs[0] # 直接选第一个
        tmp_text = text_dict[key]
        tmp_semantic_token = semantic_dict[key]
        tmp_semantic_token = tmp_semantic_token.transpose(0, 1) # transfer to (T, 8)
        print('tmp_semantic_token ', tmp_semantic_token.shape)
        len_audio = tmp_semantic_token.shape[0] / 12.5
        # if len_audio < 5:
        #     continue
        tmp_reason_token = reason_dict[key]
        tmp_reason_token = tmp_reason_token.transpose(0, 1) # transfer to (T, 8)
        gt_text = generator._text_tokenizer.decode(tmp_text)
        # print('gt_text ', gt_text)
        task_prompt = generator._text_tokenizer.decode(tmp_task_prompt)
        if args.decode_type == 'ngram':
            text_content = generator.generate_asr_with_ngram_sampling(
                task_name=data_content['task'], task_prompt = tmp_task_prompt,
                text_token=tmp_text, semantic_token = tmp_semantic_token, reason_token= tmp_reason_token,
                temperature = args.temperature, topk = args.topk, cfg_scale=args.cfg_scale
            )
        elif args.decode_type == 'beamsearch':
            text_content = generator.generate_asr_beam_search(
                task_name=data_content['task'], task_prompt = tmp_task_prompt,
                text_token=tmp_text, semantic_token = tmp_semantic_token, reason_token= tmp_reason_token
            )
        else:
            # greedy search
            text_content = generator.generate_asr(task_name=data_content['task'], task_prompt = tmp_task_prompt,
                                        text_token=tmp_text, semantic_token = tmp_semantic_token, reason_token= tmp_reason_token,
                                        temperature = args.temperature, topk = args.topk, cfg_scale=args.cfg_scale)
        #post_text = post_process_text(text_content)
        f_out.write(key+'\t'+text_content+'\t'+gt_text+'\n')

