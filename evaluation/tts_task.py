# build the TTS task evlaution set
# (1) without the audio timbre prompt
# (2) using the audio timbre prompt

# 测试集: 
# - (libriTTS test clean)
# - seedtts-eval (主要用这个？)

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

    def prepare_tts_task_for_cfg(self, task_prompt, text_seq):
        ''' for the nagative prompt of CFG
        '''
        # (1) we define the task prompt
        task_prompt = torch.ones_like(task_prompt)*self.text_pad_token # replace the task prompt
        this_prompt_data = self.text_pad(task_prompt)
        this_prompt_mask = torch.zeros((this_prompt_data.shape[0], self.parallel_number))
        this_prompt_mask[:,-1] = True
        
        # (3) add the transcriptions
        text_seq = self.add_special_token('transcription_seq', text_seq) 
        text_seq = torch.ones_like(text_seq)*self.text_pad_token # replace the text sequence
        this_text_data = self.text_pad(text_seq)
        this_text_mask = torch.zeros((this_text_data.shape[0], self.parallel_number))
        this_text_mask[:,-1] = True
        # (4) concat all of them
        this_data = torch.cat([this_prompt_data,  this_text_data], dim=0)
        this_mask = torch.cat([this_prompt_mask, this_text_mask], dim=0)
        return this_data, this_mask


    def prepare_tts_task(self, task_prompt, text_seq):
        sequence, seq_mask = [], []
        this_prompt_data = self.text_pad(task_prompt)
        this_prompt_mask = torch.zeros((this_prompt_data.shape[0], self.parallel_number))
        this_prompt_mask[:,-1] = True

        text_seq = self.add_special_token('transcription_seq', text_seq) 
        this_text_data = self.text_pad(text_seq)
        this_text_mask = torch.zeros((this_text_data.shape[0], self.parallel_number))
        this_text_mask[:,-1] = True

        this_data = torch.cat([this_prompt_data, this_text_data], dim=0)
        this_mask = torch.cat([this_prompt_mask, this_text_mask], dim=0)
        return this_data, this_mask


    @torch.inference_mode()
    def generate_tts(
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
        tokens, tokens_mask = self.prepare_tts_task(task_prompt, text_token)
        if self.is_cfg:
            cfg_prompt_tokens, cfg_prompt_tokens_mask = self.prepare_tts_task_for_cfg(task_prompt, text_token)
            cfg_prompt_tokens = cfg_prompt_tokens.to(self.device)
            cfg_prompt_tokens_mask = cfg_prompt_tokens_mask.bool().to(self.device)
        
        prompt_tokens = tokens.to(self.device)
        prompt_tokens_mask = tokens_mask.bool().to(self.device)
        samples = []
        if self.is_cfg:
            bs_size = 2
            curr_tokens = torch.cat([prompt_tokens.unsqueeze(0), cfg_prompt_tokens.unsqueeze(0)], dim=0)
            curr_tokens_mask = torch.cat([prompt_tokens_mask.unsqueeze(0), cfg_prompt_tokens_mask.unsqueeze(0)], dim = 0)
            curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device).repeat(2, 1)
        else:
            bs_size = 1
            curr_tokens = prompt_tokens.unsqueeze(0)
            curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
            curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        
        # pre-filling
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
            if self.is_cfg:
                # 如果是cfg模式，只关注有条件生成结果
                sample = sample[0:1,:]
            text_token = sample[:, 0:1]
            audio_tokens  = sample[:, 1:]
            if torch.all(audio_tokens[0:1,:] == (self.semantic_eos+self.audio_reason_card)):
                break
            if torch.all(audio_tokens[0:1,:] == self.reason_eos):
                is_reason = False
                save_flag = False
                forbid_prefix = self.audio_reason_card
            if save_flag:
                if is_reason:
                    pre_reason_token.append(audio_tokens[0:1,:])
                else:
                    pre_semantic_token.append(audio_tokens[0:1,:]-self.audio_reason_card)
            else:
                save_flag = True # transfer to true for the next frame
            text_samples.append(text_token[0,0])
            curr_tokens = torch.cat([audio_tokens, text_token], dim=-1)
            curr_tokens = curr_tokens.unsqueeze(1).to(self.device)
            curr_tokens_mask = torch.cat([torch.ones_like(audio_tokens).bool(), torch.zeros(bs_size, 1).bool().to(self.device)], dim=1 ).unsqueeze(1)
            if self.is_cfg:
                curr_tokens = curr_tokens.repeat(2, 1, 1)
                curr_tokens_mask = curr_tokens_mask.repeat(2, 1, 1)
            curr_pos.add_(1)
            input_pos_maxp1 += 1
        de_reason_tokens = torch.stack(pre_reason_token[1:]).permute(1, 2, 0).squeeze(0)
        de_semantic_tokens = torch.stack(pre_semantic_token[1:]).permute(1, 2, 0).squeeze(0)
        return de_reason_tokens, de_semantic_tokens
    
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

    # inference related: 
    parser.add_argument('--seed', type=int, default=888, help='random seed')

    # device related
    parser.add_argument('--rank', type=int, default=-1, help='GPU rank. -1 means CPU')
    # data related
    parser.add_argument('--output_dir', type=str, help="tag for decoding")
    return parser 


if __name__ == '__main__':
    #from jiwer import compute_measures 
    parser = get_parser()
    args = parser.parse_args()
    train_config = args.llm_train_config
    with open(train_config, "r", encoding="utf-8") as f:
        train_args = yaml.safe_load(f)
        train_args = argparse.Namespace(**train_args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.rank >= 0:
        args.rank = args.rank % torch.cuda.device_count()
        device = torch.device(f'cuda:{args.rank}') 
    else:
        device = torch.device('cpu')
    
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
    for key in semantic_dict.keys():
        cnt += 1
        if key not in text_dict.keys():
            continue 
        tmp_task_prompt = task_seqs[0] # 直接选第一个
        tmp_text = text_dict[key]
        tmp_semantic_token = semantic_dict[key]
        tmp_semantic_token = tmp_semantic_token.transpose(0, 1) # transfer to (T, 8)
        tmp_reason_token = reason_dict[key]
        tmp_reason_token = tmp_reason_token.transpose(0, 1) # transfer to (T, 8)
        gt_text = generator._text_tokenizer.decode(tmp_text)
        task_prompt = generator._text_tokenizer.decode(tmp_task_prompt)
        
        torch.save(tmp_semantic_token.transpose(0,1), f'{args.output_dir}/{key}_gt.pt')
        de_reason_tokens, de_semantic_tokens = generator.generate_tts(task_name=data_content['task'], task_prompt = tmp_task_prompt,
                                    text_token=tmp_text, semantic_token = tmp_semantic_token, reason_token= tmp_reason_token,
                                    temperature = args.temperature, topk = args.topk, cfg_scale=args.cfg_scale)
        torch.save(de_reason_tokens.detach().cpu(), f'{args.output_dir}/{key}_reason.pt')
        torch.save(de_semantic_tokens.detach().cpu(), f'{args.output_dir}/{key}_semantic.pt')
