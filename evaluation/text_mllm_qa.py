# build the text ability evaluation

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
from typing import List, Dict, Any

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

    def combine_muq_tag_text(self, muq_seq, tag_seq, text_seq):
        ''' muq_seq: (5, D); ''' 
        sequence, seq_mask, continuous_segment = [], [], []
        num_embeddings = muq_seq.shape[0]
        idx = random.randint(0, num_embeddings-1)
        pad_seq = torch.ones(1, self.parallel_number)*self.continuous_token # using a special token to represent the continuous
        pad_seq[:,-1] = 0 # the last token is the text token, we set it as 0, denotes the text empty
        
        this_mu_mask = torch.zeros((pad_seq.shape[0], self.parallel_number))
        this_mu_mask[:,:-1] = True # 
        sequence.append(pad_seq)
        seq_mask.append(this_mu_mask)
        start_tag_tokens = torch.tensor([128000, 17224, 29, 128001])  # <tag> token
        end_tag_tokens = torch.tensor([128000, 524, 4681, 29, 128001])  # </tag> token
        if tag_seq is None:
            tag_seq = torch.tensor([])
        tag_seq = torch.cat([start_tag_tokens, tag_seq, end_tag_tokens])
        this_tag_data = self.text_pad(tag_seq)
        this_tag_mask = torch.zeros((this_tag_data.shape[0], self.parallel_number))
        this_tag_mask[:,-1] = True
        sequence.append(this_tag_data)
        seq_mask.append(this_tag_mask)

        ''' lyric '''
        this_text_data = self.text_pad(text_seq)
        this_text_mask = torch.zeros((this_text_data.shape[0], self.parallel_number))
        this_text_mask[:,-1] = True
        sequence.append(this_text_data)
        seq_mask.append(this_text_mask)

        sequence = torch.cat(sequence, dim=0).to(torch.int64)
        seq_mask = torch.cat(seq_mask, dim=0)
        return sequence, seq_mask, muq_seq[idx,:].unsqueeze(0)

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

    def get_text_token(self, user_prompt):
        return self._text_tokenizer.tokenize(user_prompt)

    def prepare_text_task(self, task_prompt):
        ''' we add the start of transcirption as the start?
        '''
        this_text_data = self.text_pad(task_prompt)
        this_text_mask = torch.zeros((this_text_data.shape[0], self.parallel_number))
        this_text_mask[:,-1] = True
        return this_text_data[:-1,:], this_text_mask[:-1,:] # remove the end token

    @torch.inference_mode()
    def generate_text(
        self,
        task_prompt,
        task_name = None,
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
        if isinstance(task_prompt, str):
            task_prompt = self.get_text_token(task_prompt)
            task_prompt = torch.tensor(task_prompt)
            print('task_prompt ', task_prompt)
        tokens, tokens_mask = self.prepare_text_task(task_prompt)
        
        prompt_tokens = tokens.to(self.device)
        prompt_tokens_mask = tokens_mask.bool().to(self.device)
        samples = []
        
        bs_size = 1
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)
        
        # pre-filling
        _ = self._model(curr_tokens[:,:-1], labels=curr_tokens[:,1:,:-1], tokens_mask=curr_tokens_mask, loss_mask=curr_tokens_mask, input_pos=curr_pos[:,:-1])
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
    from jiwer import compute_measures 
    parser = get_parser()
    args = parser.parse_args()
    train_config = args.exp_dir + '/config.yaml'
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
    #text_dict = torch.load(data_content['keys']['text_seq'], map_location='cpu')
    #cnt = 0
    # f_out = open(f"{args.output_dir}/results.txt", 'w')
    # for key in text_dict.keys():
    #     cnt += 1
    #     tmp_text = text_dict[key]
    #     gt_text = generator._text_tokenizer.decode(tmp_text)
        
    #     text_content = generator.generate_text(task_name=data_content['task'], task_prompt = tmp_text,
    #                                 temperature = args.temperature, topk = args.topk, cfg_scale=args.cfg_scale)
    #     f_out.write(key+'\t'+gt_text+'\t'+text_content+'\n')

    tmp_prompt = '你了解音乐生成嘛 '
    text_content = generator.generate_text(task_prompt = tmp_prompt,
                                     temperature = args.temperature, 
                                     topk = args.topk, cfg_scale=args.cfg_scale)
    print('text_content ', text_content)