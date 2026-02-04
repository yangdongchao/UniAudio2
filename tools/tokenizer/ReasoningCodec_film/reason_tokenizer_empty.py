import os
import sys
from omegaconf import OmegaConf
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from tools.tokenizer.abs_tokenizer import AbsTokenizer
import torchaudio
import yaml
import argparse
import os 
import glob

class ReasoningTokenizer(AbsTokenizer):
    def __init__(self, device=torch.device('cpu')):
        super(ReasoningTokenizer, self).__init__()
        
        self.sample_rate = 24000
        self.device = device
        self.MAX_DURATION = 360
        self.n_codebook = 8
        self.sq_codec_hz = 25 # the frame-rate of SQCodec
        self.rec_frame_rate = 12.5 # 
        self.reason_frame_rate = 5 
        
    def find_length(self, x):
        '''x: '''
        return x.shape[1]

    def tokenize2(self, token):
        '''Token: '''
        if isinstance(token, torch.Tensor):
            return token.to(torch.int64).transpose(0, 1)
        else:
            raise NotImplementedError
    
    @property
    def is_discrete(self):
        return True
    
if __name__ == '__main__':
    pass