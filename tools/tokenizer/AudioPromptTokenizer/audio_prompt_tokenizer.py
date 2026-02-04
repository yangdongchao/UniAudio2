import torch
import random
from tools.tokenizer.abs_tokenizer import AbsTokenizer

def clip_by_length(x, length):
    '''x: (T, 8)'''

    if x.shape[0] <= length:
        return x

    start = random.randint(0, x.shape[0] - length - 1)
    x = x[start: start + length,:]
    return x

class AudioPromptTokenizer(AbsTokenizer):
    """ This tokenizer samples a audio prompt from the given speaker 
    """
    def __init__(self, data_dict, prompt_length):
        AbsTokenizer.__init__(self)

        self.data_dict = data_dict
        self.spk2utt = self.parse_spk2utt(data_dict)
        self.prompt_length = int(prompt_length*12.5)
        self.speakers = list(self.spk2utt.keys())

    def parse_spk2utt(self, data_dict):
        spk2utt = {}
        for example_id, d in data_dict.items():
            if d['task'] not in ['PromptTTS', 'PromptLTS']:
                continue

            spk = d['audio_prompt_seq'] # audio_prompt_seq存spk
            if isinstance(spk, torch.Tensor):
                continue

            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(example_id)

        return spk2utt

    def tokenize(self, x, task=None, cache=None):
        if isinstance(x, torch.Tensor):
            return self.tokenize_audio(x)
        else:
            return self.tokenize_spk(x, task, cache)

    def tokenize2(self, x, task=None, cache=None):
        if isinstance(x, torch.Tensor):
            return self.tokenize_audio(x)
        else:
            return self.tokenize_spk(x, task, cache)

    def find_length(self, _):
        return self.prompt_length

    def tokenize_audio(self, x):
        """ here x (T, 8) """
        assert x.shape[1] == 8
        if x.shape[0] > self.prompt_length:
            start = random.randint(0, x.shape[0] - self.prompt_length - 1)
            return x[start: start + self.prompt_length,:]
        else:
            return x

    def tokenize_spk(self, x, task=None, cache=None):
        """ Here x is the spk-id """
        for _ in range(5):
            uttid = random.sample(self.spk2utt[x], 1)[0]
            audio = self.data_dict[uttid]['semantic_seq'].transpose(0, 1)
            
            # if audio.shape[0] <= (self.prompt_length)//2:
            #     continue # ignore the current audio if this is too short
            return clip_by_length(audio, self.prompt_length)

        return audio


