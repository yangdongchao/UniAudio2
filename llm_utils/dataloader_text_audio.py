
import json
import os
import sys
import torch
import copy
import random
import logging
import torch.distributed as dist
from tools.tokenizer.ReasoningCodec.reason_tokenizer_empty import ReasoningTokenizer
from tools.tokenizer.Text2ID.text_tokenizer import TextTokenizer
from tools.tokenizer.AudioPromptTokenizer.audio_prompt_tokenizer import AudioPromptTokenizer
from llm_utils.task_definition import (
    load_data_for_all_tasks,
    task_formats
)

def print_log(content: str):
    logging.info(content)

def build_data_iterator(
        data_dict, text_dict, tokenizers,
        delay_step = 1, max_length = -1, min_length = -1,
        batch_scale = 1000, text_batch_scale = 1000,
        is_train = True, n_worker = 1, seed = 999,
        minibatch_debug = -1, parallel_number = 9,
        text_empty_token = 0, text_pad_token = 128003,
        semantic_eos = 10000, semantic_bos = 10001,
        semantic_pad_token = 10002,
        reason_eos = 4097, reason_bos = 4096,
        audio_prompt_bos = 8196, audio_prompt_eos = 8197,
        reason_pad_token = 4099,
        audio_semantic_card = 1000,
        audio_reason_card = 1000,
        prompt_tokens = None,
    ):
    find_all_length(data_dict, tokenizers) # get the length for audio data
    find_all_length(text_dict, tokenizers) # get the length for text-only data
    valid_utts = filter_data(data_dict, max_length, min_length) 
    logging.info("begin text")
    valid_text_utts = filter_data(text_dict, max_length, min_length) 
    batches = batchfy(data_dict, valid_utts, text_dict, valid_text_utts, batch_scale, text_batch_scale) # prepare batch
    logging.info(f"Finish pre-process all data. {len(valid_utts)} examples and {len(batches)} batches")
    all_data_dict = {}
    all_data_dict.update(data_dict)
    all_data_dict.update(text_dict) # merge the text and others data
    if minibatch_debug > 0:
        batches = batches[:min(minibatch_debug, len(batches))]
        logging.info(f"only use {len(batches)} as this is a debug mode")
    dataset = Dataset(batches, all_data_dict)
    sampler = DDPSyncSampler(size=len(batches), seed=seed, is_train=is_train)
    if prompt_tokens is not None:
        prompt_tokens = torch.load(prompt_tokens, map_location='cpu')
    # Build iterator. No multi-process when debug
    collate_fn = Collate_Fn_Factory(
            tokenizers = tokenizers,
            max_length=max_length if max_length > 0 else 15000,
            delay_step = delay_step, parallel_number = parallel_number,
            text_pad_token=text_pad_token,
            semantic_pad_token = semantic_pad_token,
            semantic_eos = semantic_eos, semantic_bos = semantic_bos,
            reason_eos = reason_eos, reason_bos = reason_bos,
            audio_prompt_bos = audio_prompt_bos, audio_prompt_eos = audio_prompt_eos,
            reason_pad_token = reason_pad_token, 
            audio_semantic_card = audio_semantic_card, audio_reason_card = audio_reason_card,
            is_train = is_train,
            prompt_tokens = prompt_tokens
    )
    if minibatch_debug != -1:
        iterator = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            collate_fn=collate_fn,
        )
        logging.info("disable multi-processing data loading: debug mode")
    else:
        # debug 
        iterator = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=n_worker,
            prefetch_factor=min(100, len(batches)),
            collate_fn=collate_fn,
        )
    return iterator

def rebalance_data(data_dict, valid_utts, alpha):
    # Cannot do it with num_egs of each task as the
    # average length of each task varies.
    # statistics and divide based on tasks
    utt_list_per_task = {}
    for uttid in valid_utts:
        assert uttid in data_dict
        task = data_dict[uttid]['task']
        if task not in utt_list_per_task:
            utt_list_per_task[task] = []
        utt_list_per_task[task].append(uttid)
    data_statistics = {
        'text_only': 50,
        'audio_only': 30,
        'setence_level_text_audio_interleaved': 10,
        'segment_level_audio_text_interleaved': 10,
        'word_level_audio_text_interleaved': 10,
        'word_level_audio_text_alignment': 10,
    }
    data_statistics_new = {}
    for key in utt_list_per_task.keys():
        data_statistics_new[key] = data_statistics[key]
    sum_hours = sum(list(data_statistics_new.values()))
    data_weight = {
        k: (v / sum_hours) ** alpha
        for k, v in data_statistics_new.items()
    }
    data_weight = {
        k: v / sum(list(data_weight.values()))
        for k, v in data_weight.items()
    }
    for task in utt_list_per_task.keys():
        length = len(utt_list_per_task[task])
        logging.info(f'Initially, task {task} has {length} examples')
        logging.info(f'Sampling weight of task {task} is {data_weight[task]}')
    task_list = list(data_weight.keys())
    sampling_weight = list(data_weight.values())
    resampled_utts = []
    resampled_statistics = {k: 0 for k in task_list}
    for _ in range(min(len(valid_utts), 1000000)):
        task_index = np.random.choice(
            len(task_list),
            p=sampling_weight
        )
        task = task_list[task_index]
        sampled_utt = random.choice(utt_list_per_task[task])
        resampled_utts.append(sampled_utt)
        resampled_statistics[task] += 1

    return resampled_utts

def filter_data(data_dict, max_length, min_length):
    # we find the valid key rather than remove the whole exmaple as the invalid exmaples can 
    # also work as the prompt
    keys = list(data_dict.keys())
    if max_length <= 0 and min_length <= 0:
        return keys

    valid_keys = []
    if max_length > 0:
        for k in keys:
            if  (data_dict[k]['length'] <= max_length or max_length <= 0) \
            and (data_dict[k]['length'] >= min_length or min_length <= 0):
                valid_keys.append(k)
    logging.info(f"you requires length between [{min_length}, {max_length}] so only {len(valid_keys)} examples are reserved.")
    return valid_keys

def find_all_length(data_dict, tokenizers):
    """ length found here is only for batchfy. it is not the real length as there may be more special tokens """
    for example_id, d in data_dict.items():
        data_format = task_formats[d['task']]
        length = 0
        for key, key_type in zip(data_format['keys'], data_format['type']):
            this_length = tokenizers[key_type].find_length(d[key])
            length += this_length
        d['length'] = length

def batchfy(data_dict, batch_utts, text_dict, batch_text_utts, batch_scale, text_batch_scale):
    # we should make sure each batch includes at least one text-only?
    ''' we sort the batch for text-only and others respectively. 
        We make sure the text-only data is always exists in the batch. 
        So, we will first push audio-text related data. Then we put text-only data into the last
        The real batch scale is batch_scale + text_batch_scale
    '''
    batch_utts.sort(key=lambda x: data_dict[x]['length']) # sort audio-related data
    batch_lengths = [data_dict[k]['length'] for k in batch_utts] # 

    batch_text_utts.sort(key=lambda x: text_dict[x]['length']) # sort text-realted data
    batch_text_lengths = [text_dict[k]['length'] for k in batch_text_utts]
    n_text = len(batch_text_lengths)
    
    # Only take care of the uttid rather than the whole example
    batches, batch, summed_tokens = [], [], 0
    idx = 0
    tmp_len = 0 # 一个batch里面audio相关的数量
    min_audio_len = 2048 # 
    max_size = 20 # 设置最大的sample数量为30，保证了训练初期，不要因为短音频和把显存搞爆炸
    for utt, l in zip(batch_utts, batch_lengths):
        if (tmp_len >= max_size) or (l + summed_tokens > batch_scale):
            tmp_text_len = 0
            attempts = 0
            max_attempts = 5
            # for each batch, we put text_batch_scale text-only tokens
            while (n_text > 0) and ((summed_tokens + batch_text_lengths[(idx % n_text)]) < (batch_scale+text_batch_scale)):
                idx = idx % n_text # if the text samples are less than audio samples
                text_utt = batch_text_utts[idx]
                # print('text ', batch_text_lengths[idx])
                len_text_utt = batch_text_lengths[idx]
                if (tmp_len > 0) and (len_text_utt < min_audio_len):
                    idx += max_attempts
                    len_text_utt = batch_text_lengths[idx % n_text]
                    text_utt = batch_text_utts[idx % n_text]
                summed_tokens += len_text_utt
                batch.append(text_utt)
                idx += 1
                tmp_text_len += 1
                attempts = 0
                if tmp_text_len > tmp_len+2: # 可能文本这边短的数据比较多, 如果文本这边选太多，会导致模型整个batch太大
                    break
            assert len(batch) > 0, f"batch_tokens should be larger: {batch_scale}"
            batches.append(copy.deepcopy(batch))
            batch, summed_tokens = [], 0
            tmp_len = 0
            min_audio_len = 2048
            # assert 1==2
        summed_tokens += l
        # print('audio ', l)
        batch.append(utt) # put the audio-related tokens
        tmp_len += 1 # 记录audio的item
        min_audio_len = min(l, min_audio_len) # 记录audio的最短length

    if len(batch) > 0:
        batches.append(copy.deepcopy(batch))

    # TODO: maybe report statistics
    logging.info(f'After batchfy, there are {len(batches)} batches')
    return batches 


class Dataset(torch.utils.data.Dataset):
    """ Dataset. Each example is exactly a batch """
    def __init__(self, data_split, data_dict):
        self.data_split = data_split # batches
        self.data_dict = data_dict

    def __getitem__(self, index):
        uttids = self.data_split[index]
        return [(uttid, self.data_dict[uttid]) for uttid in uttids]

    def __len__(self):
        return len(self.data_split)

class SequentialSampler(object):
    def __init__(self, sequence):
        self.seq = sequence

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

    def refresh(self):
        pass

class DDPSyncSampler(object):
    def __init__(self, size, seed, is_train=True):
        self.size = size
        self.seed = seed
        self.epoch = 0
        self.is_train = is_train

        # Ensure that data iterator aross all GPUs has the same number of batches
        if dist.is_initialized() and torch.cuda.is_available():
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device(f"cuda:{local_rank}")
            size = torch.Tensor([size]).to(device).int()
            dist.all_reduce(size, dist.ReduceOp.MAX)

            self.pad_number = size.item() - self.size
            self.rank = dist.get_rank()
        else:
            logging.warning("torch.distributed is not available!")
            self.pad_number = 0
            self.rank = 0

        self.refresh()

    def refresh(self):
        seq = list(range(self.size))

        if self.is_train:
            # Assume the batches are sorted from shortest to longest
            # This introduces local randomness by local random shuffling
            # otherwise each global batch will be identical across epochs
            chunk_size, start = 10, 0
            random.seed(self.rank + self.seed + self.epoch)
            while start < self.size:
                seg = seq[start: min(self.size, start + chunk_size)]
                local_random_order = random.sample(list(range(len(seg))), len(seg))
                seg = [seg[i] for i in local_random_order]
                seq[start: min(self.size, start + chunk_size)] = seg
                start += len(seg)

            # even after this shuffle, the batch lengths across GPUs 
            # are very similar
            random.seed(self.seed + self.epoch)
            random.shuffle(seq)

        # so the #batches are identical across GPUs
        if self.pad_number > 0:
            seq = list(range(self.pad_number)) + seq

        self.seq = seq
        self.epoch += 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

    def get_state_dict(self):
        state_dict = {
            'epoch': self.epoch,
            'seed': self.seed,
        }
        return state_dict

    def load_state_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)

class Collate_Fn_Factory(object):
    ''' We need to carefully define our special tokens
        Empty token must different with padding tokens.
        llama3 tokenizer: reserved tokens from 128002-128255
        for audio data, we need to set the special tokens
    '''
    def __init__(self, 
                 tokenizers=None,
                 max_length = 15000,
                 delay_step = 1,
                 parallel_number = 2,
                 text_pad_token = 0,
                 semantic_empty_token = 0,
                 semantic_pad_token = 0,
                 semantic_eos = 10001,
                 semantic_bos = 10002,
                 reason_eos = 4096,
                 reason_bos = 4097,
                 audio_prompt_bos = 8196, 
                 audio_prompt_eos = 8197,
                 reason_pad_token = 4099,
                 audio_semantic_card = 1000,
                 audio_reason_card = 1000,
                 is_train = False,
                 prompt_tokens = None,
    ):
        self.max_length = max_length
        self.delay_step = delay_step
        self.empty_token = 0 # all of the emtpy token can be set as 0. It cannot produce any influence
        self.text_pad_token = text_pad_token 
        self.semantic_pad_token = semantic_pad_token
        self.semantic_eos = semantic_eos
        self.semantic_bos = semantic_bos
        self.reason_eos = reason_eos
        self.reason_bos = reason_bos
        self.audio_prompt_bos = audio_prompt_bos
        self.audio_prompt_eos = audio_prompt_eos
        self.reason_pad_token = reason_pad_token
        self.audio_reason_card = audio_reason_card
        self.audio_semantic_card = audio_semantic_card
        self.parallel_number = parallel_number # how many parallel tokens
        self.tokenizers = tokenizers
        self.is_train = is_train
        self.task_prompts = prompt_tokens #  # load the prompt tokens
        self.special_token_dict = self.get_special_token()
        self.generation_tasks = ['TTS', "Yue_TTS", 'PromptTTS', 'InstructTTS', 'TTM', 'TTA', 'LTS', 'PromptLTS']

    def get_special_token(self):
        return {'<think>': 128002, '</think>': 128003, '</answer>': 128005,
                '<transcription>': 128011, '</transcription>': 128012, '<lyric>': 128013,
                '</lyric>': 128014, '<caption>': 128015, '</caption>': 128016, '<answer>': 128017,
                '<reason_token>': 128018, '<semantic_token>': 128019}
            
    def text_pad(self, x):
        '''input 1-dimension sequence. add empty token for semantic streaming.
        '''
        sequences = torch.ones((len(x), self.parallel_number)).to(torch.int64)
        sequences[:, -1] = x # the text tokens
        sequences[:,:-1] = sequences[:,:-1]*self.empty_token # we will set as 0
        return sequences

    def audio_pad(self, x):
        '''input audio (T, 4) sequence. Add empty token for text.'''
        sequences = torch.ones((x.shape[0], self.parallel_number)).to(torch.int64)*self.empty_token
        sequences[:,:-1] = x 
        return sequences

    def add_offset_semantic(self, x, offset_value):
        '''Add offset for semantic tokens'''
        x =  x + offset_value 
        return x 

    def reason_seq_bos_eos(self, this_reason_data):
        reason_bos_frame = torch.ones(1, this_reason_data.shape[1])*self.reason_bos # add bos tokens
        reason_eos_frame = torch.ones(1, this_reason_data.shape[1])*self.reason_eos # add eos tokens
        this_reason_data = torch.cat([reason_bos_frame, this_reason_data, reason_eos_frame], dim=0) # (T+2, 8)
        return this_reason_data

    def semantic_seq_bos_eos(self, this_semantic_data):
        bos_frame = torch.ones(1, this_semantic_data.shape[1])*self.semantic_bos # add bos tokens
        eos_frame = torch.ones(1, this_semantic_data.shape[1])*self.semantic_eos # add eos tokens
        this_semantic_data = torch.cat([bos_frame, this_semantic_data, eos_frame], dim=0)
        this_semantic_data = self.add_offset_semantic(this_semantic_data, self.audio_reason_card) # add the offset
        return this_semantic_data
    
    def audio_prompt_seq_bos_eos(self, this_semantic_data):
        bos_frame = torch.ones(1, this_semantic_data.shape[1])*self.audio_prompt_bos # add bos tokens
        eos_frame = torch.ones(1, this_semantic_data.shape[1])*self.audio_prompt_eos # add eos tokens
        this_semantic_data = torch.cat([bos_frame, this_semantic_data[1:-1,:], eos_frame], dim=0)
        return this_semantic_data
    
    def add_special_token(self, key, this_data):
        if key.startswith('text_seq'):
            return this_data
        key = key.replace('_seq','')
        tmp_start = self.special_token_dict['<' + key + '>']
        tmp_end = self.special_token_dict['</' + key + '>']
        bos_frame = torch.ones(1)*tmp_start # add bos tokens
        eos_frame = torch.ones(1)*tmp_end # add eos tokens
        this_data = torch.cat([bos_frame, this_data, eos_frame], dim=0)
        return this_data

    def get_drop_keys(self, d):
        '''design the whether drop-out the reason token or semantic token
           目前，我们选择先设置部分可以被drop-out的任务. 其中只设置30%的概率被drop-out
        '''
        if d['task'] in ['ASR', 'TTS', 'TTA', 'LTS', 'lyric_recognition', 
                         'audio_caption', 'audio_understanding', 'audio_understanding_cot', 'music_caption', 'audio_only']:
            p1 = random.random()
            p2 = random.random()
            if p1 > 0.7:
                if p2 >= 0.5:
                    drop_keys = ['reason_seq']
                else:
                    drop_keys = ['semantic_seq']
            else:
                drop_keys = []
        else:
            drop_keys = []
        return drop_keys

    def splice_sequence(self, d, keys, types, loss_keys, is_cfg=False):
        ''' 构建seq的函数。根据json文件中的keys顺序构造. 注意，目前数据类型只有audio 和 text两种类型
            但是audio分成两种: reason token 和 semantic token
            text又分成了多种用途: text-only, transcription, lyric, caption, instruction等
            严格来说，应该对不同用途的文本添加special token. 但在第一、二阶段的pre-training, 我们选择先不扩充词表
            To do list: 考虑生成任务加入进来后的情况
        '''
        sequence, mask, loss_mask, start = [], [], [], 0
        drop_keys = self.get_drop_keys(d)
        drop_keys = []
        if 'prompt_seq' not in keys and d['task'] not in ['text_only', 'audio_only']: 
            '''pre-training阶段，我们一般不在json里面写prompt_seq.
               但默认都需要prompt. 所以可以直接先把prompt_seq放进去. prompt seq是一定不算loss的
               注意，我们对text-only和audio_only数据不加prompt
            '''
            this_data =  random.choice(self.task_prompts[d['task']])
            if is_cfg:
                # 如果这个任务被标记为使用cfg, 那prompt部分应该被替换成padding
                this_data = torch.ones_like(this_data)*self.text_pad_token
            this_data = self.text_pad(this_data)
            this_mask = torch.zeros((this_data.shape[0], self.parallel_number))
            this_mask[:,-1] = True
            this_loss_mask = torch.zeros_like(this_mask)
            sequence.append(this_data)
            mask.append(this_mask)
            loss_mask.append(this_loss_mask)

        for key, tp in zip(keys, types):
            if key in drop_keys: # 如果是被drop out的reason token or semantic token. 我们选择直接跳过
                continue
            if key == 'prompt_seq': # 单独处理prompt sequence
                this_data =  random.choice(self.task_prompts[d['task']])
            else:
                this_data = self.tokenizers[tp].tokenize2(d[key])
            if tp == 'text':
                this_data = self.add_special_token(key, this_data) # add special tokens
                if is_cfg and key not in loss_keys:
                    this_data = torch.ones_like(this_data)*self.text_pad_token
                
                this_data = self.text_pad(this_data)
                this_mask = torch.zeros((this_data.shape[0], self.parallel_number))
                this_mask[:,-1] = True
            elif tp == 'audio_prompt':
                # firstly, we transfer it as the semantic token seq
                this_data = self.semantic_seq_bos_eos(this_data) 
                this_data = self.audio_prompt_seq_bos_eos(this_data) # add the audio_prompt_bos and audio_prompt_eos
                if is_cfg and key not in loss_keys:
                    this_data = torch.ones_like(this_data)*self.semantic_pad_token # semantic pad
                this_data = self.audio_pad(this_data) # pad to 9 streaming
                this_mask = torch.zeros((this_data.shape[0], self.parallel_number))
                this_mask[:,:-1] = True 
            else:
                #  we only define three types of data: text, audio_prompt and audio
                #  to reduce the efforts to find audio promt in advance, we design the special audio_prompt type
                if key.startswith('reason_seq'):
                    this_data = self.reason_seq_bos_eos(this_data)
                    if is_cfg and key not in loss_keys:
                        this_data = torch.ones_like(this_data)*self.reason_pad_token # semantic pad
                else:
                    this_data = self.semantic_seq_bos_eos(this_data)
                    if is_cfg and key not in loss_keys:
                        this_data = torch.ones_like(this_data)*self.semantic_pad_token # semantic pad
                this_data = self.audio_pad(this_data)
                this_mask = torch.zeros((this_data.shape[0], self.parallel_number))
                this_mask[:,:-1] = True 
            if key in loss_keys:
                this_loss_mask = this_mask
            else:
                this_loss_mask = torch.zeros_like(this_mask)
            
            sequence.append(this_data)
            mask.append(this_mask)
            loss_mask.append(this_loss_mask)
        
        sequence = torch.cat(sequence, dim=0).to(torch.int64)
        mask = torch.cat(mask, dim=0)
        loss_mask = torch.cat(loss_mask, dim=0)
        
        return sequence, mask, loss_mask, sequence.shape[0]

    def init_sequence(self, batch_size):
        sequences = torch.ones((batch_size, self.max_length+40, self.parallel_number)).long() 
        sequences[:,:,-1] = sequences[:,:,-1]*self.text_pad_token
        sequences[:,:,:-1] = sequences[:,:,:-1]*(self.semantic_pad_token+self.audio_reason_card)
        return sequences

    def decoder_only_collate_fn(self, batch):
        """Output: data and mask [B, T, L] 
          # add 40 for the task prompt
        """
        batch_size = len(batch)
        sequences = self.init_sequence(batch_size)
        masks = torch.zeros((batch_size, self.max_length+40, self.parallel_number)) #.bool() # record the loss weight
        loss_masks = torch.zeros((batch_size, self.max_length+40, self.parallel_number))
        lengths, example_ids= [], []
        for idx, (example_id, d) in enumerate(batch):
            task_format = task_formats[d['task']]
            cfg_p = random.random()
            is_cfg = False
            if d['task'] in self.generation_tasks and cfg_p < 0.1:
                # 对生成任务添加cfg训练
                is_cfg = True
            sequence, mask, loss_mask, length = self.splice_sequence(d, task_format['keys'], task_format['type'], task_format['loss_key'], is_cfg)
            sequences[idx, :sequence.shape[0], :] = sequence
            masks[idx, :mask.shape[0], :] = mask # we donot calculate loss for PADING part
            loss_masks[idx, :loss_mask.shape[0], :] = loss_mask
            lengths.append(length)
            example_ids.append(example_id)
        sequences = sequences[:, :max(lengths), :].long() # 
        masks = masks[:, :max(lengths), :]
        loss_masks = loss_masks[:,:max(lengths), :]
        lengths = torch.Tensor(lengths).long()
        return sequences, masks, loss_masks, lengths, example_ids

    def __call__(self, batch):
        assert len(batch) == 1, "batch size should only be 1"
        batch = batch[0] # a list of data
        return self.decoder_only_collate_fn(batch)

def get_data_iterator_tokenizer_vocabulary(
        args,
        train_jsons,
        batch_scale=3000,
        text_batch_scale = 3000,
        delay_step=1,
        minibatch_debug=-1,
        max_length=-1,
        min_length=-1,
        non_acoustic_repeat=1,
        n_worker=4,
        decoder_only=True,
        parallel_number=9,
        text_empty_token = 128002,
        semantic_empty_token=2048,
        semantic_pad_token = 2049,
        semantic_eos = 10000,
        semantic_bos = 10001,
        reason_eos = 4097,
        reason_bos = 4096,
        audio_prompt_bos = 8196,
        audio_prompt_eos = 8197,
        reason_pad_token = 4099,
        text_pad_token=128003,
        audio_semantic_card = 1000,
        audio_reason_card = 1000,
        seed=999,
        is_train = False,
        prompt_tokens = None,
    ):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    # (1) load all data in the raw format
    logging.info(f"loading train: {train_jsons}")
    train_data_dict, train_text_dict, train_prompt_data_dict = load_data_for_all_tasks(train_jsons, args.root_path)
    tokenizers = {}
    if args.audio_tokenizer is not None:
        audio_tokenizer = ReasoningTokenizer() # we fix it
        tokenizers['audio'] = audio_tokenizer
    else:
        audio_tokenizer = None
        logging.info(f"Did not build audio tokenizer: {args.audio_tokenizer}")
    if args.text_tokenizer is not None:
        if args.text_tokenizer == 'llama3-3B' or args.text_tokenizer == 'qwen':
            text_tokenizer = TextTokenizer(args.text_tokenizer_path)
        else:
            raise NotImplementedError(args.text_tokenizer)
        tokenizers['text'] = text_tokenizer
    else:
        text_tokenizer = None
        logging.info(f"Did not build audio tokenizer: {args.text_tokenizer}")
    
    if args.audio_prompt_tokenizer is not None:
        audio_prompt_tokenizer = AudioPromptTokenizer(train_prompt_data_dict, prompt_length=3)
        tokenizers['audio_prompt'] = audio_prompt_tokenizer
    else:
        audio_prompt_tokenizer = None
        logging.info(f"Did not build audio prompt tokenizer: {args.audio_prompt_tokenizer}")
    
    # (2) build data iterator
    train_iterator = build_data_iterator(
        train_data_dict, 
        train_text_dict,
        tokenizers,
        delay_step = delay_step, 
        max_length = max_length,
        min_length = min_length,
        batch_scale = batch_scale, 
        text_batch_scale = text_batch_scale,
        n_worker = n_worker,
        seed = seed,
        minibatch_debug = minibatch_debug,
        parallel_number = parallel_number,
        semantic_pad_token = semantic_pad_token,
        semantic_eos = semantic_eos,
        semantic_bos = semantic_bos,
        text_pad_token = text_pad_token,
        reason_eos = reason_eos,
        reason_bos = reason_bos,
        audio_prompt_bos = audio_prompt_bos,
        audio_prompt_eos = audio_prompt_eos,
        reason_pad_token = reason_pad_token,
        audio_semantic_card = audio_semantic_card,
        audio_reason_card = audio_reason_card,
        is_train = is_train,
        prompt_tokens = prompt_tokens
    )
    logging.info('all iterator built')
    return train_iterator

if __name__ == "__main__":
    get_data_iterator_tokenizer_vocabulary(sys.argv[1:2], sys.argv[2:3], n_worker=1) 
