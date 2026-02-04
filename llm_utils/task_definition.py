import torch
import logging
import json
import random
from pathlib import Path
import os 
# For some data types that are large and can hardly be fully stored in memory,
# We do offline tokenization and save them as codec sequences. e.g., audio
def load_pt_data(f):
    return torch.load(f, map_location='cpu')

def load_text_data(f):
    lines = open(f, encoding='utf-8').readlines()
    lines = [line.strip().split() for line in lines]
    ret = {}
    for line in lines:
        if len(line) < 2:
            logging.warning(f"find an empty entry: {line}")
            continue
        example_id, ctx = line[0], " ".join(line[1:])
        ret[example_id] = ctx
    return ret

def unified_loading(f):
    """ allow both format """
    if f.endswith('.pt'):
        return load_pt_data(f)
    else:
        return load_text_data(f)

loading_methods = {
    'audio': load_pt_data,
    'audio_prompt': unified_loading,
    'text': unified_loading,
    'hybrid': unified_loading,
}
        
# 2. This part defines all valid task format.
# The data format of each task is defined as below:
# (1)   keys: data keys in order. This determines the order of the components in the sequences
# (2)   type: type of each data key. It determines the tokenizer for each data key
# (3)   features: some features belong to the examples but are not in the training sequence. e.g., speaker-id
# (4)   loss_key: key to predict. it determines which data key the loss should be computed on.
# (5)   encoder_keys: keys that are placed in encoder input when using the encoder-decoder format. 
#         Should always be the first several entry in "keys"
#         If this is set to None or [], it means encoder-decoder format is not supported, e.g., LM

text_only_format = {
    'keys': ["text_seq"],
    'type': ["text"],
    'features': [],
    'loss_key': ['text_seq']
} # text-only data will calculate all sequence loss

audio_only_format = {
    'keys': ["reason_seq", "semantic_seq"],
    'type': ["audio", "audio"],
    'features': [],
    'loss_key': ['reason_seq', 'semantic_seq']
} # pre-training audio-only: directly predict [reason_seq, semantic_seq]
reason2semantic_format = {
    'keys': ["reason_seq", "semantic_seq"],
    'type': ["audio", "audio"],
    'features': [],
    'loss_key': ['semantic_seq']
}

semantic2reason_format = {
    'keys': ["semantic_seq", "reason_seq"],
    'type': ["audio", "audio"],
    'features': [],
    'loss_key': ['reason_seq']
}

semantic_copy_format = {
    'keys': ["reason_seq", "semantic_seq", "semantic_seq2"],
    'type': ["audio", "audio", "audio"],
    'features': [],
    'loss_key': ['semantic_seq2']
}

reseaon_copy_format = {
    'keys': ["reason_seq", "semantic_seq", "reason_seq2"],
    'type': ["audio", "audio", "audio"],
    'features': [],
    'loss_key': ['reason_seq2']
}

pretraining_asr_format = {
    'keys': ["reason_seq", "semantic_seq", "transcription_seq"],
    'type': ["audio", "audio", "text"],
    'features': [],
    'loss_key': ["transcription_seq"]
} # 

pretraining_tts_format = {
    'keys': ["transcription_seq", "reason_seq", "semantic_seq"],
    'type': ["text", "audio", "audio"],
    'features': [],
    'loss_key': ["reason_seq", "semantic_seq"]
} # 

yue_asr_format = {
    'keys': ["reason_seq", "semantic_seq", "transcription_seq"],
    'type': ["audio", "audio", "text"],
    'features': [],
    'loss_key': ["transcription_seq"]
} # 

dysarthria_asr_format = {
    'keys': ["reason_seq", "semantic_seq", "transcription_seq"],
    'type': ["audio", "audio", "text"],
    'features': [],
    'loss_key': ["transcription_seq"]
} # 

yue_tts_format = {
    'keys': ["transcription_seq", "reason_seq", "semantic_seq"],
    'type': ["text", "audio", "audio"],
    'features': [],
    'loss_key': ["reason_seq", "semantic_seq"]
} # 

pretraining_instruct_tts_format = {
    'keys': ["caption_seq", "transcription_seq", "reason_seq", "semantic_seq"],
    'type': ["text", "text", "audio", "audio"],
    'features': [],
    'loss_key': ["reason_seq", "semantic_seq"]
} # 

speech_sound_generation_format = {
    'keys': ["transcription_seq", "reason_seq", "semantic_seq"],
    'type': ["text", "audio", "audio"],
    'features': [],
    'loss_key': ["reason_seq", "semantic_seq"]
} # 

prompt_instruct_tts_format = {
    'keys': ["audio_prompt_seq", "caption_seq", "transcription_seq", "reason_seq", "semantic_seq"],
    'type': ["audio_prompt", "text", "text", "audio", "audio"],
    'features': [],
    'loss_key': ["reason_seq", "semantic_seq"]
} # 

pretraining_audio_generation_format = {
    'keys': ["caption_seq", "reason_seq", "semantic_seq"],
    'type': [ "text", "audio", "audio"],
    'features': [],
    'loss_key': ["reason_seq", "semantic_seq"]
} #

pretraining_song_generation_format = {
    'keys': ["lyric_seq","reason_seq", "semantic_seq"],
    'type': ["text", "audio", "audio"],
    'features': [],
    'loss_key': ["reason_seq", "semantic_seq"]
} # 

pretraining_prompt_song_generation_format = {
    'keys': ["audio_prompt_seq","lyric_seq","reason_seq", "semantic_seq"],
    'type': ["audio_prompt", "text", "audio", "audio"],
    'features': [],
    'loss_key': ["reason_seq", "semantic_seq"]
} # 

pretraining_audio_caption_format = {
    'keys': ["reason_seq", "semantic_seq", "caption_seq"],
    'type': ["audio", "audio", "text"],
    'features': [],
    'loss_key': ["caption_seq"]
} # 

pretraining_lyric_recognition_format = {
    'keys': ["reason_seq", "semantic_seq", "lyric_seq"],
    'type': ["audio", "audio", "text"],
    'features': [],
    'loss_key': ["lyric_seq"]
} # 


speech_edit_format = {
    'keys': ["text_seq", "reason_seq_1", "semantic_seq_1", "reason_seq_2", "semantic_seq_2"],
    'type': ["text", "audio", "audio", "audio", "audio"],
    'features': [],
    'loss_key': ["reason_seq_2", "semantic_seq_2"]
} # 

speech_denoise_format = {
    'keys': ["reason_seq_1", "semantic_seq_1", "reason_seq_2", "semantic_seq_2"],
    'type': ["audio", "audio", "audio", "audio"],
    'features': [],
    'loss_key': ["reason_seq_2", "semantic_seq_2"]
} # 

speech_ss_format = {
    'keys': ["reason_seq_mix", "semantic_seq_mix", 
             "reason_seq_1", "semantic_seq_1", "reason_seq_2", "semantic_seq_2"],
    'type': ["audio", "audio", "audio", "audio", "audio", "audio"],
    'features': [],
    'loss_key': ["reason_seq_1", "semantic_seq_1", "reason_seq_2", "semantic_seq_2"]
} #

music_ss_format = {
    'keys': ["reason_seq_mix", "semantic_seq_mix", "reason_seq_1", "semantic_seq_1", "reason_seq_2", "semantic_seq_2"],
    'type': ["audio", "audio", "audio", "audio", "audio", "audio"],
    'features': [],
    'loss_key': ["reason_seq_1", "semantic_seq_1", "reason_seq_2", "semantic_seq_2"]
} #


speech_s2t_format = {
    'keys': ["reason_seq", "semantic_seq", "text_seq"],
    'type': ["audio", "audio", "text"],
    'features': [],
    'loss_key': ["text_seq"]
}
speech_s2s_format = {
    'keys': ["reason_seq_1", "semantic_seq_1", "reason_seq_2", "semantic_seq_2"],
    'type': ["audio", "audio", "audio", "audio"],
    'features': [],
    'loss_key': ["reason_seq_2", "semantic_seq_2"]
}
audio_understanding_format = {
    'keys': ["text_seq_question", "reason_seq", "semantic_seq", "text_seq_answer"],
    'type': ["text", "audio", "audio", "text"],
    'features': [],
    'loss_key': ["text_seq_answer"]
}

task_formats = {
    'ASR': pretraining_asr_format,
    "Yue_ASR": yue_asr_format,
    "D_ASR": dysarthria_asr_format,
    'lyric_recognition': pretraining_lyric_recognition_format,
    'audio_caption': pretraining_audio_caption_format,
    'music_caption': pretraining_audio_caption_format,
    'audio_only': audio_only_format,
    'text_only': text_only_format,
    "TTS": pretraining_tts_format,
    "Yue_TTS": yue_tts_format,
    "TTA": pretraining_audio_generation_format,
    "TTM": pretraining_audio_generation_format,
    "LTS": pretraining_song_generation_format,
    "InstructTTS": pretraining_instruct_tts_format,
    "reason_to_semantic": reason2semantic_format,
    "semantic_to_reason": semantic2reason_format,
    "semantic_copy": semantic_copy_format,
    "reason_copy": reseaon_copy_format,
    "speech_edit": speech_edit_format,
    "speech_denoise": speech_denoise_format,
    "speech_ss": speech_ss_format,
    "music_ss": music_ss_format,
    "speech_s2t": speech_s2t_format,
    "speech_s2s": speech_s2s_format,
    "audio_understanding": audio_understanding_format,
    "prompt_instruct_tts": prompt_instruct_tts_format
}

def load_data_for_all_tasks(json_files, root_path=None):
    """ accept and parse multiple json_files, each of which represents a task dataset"""
    data_dict = {}
    text_dict = {}
    audio_prompt_related_dict = {}
    for json_file in json_files:
        dataset_json = json.load(open(json_file)) 
        logging.info(f"loading dataset file: {json_file} for {dataset_json['task']} task") 
        print(f"loading dataset file: {json_file} for {dataset_json['task']} task") 
        task_data = load_data_for_one_task(dataset_json, root_path)     
        if dataset_json['task'] == 'text_only':
            text_dict.update(task_data)
        else:
            data_dict.update(task_data)
        
        if dataset_json['task'] in ['PromptTTS', 'PromptLTS']:
            audio_prompt_related_dict.update(task_data)
        
    logging.info(f"from all json files, we have {len(data_dict)} examples and {len(text_dict)} text only examples")
    return data_dict, text_dict, audio_prompt_related_dict

def load_data_for_one_task(dataset_json, root_path=None):
    task_type = dataset_json['task']
    if 'repeat_num' in dataset_json.keys():
        repeat_num = dataset_json['repeat_num']
    else:
        repeat_num = 1
    task_format = task_formats[task_type]
    # load data for each data key
    data_dict = {}
    for key, data_type in zip(task_format['keys'], task_format['type']):
        if key not in dataset_json['keys']:
            raise ValueError(f"For task {task_type}, data key {key} is needed but missing.")
        logging.info(f"loading file: {dataset_json['keys'][key]} as key: {key}")
        print(f"loading file: {dataset_json['keys'][key]} as key: {key}")
        if root_path is not None:
            tmp_data_path = os.path.join(root_path, dataset_json['keys'][key])
        else:
            tmp_data_path = dataset_json['keys'][key]
        this_data_dict = loading_methods[data_type](tmp_data_path)
        this_data_dict = {f"{dataset_json['task']}_{k}": v 
                for k, v in this_data_dict.items()} 
        for example_id, data in this_data_dict.items():
            if example_id not in data_dict:
                data_dict[example_id] = {}
            data_dict[example_id][key] = data
        if repeat_num > 1:
            for kk in range(repeat_num-1):
                for example_id, data in this_data_dict.items():
                    tmp_example_id = example_id + '_' + str(kk)
                    if tmp_example_id not in data_dict:
                        data_dict[tmp_example_id] = {}
                    data_dict[tmp_example_id][key] = data
    # Validate the data: remove the examples when some entries are missing.
    # add the task label after validation
    example_ids = list(data_dict.keys())
    for example_id in example_ids:
        for key in task_format['keys']:
            if key not in data_dict[example_id]:
                del data_dict[example_id]
                #logging.warning(f"{task_type} example {example_id} is removed since {key} is missing")
                break
    example_ids = list(data_dict.keys())
    for example_id in example_ids:
        data_dict[example_id]['task'] = task_type
        data_dict[example_id]['loss_key'] = task_format['loss_key']
    logging.info(f"done loading this raw data dict: {len(data_dict)} valid examples")
    print(f"done loading this raw data dict: {len(data_dict)} valid examples")
    return data_dict



if __name__ == "__main__":
    pass

