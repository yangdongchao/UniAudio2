import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    # args for randomness
    parser.add_argument('--seed', type=int, default=2048, help='seed for initializing training. ')
    parser.add_argument('--cudnn_deterministic', default=False, action='store_true', help='set cudnn.deterministic True')
    # args for data
    parser.add_argument('--train_data_jsons', type=str, nargs="+", help="list of train data jsons, separated by comma,")
    parser.add_argument('--batch_scale', type=int, default=1000, help="summed sequence length of each batch")
    parser.add_argument('--text_batch_scale', type=int, default=1000, help="summed text sequence length of each batch")
    parser.add_argument('--max_length', type=int, default=1000, help="maximum length of each example sequence. -1 means no constraint. The real allowed length may exceed this slightly")
    parser.add_argument('--min_length', type=int, default=100, help="minimum length of each example sequence. -1 means no constraint. The real allowed length may exceed this slightly")
    parser.add_argument('--n_worker', type=int, default=4, help='number of loading workers for each GPU')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--minibatch_debug', type=int, default=-1, help="if > 0, chuncate the data iterator for debug")
    
    # args for local model
    parser.add_argument('--audio_semantic_card', type=int, default=2050, help='the audio token space of LLM')
    parser.add_argument('--audio_reason_card', type=int, default=2050, help='the audio token space of LLM')
    parser.add_argument('--local_model', type=str, default='llama-300M', help="the local model of UniAudio2.0")
    # args for save model and log: 
    parser.add_argument('--parallel_number', type=int, default=9, help='the number of training streaming')
    parser.add_argument('--reason_pad_token', type=int, default=1025, help='the pading token for semantic')
    parser.add_argument('--semantic_pad_token', type=int, default=1025, help='the pading token for semantic')
    parser.add_argument('--llm_pretrained_model', type=str, default='./ckpt.pt', help='the pretrained model for LLM')
    parser.add_argument('--llm_name', type=str, default='meta/llama3.2-3B', help='the name of llm')
    parser.add_argument('--text_tokenizer_path', type=str, default='', help='the path of text tokenizer')
    parser.add_argument('--semantic_eos', type=int, default=10000, help='the eos token for semantic')
    parser.add_argument('--semantic_bos', type=int, default=10000, help='the bos token for semantic')
    parser.add_argument('--reason_bos', type=int, default=10000, help='the bos token for reason tokens')
    parser.add_argument('--reason_eos', type=int, default=10000, help='the eos token for reason tokens')
    parser.add_argument('--audio_prompt_bos', type=int, default=8196, help='the bos token for reason tokens')
    parser.add_argument('--audio_prompt_eos', type=int, default=8197, help='the eos token for reason tokens')
    parser.add_argument('--text_pad_token', type=int, default=1025, help='the number of training streaming')
    parser.add_argument('--exp_dir', type=str, default='./log', help='directory of this experiment')
    parser.add_argument('--print_freq', type=int, default=100, help='the print frequency')
    parser.add_argument('--save_interval', type=int, default=10000, help='save a checkpoint within an epoch')
    parser.add_argument('--training_stage', type=int, default=1, help='which traning stage in the MLLM training?')
    parser.add_argument('--resume', type=str, default=None, help='whether re-train model')
    parser.add_argument('--prompt_token_path', type=str, default=None, help='the path of prompt token')
    parser.add_argument('--audio_embeddings_path', type=str, default=None, help="the audio embedding path")
    parser.add_argument('--audio_understanding_expert_path', type=str, default=None, help="the audio embedding path")
    parser.add_argument('--root_path', type=str, default=None, help='the json root path')

    # args for training / optimization
    parser.add_argument('--n_epoch', type=int, default=20, help='Total training epoch')
    parser.add_argument('--grad_accum', type=int, default=1, help='help to simulate large batch')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='The learning rate for training')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--warmup_steps', type=int, default=10000, help="step of warmup")
    parser.add_argument('--schedule', type=str, default='cosine', help="the schedule strategy of training")
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument('--data-parallel', type=str, default='fsdp', help='data parallel strategy: fsdp, sdp, hsdp. ')
    parser.add_argument('--mixed-precision', type=str, default='bf16', help='mixed precision: fp32, tf32, bf16, fp16')
    parser.add_argument('--grad-precision', type=str, default='bf16', help='gradient precision: fp32, tf32, bf16, fp16')
    parser.add_argument('--activation-checkpointing', type=bool, default=True, help='use activation checkpointing')
    parser.add_argument("--no-wandb", type=str2bool, default='true', help='whether use wandb')

    # dataloader config
    parser.add_argument('--audio_tokenizer', type=str, default='semantic', help='the type of audio tokenizer')
    parser.add_argument('--text_tokenizer', type=str, default='llama3-3B', help='the type of audio tokenizer')
    parser.add_argument('--audio_prompt_tokenizer', type=str, default='audio_prompt_tokenizer', help='the audio prompt tokenizer')
    args = parser.parse_args()
    
    return args