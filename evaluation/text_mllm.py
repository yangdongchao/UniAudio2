import torch
import torch.nn as nn
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
from datasets import load_dataset

def load_text_tokenizer(tokenizer_checkpoint_path):
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer = TextTokenizer(checkpoint_dir=tokenizer_checkpoint_path)
    # bos = tokenizer.bos_id
    # eos = tokenizer.eos_id
    return tokenizer

# ==============================================================================
# ==============================================================================
# 2. 核心适配器函数：将文本 TokenID 转换为 Model_stage3 的五维输入
def prepare_text_input_for_stage3(token_ids: torch.Tensor, num_cb: int, device: torch.device, max_length: int) -> Dict[str, torch.Tensor]:
    """
    将标准的 (B, S) 文本 Token ID 转换为 Model_stage3 forward 所需的五维输入张量。
    
    由于是纯文本任务 (MMLU)，我们将所有音频相关的输入设为零/忽略。
    """
    B, S = token_ids.size()
    S = min(S, max_length) # 截断
    
    # --- 1. tokens (B, S, num_cb + 1) ---
    # 最后一列是文本 Token ID，其余 num_cb 列是音频 Token ID (设为 0)
    tokens = torch.zeros((B, S, num_cb + 1), dtype=torch.long, device=device)
    tokens[:, :, -1] = token_ids[:, :S] # 填充文本
    
    # --- 2. labels (B, S, num_cb) ---
    # 纯文本任务，labels 也是音频相关，设为 0
    labels = torch.zeros((B, S, num_cb), dtype=torch.long, device=device)
    
    # --- 3. tokens_mask (B, S, num_cb + 1) ---
    # 只有文本流有效 (最后一列为 1)
    tokens_mask = torch.zeros((B, S, num_cb + 1), dtype=torch.long, device=device)
    tokens_mask[:, :, -1] = 1 # 文本流 mask 设为 1
    
    # --- 4. loss_mask (B, S, num_cb) ---
    # 纯文本任务，loss_mask 也是音频相关，设为 0
    loss_mask = torch.zeros((B, S, num_cb), dtype=torch.long, device=device)

    # --- 5. input_pos (B, S) ---
    input_pos = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)
    # print('tokens ', tokens.shape)
    # print('labels ', labels.shape)
    # print('tokens_mask ', tokens_mask.shape)
    return {
        'tokens': tokens,
        'labels': labels,
        'tokens_mask': tokens_mask,
        'loss_mask': loss_mask,
        'input_pos': input_pos
    }

# ==============================================================================
# 3. MMLU Log-Likelihood 计算函数 (适配 Model_stage3)
def get_log_likelihood_choice(model: Model_stage3, tokenizer: Any, context: str, continuation: str, max_length: int, device):
    """
    计算 continuation 的总对数似然 (LL)
    LL(continuation | context)，适配 Model_stage3 的输入。
    """
    # --- 1. Tokenize 并转换为适配器输入 ---
    
    # Tokenize 完整文本 (Context + Continuation)
    full_text = context + ' ' + continuation
    #print('full_text ', type(full_text))
    # print('context ', context)
    # print('continuation ', full_text)
    # assert 1==2
    tokenized = tokenizer.tokenize(full_text)[:-1]
    input_ids = torch.tensor(tokenized).to(device).unsqueeze(0)
    #print('input_ids ', input_ids)
    #input_ids = tokenized.input_ids.to(model.device)
    
    # 计算 context 的长度 (用于确定续写 LL 的起始点)
    context_token_ids = tokenizer.tokenize(context)[:-1]
    context_token_ids = torch.tensor(context_token_ids).to(device).unsqueeze(0)
    continuation_start_index = context_token_ids.size(1) 
    # print('context_token_ids ', context_token_ids)
    # assert 1==2
    # 准备适配器输入 (B=1)
    inputs = prepare_text_input_for_stage3(input_ids, model.config.audio_num_codebooks, device, max_length)
    
    # --- 2. 模型前向传播 ---
    with torch.no_grad():
        # 只取 text_logits
        text_logits = model.forward_text(**inputs) 

    # --- 3. Logits 和标签处理 ---
    
    # Logits 的形状是 (1, Sequence_Length, Vocab_Size)
    # 确定续写对应的 Logits (从 context 结束后的第一个 token 的预测开始)
    # Logits 对应 input_ids[:-1]
    #print('text_logits ', text_logits.shape)
    # Logits 对应预测的 token 从 index=1 开始
    # 续写 Logits 从 (continuation_start_index - 1) 处开始计算
    shift_logits = text_logits[:, continuation_start_index-1:-1, :].contiguous()
    
    # 续写标签 (从 input_ids 的 continuation_start_index 处开始)
    shift_labels = input_ids[:, continuation_start_index:].contiguous()
    #print('shift_labels ', shift_labels)
    # --- 4. 计算总对数似然 (LL) ---
    log_probs = torch.nn.functional.log_softmax(shift_logits.squeeze(0), dim=-1)
    
    # print('shift_logits ', shift_logits.shape)
    # print('shift_labels ', shift_labels.shape)
    # 提取每个目标 token 的对数概率
    target_log_probs = log_probs.gather(1, shift_labels.squeeze(0).unsqueeze(1)).squeeze(1)
    
    # 返回总对数似然 (求和)
    return target_log_probs.sum().item()


# ==============================================================================
# 4. MMLU 评估主循环
def run_mmlu_evaluation(model, tokenizer, subtask, max_ctx_len, device):
    
    # --- 2. 加载数据集 ---
    try:
        #mmlu_dataset = load_dataset("cais/mmlu", subtask, split="test")
        mmlu_dataset = load_dataset("ai2_arc", "ARC-Easy", split="test")
        print(f"Dataset {subtask} loaded successfully.")
    except Exception as e:
        print(f"❌ MMLU数据集加载失败 (确保 datasets >= 4.0 已安装): {e}")
        return

    # --- 3. 评估循环 ---
    choice_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct_predictions = 0
    total_questions = 0

    print(f"\nStarting MMLU evaluation on {subtask}...")

    for example in mmlu_dataset:
        question = example['question']
        choices = example['choices']
        if 'answer' in example.keys():
            true_answer_index =  example['answer'] # 
        else:
            true_answer_index = example['answerKey']
        
        # Zero-shot Prompt 格式 (注意：这可能需要根据您的 LLM 实际训练格式调整)
        prompt_context = f"{question}\n\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nAnswer:"
        
        log_likelihoods = []
        
        # 遍历所有选项 (A, B, C, D)
        for i in range(len(choices)):
            # 续写只包含模型预测的答案字母 (例如 "A")
            continuation = choice_map[i]
            # try:
            ll = get_log_likelihood_choice(model, tokenizer, prompt_context, continuation, max_ctx_len, device)
            log_likelihoods.append(ll)
            # except Exception as e:
            #     # 捕获可能的 Tokenizer 或模型输入错误，跳过该样本
            #     print(f"  Warning: Skipping sample due to error: {e}")
            #     log_likelihoods.append(-float('inf')) # 确保跳过的样本不会被选为最佳答案
        
        # 找到对数似然最高的选项
        predicted_answer_index = log_likelihoods.index(max(log_likelihoods))
        
        # 计分
        if predicted_answer_index == true_answer_index:
            correct_predictions += 1
        
        total_questions += 1
        
        if total_questions % 50 == 0:
            accuracy = (correct_predictions / total_questions) * 100
            print(f"Processed {total_questions} | Current Accuracy: {accuracy:.2f}%")

    # --- 4. 最终结果 ---
    if total_questions > 0:
        final_accuracy = (correct_predictions / total_questions) * 100
        print(f"\n--- MMLU Results ({subtask}) ---")
        print(f"Total Questions: {total_questions}")
        print(f"Final Accuracy: {final_accuracy:.2f}%")
    else:
        print("No valid questions were processed.")

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


# ==============================================================================
# 5. 主执行函数
if __name__ == '__main__':
    # ⚠️ 请修改以下路径和参数为您的实际值 ⚠️
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
    text_tokenizer = load_text_tokenizer(args.text_tokenizer_path)
    resume_for_inference(args.resume, args.exp_dir, model, device) # init the model

    EVAL_SUBTASK = "all" # 选择您想评估的 MMLU 子任务
    MAX_CONTEXT_LENGTH = 1024 # 根据您的 LLM backbone 设定

    run_mmlu_evaluation(
        model=model,
        tokenizer=text_tokenizer,
        subtask=EVAL_SUBTASK,
        max_ctx_len=MAX_CONTEXT_LENGTH,
        device=device
    )
