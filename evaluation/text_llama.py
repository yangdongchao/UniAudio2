import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import random
import argparse
from pathlib import Path
import yaml
import os 
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedModel

# ==============================================================================
# 0. 占位符/依赖导入 (假设这些在您的环境中可用)
# ------------------------------------------------------------------------------
# 假设这些类和函数在您的 'llm_models.model_new' 和 'llm_utils' 中
# 请确保这些导入路径与您的实际项目结构匹配
# from llm_models.model_new import Model, Model_stage1, ModelArgs, Model_stage2, Model_stage3
# from tools.tokenizer.Text2ID.text_tokenizer import TextTokenizer
# from llm_utils.arguments import str2bool
# from llm_utils.train_utils import resume_for_inference

# 为了让代码可以运行，我将使用占位符或简化的定义：
@dataclass
class ModelArgs:
    decoder_name: str = 'llama'
    llm_pretrained_model: str = ''
    llm_name: str = 'llama'
    audio_semantic_vocab_size: int = 1000
    audio_reason_vocab_size: int = 100
    audio_num_codebooks: int = 4
    audio_embeddings_path: str = ''
    audio_understanding_expert_path: str = ''

class Model_stage3(nn.Module):
    # 假设您的 Model_stage3 已经完整定义
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
    def forward_text(self, **inputs):
        # 这是一个占位符，如果 model 是 Model_stage3，则调用您的实际 forward_text
        raise NotImplementedError("Model_stage3 forward_text must be implemented or model must be Llama.")
    
# 简化版的工具函数
def load_text_tokenizer(tokenizer_checkpoint_path):
    # 假设 TextTokenizer 满足 TextTokenizer(checkpoint_dir)
    # 替换为 Llama Tokenizer 兼容逻辑
    return AutoTokenizer.from_pretrained(tokenizer_checkpoint_path)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 占位符：在实际代码中，您会从 checkpoint 加载您的模型权重
def resume_for_inference(resume_path, exp_dir, model, device):
    print(f"Skipping custom checkpoint loading for Llama verification.")
    pass 
# ------------------------------------------------------------------------------
# ==============================================================================

# 2. 核心适配器函数 (保持不变)
def prepare_text_input_for_stage3(token_ids: torch.Tensor, num_cb: int, device: torch.device, max_length: int) -> Dict[str, torch.Tensor]:
    """
    将标准的 (B, S) 文本 Token ID 转换为 Model_stage3 forward 所需的五维输入张量。
    """
    B, S = token_ids.size()
    S = min(S, max_length) 
    
    # --- 1. tokens (B, S, num_cb + 1) ---
    tokens = torch.zeros((B, S, num_cb + 1), dtype=torch.long, device=device)
    tokens[:, :, -1] = token_ids[:, :S]
    
    # --- 2. labels (B, S, num_cb) ---
    labels = torch.zeros((B, S, num_cb), dtype=torch.long, device=device)
    
    # --- 3. tokens_mask (B, S, num_cb + 1) ---
    tokens_mask = torch.zeros((B, S, num_cb + 1), dtype=torch.long, device=device)
    tokens_mask[:, :, -1] = 1
    
    # --- 4. loss_mask (B, S, num_cb) ---
    loss_mask = torch.zeros((B, S, num_cb), dtype=torch.long, device=device)

    # --- 5. input_pos (B, S) ---
    input_pos = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)

    return {
        'tokens': tokens,
        'labels': labels,
        'tokens_mask': tokens_mask,
        'loss_mask': loss_mask,
        'input_pos': input_pos
    }

# 3. MMLU Log-Likelihood 计算函数 (适配 Llama 验证)
def get_log_likelihood_choice(model: PreTrainedModel, tokenizer: Any, context: str, continuation: str, max_length: int, device):
    
    is_stage3_model = False
    
    # **【核心分词】** 使用 Llama 兼容的 encode/call，并禁用特殊 token
    full_text = context + ' ' + continuation
    
    # A. Tokenize Context 确保找到精确起始点
    context_tokenized = tokenizer(context, return_tensors="pt", truncation=False, add_special_tokens=False)
    #context_tokenized = context_tokenized[:,:-1]
    c_input_ids = context_tokenized.input_ids.to(device)
    continuation_start_index = c_input_ids.size(1)
    # print('context_tokenized ', context_tokenized)
    # print('continuation_start_index ', continuation_start_index)
    # B. Tokenize Full Text 
    full_tokenized = tokenizer(full_text, return_tensors="pt", truncation=False, add_special_tokens=False)
    input_ids = full_tokenized.input_ids.to(device)

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is not None:
        bos_tensor = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([bos_tensor, input_ids], dim=1)
        continuation_start_index += 1
    # print('input_ids ', input_ids)
    # assert 1==2
    S = input_ids.size(1)
    #print('S ', S, continuation_start_index)
    if S > max_length:
         # 截断输入，确保不超过最大长度
        input_ids = input_ids[:, :max_length]
        S = max_length
        if continuation_start_index >= S:
             # 如果截断导致续写点被移除，则跳过
            print('error ')
            return -float('inf')
    # --- 2. 模型前向传播 ---
    with torch.no_grad():
        # 标准 Llama 模型模式 (用于验证)
        outputs = model(input_ids=input_ids)
        text_logits = outputs.logits 

    # --- 3. Logits 和标签处理 ---
    # Logits 形状: (1, S_truncated, V)
    
    # 续写 Logits: 从预测第一个续写 token (索引 k-1) 开始
    # Logits 长度 S_Llama = S (等于 input_ids 长度)
    # Llama Logits[i] 预测 input_ids[i+1]
    
    # 续写 Logits: 从预测第一个续写 token 对应的 Logits 处开始
    shift_logits = text_logits[:, continuation_start_index - 1:-1, :].contiguous()
    
    # 续写 Labels: 从第一个续写 token (索引 k) 开始
    shift_labels = input_ids[:, continuation_start_index:].contiguous()
    # print('shift_labels ', shift_labels)
    # print('shift_logits ', shift_logits.shape)
    # print('shift_labels ', shift_labels.shape)
    # 再次检查长度，如果 Logits 和 Labels 长度不匹配则返回负无穷
    # if shift_logits.size(1) != shift_labels.size(1):
    #     # 调试信息: 告知用户切片可能出了问题 (特别是对于非标准分词器)
    #     print(f"❌ LL Failure: Logits {shift_logits.size(1)} != Labels {shift_labels.size(1)}")
    #     return -float('inf')

    # --- 4. 计算总对数似然 (LL) ---
    log_probs = torch.nn.functional.log_softmax(shift_logits.squeeze(0), dim=-1)
    target_log_probs = log_probs.gather(1, shift_labels.squeeze(0).unsqueeze(1)).squeeze(1)
    
    return target_log_probs.sum().item()

# 4. MMLU 评估主循环 (保持不变)
def run_mmlu_evaluation(model, tokenizer, subtask, max_ctx_len, device):
    
    try:
        # Llama 验证通常使用 'abstract_algebra' 作为快速测试
        mmlu_dataset = load_dataset("cais/mmlu", subtask, split="test")
        print(f"Dataset {subtask} loaded successfully.")
    except Exception as e:
        print(f"❌ MMLU数据集加载失败: {e}")
        return

    # ... (评估循环逻辑保持不变) ...
    choice_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct_predictions = 0
    total_questions = 0

    print(f"\nStarting MMLU evaluation on {subtask}...")

    for example in mmlu_dataset:
        question = example['question']
        choices = example['choices']
        true_answer_index = example['answer']
        
        # Zero-shot Prompt 格式 
        prompt_context = f"{question}\n\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nAnswer:"
        
        log_likelihoods = []
        
        for i in range(len(choices)):
            continuation = choice_map[i]
            
            # 使用修正后的 get_log_likelihood_choice
            ll = get_log_likelihood_choice(model, tokenizer, prompt_context, continuation, max_ctx_len, device)
            log_likelihoods.append(ll)
        
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
    # ... (parser 定义保持不变) ...
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='model to resume. If None, use the latest checkpoint in exp_dir')
    parser.add_argument('--exp_dir', type=str, default='./exp', help='experiment directory')
    parser.add_argument('--text_tokenizer_path', type=str, default='meta-llama/Meta-Llama-3-8B', help='the path of text tokenizer')
    parser.add_argument('--audio_tokenizer_config', type=str, default=None, help='the audio tokenizer version')
    parser.add_argument('--audio_model_path', type=str, default=None, help='the audio detokenizer model path')
    parser.add_argument('--test_data_json', type=str, default=None, help='the json path')
    parser.add_argument('--use_cfg', type=str2bool, default=False, help="whether to use the cfg guidance")
    parser.add_argument('--cfg_scale', type=float, default=1.0, help="whether to use the value of cfg scale")
    parser.add_argument('--temperature', type=float, default=0.9, help="the sampling temperature")
    parser.add_argument('--topk', type=int, default=200, help="the top k value")
    parser.add_argument('--prompt_tokens', type=str, default=None, help="the task prompt tokens")
    parser.add_argument('--seed', type=int, default=888, help='random seed')
    parser.add_argument('--rank', type=int, default=0, help='GPU rank. -1 means CPU')
    parser.add_argument('--output_dir', type=str, help="tag for decoding")
    return parser 

# 5. 主执行函数
if __name__ == '__main__':
    
    parser = get_parser()
    args = parser.parse_args()
    
    # ----------------------------------------------------------------------
    # ⚠️ 1. Llama 3B 模型配置 (用于验证)
    LLAMA_MODEL_NAME = "exp/ckpts/Llama-3.2-3B" # 使用 Llama 3 8B 作为可下载的标准验证模型
    EVAL_SUBTASK = "all" # 快速测试单个子任务
    MAX_CONTEXT_LENGTH = 1024 
    
    
    # Llama 验证模式
    model_type = "llama"
    print("⚠️ Running in Llama Verification Mode. Custom Stage3 loading skipped.")

    # ----------------------------------------------------------------------

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    if args.rank >= 0 and torch.cuda.is_available():
        args.rank = args.rank % torch.cuda.device_count()
        device = torch.device(f'cuda:{args.rank}') 
        print(f"Using CUDA device: cuda:{args.rank}")
    else:
        device = torch.device('cpu')
        print("Using CPU device.")


    # 加载 Llama 模型用于验证
    try:
        # 确保你有权限访问 Llama 3
        text_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
        
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model.to(device=device)
        model.eval()
        print(f"✅ Successfully loaded {LLAMA_MODEL_NAME} for verification.")

    except Exception as e:
        print(f"❌ 无法加载 Llama 模型。请检查路径或 Hugging Face 权限: {e}")
        sys.exit(1)
    
    # ----------------------------------------------------------------------

    run_mmlu_evaluation(
        model=model,
        tokenizer=text_tokenizer,
        subtask=EVAL_SUBTASK,
        max_ctx_len=MAX_CONTEXT_LENGTH,
        device=device
    )
