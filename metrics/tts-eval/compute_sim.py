#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算 speaker similarity score
基于 WavLM 模型
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


def WavLM_SV(ge_audio, ref_audio, feature_extractor, model, device):
    """
    使用 WavLM 计算两个音频的 speaker similarity
    
    Args:
        ge_audio: 生成的音频（numpy array）
        ref_audio: 参考音频（numpy array）
        feature_extractor: Wav2Vec2FeatureExtractor
        model: WavLMForXVector model
        device: torch device
    
    Returns:
        similarity score (float)
    """
    audio = [ge_audio, ref_audio]
    inputs = feature_extractor(audio, padding=True, return_tensors="pt", sampling_rate=16000)
    for key in inputs.keys():
        inputs[key] = inputs[key].to(device)
    
    with torch.no_grad():
        embeddings = model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(embeddings[0], embeddings[1])
    
    return similarity


def calculate_speaker_similarity(prompt_scp, gen_wav_dir, model_path, device='cuda:0', output_file=None):
    """
    计算 speaker similarity
    
    Args:
        prompt_scp: prompt.scp 文件路径（格式：item_name path_to_prompt_audio）
        gen_wav_dir: 生成的音频目录
        model_path: WavLM 模型路径
        device: 设备（cuda:0 或 cpu）
        output_file: 可选，输出详细结果的文件路径
    
    Returns:
        mean similarity score
    """
    # 加载模型
    print(f"Loading WavLM model from {model_path}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_path,
        sampling_rate=16000  # 指定采样率，避免警告
    )
    model = WavLMForXVector.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # 读取 prompt.scp
    print(f"Loading prompt.scp from {prompt_scp}...")
    prompt_dict = {}
    with open(prompt_scp, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                item_name = parts[0].strip()
                prompt_wav_path = parts[1].strip()
                prompt_dict[item_name] = prompt_wav_path
    
    print(f"Loaded {len(prompt_dict)} items from prompt.scp")
    
    # 查找生成的音频文件
    gen_wav_dir_path = Path(gen_wav_dir)
    if not gen_wav_dir_path.exists():
        raise FileNotFoundError(f"Generated audio directory not found: {gen_wav_dir_path}")
    
    # 支持的音频格式
    audio_exts = ['.wav', '.flac', '.mp3', '.m4a', '.ogg']
    
    similarity_scores = []
    detailed_results = []
    missing_gen = []
    missing_prompt = []
    error_items = []
    
    print("\nComputing speaker similarity...")
    for item_name, prompt_wav_path in tqdm(prompt_dict.items(), desc="Processing"):
        # 检查 prompt 音频是否存在
        if not os.path.exists(prompt_wav_path):
            missing_prompt.append(item_name)
            continue
        
        # 查找生成的音频文件
        gen_wav_path = None
        for ext in audio_exts:
            candidate = gen_wav_dir_path / f"{item_name}{ext}"
            if candidate.exists():
                gen_wav_path = str(candidate)
                break
        
        if not gen_wav_path:
            missing_gen.append(item_name)
            continue
        
        try:
            # 加载音频（重采样到 16kHz）
            ref_audio, _ = librosa.load(prompt_wav_path, sr=16000)
            gen_audio, _ = librosa.load(gen_wav_path, sr=16000)
            
            # 计算相似度
            similarity = WavLM_SV(gen_audio, ref_audio, feature_extractor, model, device)
            similarity_score = similarity.item()
            
            similarity_scores.append(similarity_score)
            detailed_results.append({
                'item_name': item_name,
                'gen_wav': gen_wav_path,
                'prompt_wav': prompt_wav_path,
                'similarity': similarity_score,
            })
        except Exception as e:
            print(f"Error processing {item_name}: {e}")
            error_items.append(item_name)
            continue
    
    if not similarity_scores:
        raise RuntimeError("No valid similarity scores computed!")
    
    # 计算统计信息
    scores_array = np.array(similarity_scores)
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    median_score = np.median(scores_array)
    
    # 打印结果
    print("\n" + "="*70)
    print("Speaker Similarity Statistics")
    print("="*70)
    print(f"Total items processed:  {len(similarity_scores)}")
    print(f"Missing generated:      {len(missing_gen)}")
    print(f"Missing prompt:         {len(missing_prompt)}")
    print(f"Error items:            {len(error_items)}")
    print("-"*70)
    print(f"Mean (Average):         {mean_score:.6f}")
    print(f"Median:                 {median_score:.6f}")
    print(f"Std Deviation:          {std_score:.6f}")
    print(f"Min:                    {min_score:.6f}")
    print(f"Max:                    {max_score:.6f}")
    print("="*70)
    print(f"\nAverage Speaker Similarity: {mean_score:.6f} ({mean_score*100:.2f}%)")
    
    # 保存详细结果（如果指定）
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("item_name\tgen_wav\tprompt_wav\tsimilarity\n")
            for result in detailed_results:
                f.write(f"{result['item_name']}\t{result['gen_wav']}\t{result['prompt_wav']}\t{result['similarity']:.6f}\n")
        print(f"\nDetailed results saved to: {output_file}")
    
    return mean_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute speaker similarity score")
    parser.add_argument(
        '--prompt_scp',
        type=str,
        required=True,
        help="Prompt scp file (item_name path_to_prompt_audio)"
    )
    parser.add_argument(
        '--gen_wav_dir',
        type=str,
        required=True,
        help="Directory containing generated audio files"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='',
        help="Path to WavLM model (default: wavlm-base-plus-sv)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help="Device (cuda:0, cuda:1, or cpu)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Optional: output file to save detailed results"
    )
    
    args = parser.parse_args()
    
    similarity_score = calculate_speaker_similarity(
        args.prompt_scp,
        args.gen_wav_dir,
        args.model_path,
        args.device,
        args.output
    )
    
    print(f"\nFinal Average Speaker Similarity: {similarity_score:.6f}")
