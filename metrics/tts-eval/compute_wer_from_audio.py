#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对两个音频目录分别进行 Whisper 转录，然后计算 WER
输入：
  - gen_audio_dir: 生成的音频目录
  - gt_audio_dir: GT 音频目录
输出：
  - WER 统计信息
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json

try:
    import whisper
except ImportError:
    print("错误: 未安装whisper库，请运行: pip install openai-whisper")
    exit(1)

try:
    import editdistance as ed
except ImportError:
    print("警告: 未安装editdistance，无法计算WER。请运行: pip install editdistance")
    exit(1)


def load_whisper_model(model_size: str = "large-v3", device: str = "cuda"):
    """加载Whisper模型"""
    print(f"加载Whisper模型: {model_size}，设备: {device}")
    model = whisper.load_model(model_size, device=device)
    return model


def transcribe_audio_with_whisper(model, audio_path: str, language: str = None) -> str:
    """使用Whisper转录音频"""
    try:
        options = {
            "fp16": True,
            "language": language,
            "task": "transcribe",
            "verbose": False,
        }
        result = model.transcribe(audio_path, **options)
        return result["text"].strip()
    except Exception as e:
        print(f"  转录失败 {audio_path}: {e}")
        return ""


def find_audio_files(folder_path: str, extensions: list = None) -> dict:
    """
    查找文件夹中的所有音频文件，返回 {stem: full_path} 字典
    
    Args:
        folder_path: 文件夹路径
        extensions: 音频文件扩展名列表
    
    Returns:
        字典: {文件名（不含扩展名）: 完整路径}
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus']
    
    audio_dict = {}
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"错误: 文件夹不存在: {folder_path}")
        return audio_dict
    
    for ext in extensions:
        for audio_file in folder_path.glob(f"*{ext}"):
            if audio_file.is_file():
                stem = audio_file.stem  # 文件名（不含扩展名）
                if stem not in audio_dict:
                    audio_dict[stem] = str(audio_file)
    
    print(f"在 {folder_path} 中找到 {len(audio_dict)} 个音频文件")
    return audio_dict


def normalize_text(text: str, language: str = "en") -> str:
    """
    标准化文本（用于WER计算）
    
    Args:
        text: 原始文本
        language: 语言代码
    
    Returns:
        标准化后的文本
    """
    # 转换为小写
    text = text.lower()
    
    # 移除标点符号（保留空格）
    import string
    if language == "en":
        # 英文：移除标点，保留字母和数字
        text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    else:
        # 其他语言：移除标点
        text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    
    # 规范化空格
    text = ' '.join(text.split())
    
    return text


def calculate_wer(hypothesis: str, reference: str, language: str = "en") -> tuple:
    """
    计算 WER (Word Error Rate)
    
    Args:
        hypothesis: 预测文本
        reference: 参考文本
        language: 语言代码
    
    Returns:
        (wer, distance, ref_token_count, hyp_token_count)
    """
    # 标准化文本
    ref_normalized = normalize_text(reference, language)
    hyp_normalized = normalize_text(hypothesis, language)
    
    # 分词
    if language == "zh":
        # 中文：按字符分割
        ref_tokens = list(ref_normalized.replace(' ', ''))
        hyp_tokens = list(hyp_normalized.replace(' ', ''))
    else:
        # 英文等：按空格分割
        ref_tokens = ref_normalized.split()
        hyp_tokens = hyp_normalized.split()
    
    # 计算编辑距离
    distance = ed.eval(ref_tokens, hyp_tokens)
    
    # 计算 WER
    wer = distance / len(ref_tokens) if ref_tokens else 0.0
    
    return wer, distance, len(ref_tokens), len(hyp_tokens)


def main():
    parser = argparse.ArgumentParser(description='对两个音频目录进行Whisper转录并计算WER')
    parser.add_argument('--gen_audio_dir', type=str, required=True,
                        help='生成的音频目录')
    parser.add_argument('--gt_audio_dir', type=str, required=True,
                        help='GT音频目录')
    parser.add_argument('--model_size', type=str, default='large-v3',
                        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v3'],
                        help='Whisper模型大小')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备类型')
    parser.add_argument('--language', type=str, default=None,
                        help='语言代码（如en, zh等），None表示自动检测')
    parser.add_argument('--output', type=str, default=None,
                        help='输出详细结果文件路径（JSON格式）')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Whisper转录 + WER计算")
    print("="*70)
    
    # 1. 加载Whisper模型
    print("\n1. 加载Whisper模型...")
    model = load_whisper_model(args.model_size, args.device)
    
    # 2. 查找音频文件
    print("\n2. 查找音频文件...")
    gen_audio_dict = find_audio_files(args.gen_audio_dir)
    gt_audio_dict = find_audio_files(args.gt_audio_dir)
    
    if not gen_audio_dict:
        print("错误: 生成的音频目录中没有找到音频文件")
        return
    
    if not gt_audio_dict:
        print("错误: GT音频目录中没有找到音频文件")
        return
    
    # 3. 匹配对应的音频文件
    print("\n3. 匹配对应的音频文件...")
    common_items = sorted(set(gen_audio_dict.keys()) & set(gt_audio_dict.keys()))
    only_gen = sorted(set(gen_audio_dict.keys()) - set(gt_audio_dict.keys()))
    only_gt = sorted(set(gt_audio_dict.keys()) - set(gen_audio_dict.keys()))
    
    print(f"  共同项目: {len(common_items)}")
    if only_gen:
        print(f"  仅在生成目录中: {len(only_gen)} (显示前10个): {only_gen[:10]}")
    if only_gt:
        print(f"  仅在GT目录中: {len(only_gt)} (显示前10个): {only_gt[:10]}")
    
    if not common_items:
        print("错误: 没有找到匹配的音频文件")
        return
    
    # 4. 转录音频
    print(f"\n4. 开始转录 {len(common_items)} 个音频文件...")
    results = []
    
    for item_name in tqdm(common_items, desc="转录进度"):
        gen_audio_path = gen_audio_dict[item_name]
        gt_audio_path = gt_audio_dict[item_name]
        
        # 转录生成的音频
        gen_text = transcribe_audio_with_whisper(model, gen_audio_path, args.language)
        
        # 转录GT音频
        gt_text = transcribe_audio_with_whisper(model, gt_audio_path, args.language)
        
        # 计算WER
        if gen_text and gt_text:
            wer, distance, ref_tokens, hyp_tokens = calculate_wer(
                gen_text, gt_text, args.language or "en"
            )
        else:
            wer, distance, ref_tokens, hyp_tokens = None, None, None, None
        
        results.append({
            'item_name': item_name,
            'gen_audio': gen_audio_path,
            'gt_audio': gt_audio_path,
            'gen_text': gen_text,
            'gt_text': gt_text,
            'wer': wer,
            'distance': distance,
            'ref_token_count': ref_tokens,
            'hyp_token_count': hyp_tokens,
        })
    
    # 5. 计算统计信息
    print("\n5. 计算WER统计信息...")
    valid_results = [r for r in results if r['wer'] is not None]
    
    if not valid_results:
        print("错误: 没有有效的转录结果来计算WER")
        return
    
    # 总体WER
    total_distance = sum(r['distance'] for r in valid_results)
    total_ref_tokens = sum(r['ref_token_count'] for r in valid_results)
    overall_wer = total_distance / total_ref_tokens if total_ref_tokens > 0 else 0.0
    
    # 平均WER
    avg_wer = np.mean([r['wer'] for r in valid_results])
    median_wer = np.median([r['wer'] for r in valid_results])
    std_wer = np.std([r['wer'] for r in valid_results])
    min_wer = np.min([r['wer'] for r in valid_results])
    max_wer = np.max([r['wer'] for r in valid_results])
    
    # 打印结果
    print("\n" + "="*70)
    print("WER统计结果")
    print("="*70)
    print(f"总样本数:           {len(results)}")
    print(f"有效样本数:         {len(valid_results)}")
    print(f"总参考词数:         {total_ref_tokens}")
    print(f"总编辑距离:         {total_distance}")
    print("-"*70)
    print(f"总体WER:            {overall_wer:.6f} ({overall_wer*100:.2f}%)")
    print(f"平均WER:            {avg_wer:.6f} ({avg_wer*100:.2f}%)")
    print(f"中位数WER:          {median_wer:.6f} ({median_wer*100:.2f}%)")
    print(f"标准差:             {std_wer:.6f}")
    print(f"最小WER:            {min_wer:.6f} ({min_wer*100:.2f}%)")
    print(f"最大WER:            {max_wer:.6f} ({max_wer*100:.2f}%)")
    print("="*70)
    
    # 6. 保存详细结果（如果指定）
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'metadata': {
                'gen_audio_dir': args.gen_audio_dir,
                'gt_audio_dir': args.gt_audio_dir,
                'model_size': args.model_size,
                'language': args.language,
                'total_samples': len(results),
                'valid_samples': len(valid_results),
            },
            'statistics': {
                'overall_wer': overall_wer,
                'average_wer': float(avg_wer),
                'median_wer': float(median_wer),
                'std_wer': float(std_wer),
                'min_wer': float(min_wer),
                'max_wer': float(max_wer),
                'total_distance': total_distance,
                'total_ref_tokens': total_ref_tokens,
            },
            'samples': results,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细结果已保存到: {args.output}")
    
    print(f"\n✅ 处理完成!")
    print(f"总体WER: {overall_wer:.6f} ({overall_wer*100:.2f}%)")


if __name__ == "__main__":
    main()
