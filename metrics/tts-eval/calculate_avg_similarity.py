#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算 speaker similarity 结果文件的平均值和统计信息
输入文件格式：音频路径1|音频路径2 \t 相似度分数
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Calculate average speaker similarity score')
    parser.add_argument('--input', type=str, required=True,
                        help='Input .wer file containing similarity scores')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional: output file to save statistics')
    
    args = parser.parse_args()
    
    scores = []
    
    # 读取分数
    with open(args.input, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # 跳过统计行（如果存在）
            if line.startswith('ASV:') or line.startswith('ASV-var:'):
                continue
            
            # 解析分数（格式：路径1|路径2 \t 分数）
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    score = float(parts[-1].strip())
                    scores.append(score)
                except ValueError:
                    print(f"Warning: Could not parse score on line {line_num}: '{line[:100]}'")
                    continue
    
    if not scores:
        print("Error: No valid scores found in the file!")
        return
    
    scores_array = np.array(scores)
    
    # 计算统计信息
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    median_score = np.median(scores_array)
    
    # 打印结果
    print("\n" + "="*70)
    print("Speaker Similarity Statistics")
    print("="*70)
    print(f"Total items:        {len(scores)}")
    print(f"Mean (Average):    {mean_score:.6f}")
    print(f"Median:            {median_score:.6f}")
    print(f"Std Deviation:     {std_score:.6f}")
    print(f"Min:               {min_score:.6f}")
    print(f"Max:               {max_score:.6f}")
    print("="*70)
    print(f"\nAverage Similarity Score: {mean_score:.6f} ({mean_score*100:.2f}%)")
    
    # 保存到文件（如果指定）
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Speaker Similarity Statistics\n")
            f.write("="*70 + "\n")
            f.write(f"Total items:        {len(scores)}\n")
            f.write(f"Mean (Average):     {mean_score:.6f}\n")
            f.write(f"Median:             {median_score:.6f}\n")
            f.write(f"Std Deviation:      {std_score:.6f}\n")
            f.write(f"Min:                {min_score:.6f}\n")
            f.write(f"Max:                {max_score:.6f}\n")
            f.write("="*70 + "\n")
            f.write(f"\nAverage Similarity Score: {mean_score:.6f} ({mean_score*100:.2f}%)\n")
        print(f"\nStatistics saved to: {args.output}")


if __name__ == "__main__":
    main()
