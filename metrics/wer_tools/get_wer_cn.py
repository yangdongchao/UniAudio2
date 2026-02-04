import sys, os
from tqdm import tqdm
import multiprocessing
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string
import json
from collections import defaultdict
import numpy as np
import re

punctuation_all = punctuation + string.punctuation

def process_one(hypo, truth, lang="zh"):
    """
    处理一对文本，计算WER和CER
    
    Args:
        hypo: 预测文本
        truth: 参考文本
        lang: 语言类型，zh或en
        
    Returns:
        raw_truth, raw_hypo, wer, cer, subs, dele, inse
    """
    raw_truth = truth.strip()
    raw_hypo = hypo.strip()
    
    # 预处理：移除标点符号
    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')
    
    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')
    
    # 对于中文：按字分割
    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
        # 同时计算字错误率（CER）和词错误率（WER）
        # 对于中文，WER实际上就是CER
        measures = compute_measures(truth, hypo)
        ref_list = truth.split(" ")
        wer = measures["wer"]
        cer = wer  # 对于中文，WER就是CER
        subs = measures["substitutions"] / len(ref_list) if len(ref_list) > 0 else 0
        dele = measures["deletions"] / len(ref_list) if len(ref_list) > 0 else 0
        inse = measures["insertions"] / len(ref_list) if len(ref_list) > 0 else 0
    elif lang == "en":
        # 对于英文：按词计算WER
        truth = truth.lower()
        hypo = hypo.lower()
        measures = compute_measures(truth, hypo)
        ref_list = truth.split(" ")
        wer = measures["wer"]
        # 英文的CER通常不常用，这里也计算一下
        # 将英文单词按字符展开计算
        truth_chars = " ".join(list(truth.replace(" ", "")))
        hypo_chars = " ".join(list(hypo.replace(" ", "")))
        char_measures = compute_measures(truth_chars, hypo_chars)
        ref_chars = truth_chars.split(" ")
        cer = char_measures["wer"]
        subs = measures["substitutions"] / len(ref_list) if len(ref_list) > 0 else 0
        dele = measures["deletions"] / len(ref_list) if len(ref_list) > 0 else 0
        inse = measures["insertions"] / len(ref_list) if len(ref_list) > 0 else 0
    else:
        raise NotImplementedError(f"Language {lang} not supported")
    
    return (raw_truth, raw_hypo, wer, cer, subs, dele, inse)


def calculate_statistics_from_text_file(
    input_file, 
    output_file=None, 
    lang="zh", 
    filter_high_wer=False, 
    wer_threshold=1.0,
    exclude_below=None
):
    """
    从文本文件直接计算WER/CER统计信息
    
    Args:
        input_file: 输入文本文件路径，每行格式: item_name predict_text gt_text
        output_file: 输出结果文件路径
        lang: 语言类型，zh或en
        filter_high_wer: 是否过滤WER过高的样本
        wer_threshold: WER阈值，超过此阈值的样本将被过滤
        exclude_below: WER低于此值的样本将被排除在过滤之外(可选)
    """
    all_results = []
    filtered_results = []  # 过滤后的结果
    stats = defaultdict(list)
    filtered_stats = defaultdict(list)
    
    print(f"正在处理文件: {input_file}")
    print(f"语言: {lang}")
    if filter_high_wer:
        print(f"启用WER过滤，阈值: {wer_threshold}")
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    
    for line in tqdm(lines, desc="计算WER/CER"):
        line = line.strip()
        if not line:
            continue
            
        # 解析每行，支持多种分隔符
        if '\t' in line:
            parts = line.split('\t')
        else:
            parts = line.split(' ', 2)  # 最多分割成3部分
            
        if len(parts) < 3:
            print(f"警告: 行格式不正确，跳过: {line}")
            continue
            
        item_name = parts[0]
        predict_text = parts[1]
        gt_text = parts[2]
        
        # 处理文本对
        raw_truth, raw_hypo, wer, cer, subs, dele, inse = process_one(predict_text, gt_text, lang)
        
        # 收集结果
        result = {
            "id": item_name,
            "wer": wer,
            "cer": cer,
            "distance": round(wer * len(raw_truth.replace(" ", "")) if lang == "zh" else wer * len(gt_text.split())),
            "substitutions": subs,
            "deletions": dele,
            "insertions": inse,
            "ground_truth": raw_truth,
            "predicted_text": raw_hypo,
            "ref_tokens": raw_truth.split() if lang == "en" else list(raw_truth.replace(" ", "")),
            "hyp_tokens": raw_hypo.split() if lang == "en" else list(raw_hypo.replace(" ", "")),
            "excluded": False  # 默认未排除
        }
        
        all_results.append(result)
        
        # 收集全部统计数据
        stats["wers"].append(wer)
        stats["cers"].append(cer)
        stats["subs"].append(subs)
        stats["dels"].append(dele)
        stats["inss"].append(inse)
        
        # 检查是否需要过滤
        should_exclude = False
        if filter_high_wer:
            if wer > wer_threshold:
                should_exclude = True
                result["excluded"] = True
                print(f"过滤高WER样本: {item_name}, WER={wer:.4f}")
        
        # 如果不需要过滤，添加到过滤后的结果
        if not should_exclude:
            filtered_results.append(result)
            filtered_stats["wers"].append(wer)
            filtered_stats["cers"].append(cer)
            filtered_stats["subs"].append(subs)
            filtered_stats["dels"].append(dele)
            filtered_stats["inss"].append(inse)
    
    # 计算总体统计（所有样本）
    total_samples = len(all_results)
    if total_samples == 0:
        print("错误: 没有有效的样本")
        return None
    
    # 计算全部样本的平均值
    overall_wer = np.mean(stats["wers"]) if stats["wers"] else 0
    overall_cer = np.mean(stats["cers"]) if stats["cers"] else 0
    avg_subs = np.mean(stats["subs"]) if stats["subs"] else 0
    avg_dels = np.mean(stats["dels"]) if stats["dels"] else 0
    avg_inss = np.mean(stats["inss"]) if stats["inss"] else 0
    
    # 计算过滤后样本的平均值
    filtered_samples = len(filtered_results)
    if filtered_samples > 0:
        filtered_wer = np.mean(filtered_stats["wers"]) if filtered_stats["wers"] else 0
        filtered_cer = np.mean(filtered_stats["cers"]) if filtered_stats["cers"] else 0
        filtered_subs = np.mean(filtered_stats["subs"]) if filtered_stats["subs"] else 0
        filtered_dels = np.mean(filtered_stats["dels"]) if filtered_stats["dels"] else 0
        filtered_inss = np.mean(filtered_stats["inss"]) if filtered_stats["inss"] else 0
    else:
        filtered_wer = 0
        filtered_cer = 0
        filtered_subs = 0
        filtered_dels = 0
        filtered_inss = 0
    
    # 计算其他统计信息
    min_wer = np.min(stats["wers"]) if stats["wers"] else 0
    max_wer = np.max(stats["wers"]) if stats["wers"] else 0
    
    # 计算总编辑距离和总参考token数
    total_distance = sum([r["distance"] for r in all_results])
    total_ref_tokens = sum([len(r["ref_tokens"]) for r in all_results])
    
    # 计算过滤后的总编辑距离和总参考token数
    filtered_distance = sum([r["distance"] for r in filtered_results])
    filtered_ref_tokens = sum([len(r["ref_tokens"]) for r in filtered_results])
    
    # 计算排除的样本数
    excluded_samples = total_samples - filtered_samples
    
    # 构建最终结果
    final_result = {
        "all_samples": {
            "overall_wer": overall_wer,
            "overall_cer": overall_cer,
            "average_wer": overall_wer,  # 保持向后兼容
            "min_wer": min_wer,
            "max_wer": max_wer,
            "total_samples": total_samples,
            "total_distance": total_distance,
            "total_ref_tokens": total_ref_tokens,
            "substitution_rate": avg_subs,
            "deletion_rate": avg_dels,
            "insertion_rate": avg_inss,
        },
        "filtered_samples": {
            "overall_wer": filtered_wer,
            "overall_cer": filtered_cer,
            "average_wer": filtered_wer,
            "total_samples": filtered_samples,
            "total_distance": filtered_distance,
            "total_ref_tokens": filtered_ref_tokens,
            "substitution_rate": filtered_subs,
            "deletion_rate": filtered_dels,
            "insertion_rate": filtered_inss,
        },
        "filter_settings": {
            "filter_high_wer": filter_high_wer,
            "wer_threshold": wer_threshold if filter_high_wer else None,
            "excluded_samples": excluded_samples,
        },
        "language": lang,
        "samples": all_results  # 包含所有样本，有excluded标记
    }
    
    # 输出到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("WER/CER 统计摘要")
    print("="*60)
    print(f"总样本数: {total_samples}")
    print(f"排除样本数: {excluded_samples}")
    print(f"有效样本数: {filtered_samples}")
    
    print("\n[全部样本统计]")
    print(f"  总体WER: {overall_wer:.4f}")
    print(f"  总体CER: {overall_cer:.4f}")
    print(f"  最小值: {min_wer:.4f}")
    print(f"  最大值: {max_wer:.4f}")
    print(f"  替换率: {avg_subs:.4f}")
    print(f"  删除率: {avg_dels:.4f}")
    print(f"  插入率: {avg_inss:.4f}")
    
    if filter_high_wer:
        print("\n[过滤后样本统计]")
        print(f"  总体WER: {filtered_wer:.4f}")
        print(f"  总体CER: {filtered_cer:.4f}")
        print(f"  替换率: {filtered_subs:.4f}")
        print(f"  删除率: {filtered_dels:.4f}")
        print(f"  插入率: {filtered_inss:.4f}")
        
        if excluded_samples > 0:
            print(f"\n[排除的样本详情]")
            excluded_wers = [r["wer"] for r in all_results if r["excluded"]]
            print(f"  排除样本的最小WER: {np.min(excluded_wers):.4f}")
            print(f"  排除样本的最大WER: {np.max(excluded_wers):.4f}")
            print(f"  排除样本的平均WER: {np.mean(excluded_wers):.4f}")
    
    return final_result


def calculate_cer_only_chinese(ref_text, hyp_text):
    """
    专门用于中文的CER计算（更准确）
    
    Args:
        ref_text: 参考文本
        hyp_text: 预测文本
        
    Returns:
        cer: 字符错误率
        details: 详细信息
    """
    # 移除所有非中文字符，只保留中文字符
    def clean_chinese(text):
        # 移除标点、空格、英文等
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        return text
    
    ref_clean = clean_chinese(ref_text)
    hyp_clean = clean_chinese(hyp_text)
    
    # 将字符串转换为字符列表
    ref_chars = list(ref_clean)
    hyp_chars = list(hyp_clean)
    
    # 计算编辑距离
    ref_str = " ".join(ref_chars)
    hyp_str = " ".join(hyp_chars)
    measures = compute_measures(ref_str, hyp_str)
    
    # 计算CER
    cer = measures["wer"]
    distance = measures["substitutions"] + measures["deletions"] + measures["insertions"]
    
    details = {
        "cer": cer,
        "distance": distance,
        "ref_char_count": len(ref_chars),
        "ref_chars": ref_chars,
        "hyp_chars": hyp_chars,
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"]
    }
    
    return cer, details


def process_text_file_detailed(
    input_file, 
    output_file=None, 
    lang="zh", 
    use_cer_only=False,
    filter_high_wer=False,
    wer_threshold=1.0
):
    """
    处理文本文件，生成详细的WER/CER报告
    """
    results = []
    filtered_results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(tqdm(lines, desc="处理文本")):
        line = line.strip()
        if not line:
            continue
            
        # 解析行
        if '\t' in line:
            parts = line.split('\t')
        else:
            parts = line.split(' ', 2)
            
        if len(parts) < 3:
            continue
            
        item_id = parts[0]
        pred_text = parts[1]
        gt_text = parts[2]
        
        if use_cer_only and lang == "zh":
            # 使用专门的中文CER计算
            cer, details = calculate_cer_only_chinese(gt_text, pred_text)
            wer = cer  # 对于中文，WER就是CER
        else:
            # 使用原始方法
            _, _, wer, cer, subs, dele, inse = process_one(pred_text, gt_text, lang)
            details = {
                "cer": cer,
                "substitutions": subs,
                "deletions": dele,
                "insertions": inse
            }
        
        # 创建结果对象
        result = {
            "id": i,
            "utterance_id": item_id,
            "wer": wer,
            "cer": cer,
            "distance": details.get("distance", 0),
            "ref_token_count": len(gt_text.replace(" ", "")) if lang == "zh" else len(gt_text.split()),
            "hyp_token_count": len(pred_text.replace(" ", "")) if lang == "zh" else len(pred_text.split()),
            "ground_truth": gt_text,
            "predicted_text": pred_text,
            "ref_tokens": list(gt_text.replace(" ", "")) if lang == "zh" else gt_text.split(),
            "hyp_tokens": list(pred_text.replace(" ", "")) if lang == "zh" else pred_text.split(),
            "details": details,
            "excluded": False
        }
        
        results.append(result)
        
        # 检查是否需要过滤
        if filter_high_wer and wer > wer_threshold:
            result["excluded"] = True
            print(f"过滤高WER样本: {item_id}, WER={wer:.4f}")
        else:
            filtered_results.append(result)
    
    # 计算统计信息
    if results:
        wers = [r["wer"] for r in results]
        cers = [r["cer"] for r in results]
        distances = [r["distance"] for r in results]
        ref_counts = [r["ref_token_count"] for r in results]
        
        total_samples = len(results)
        overall_wer = np.mean(wers)
        overall_cer = np.mean(cers)
        total_distance = sum(distances)
        total_ref_tokens = sum(ref_counts)
        
        # 计算过滤后的统计
        if filtered_results:
            filtered_wers = [r["wer"] for r in filtered_results]
            filtered_cers = [r["cer"] for r in filtered_results]
            filtered_distances = [r["distance"] for r in filtered_results]
            filtered_ref_counts = [r["ref_token_count"] for r in filtered_results]
            
            filtered_samples = len(filtered_results)
            filtered_wer = np.mean(filtered_wers)
            filtered_cer = np.mean(filtered_cers)
            filtered_distance = sum(filtered_distances)
            filtered_ref_tokens = sum(filtered_ref_counts)
        else:
            filtered_samples = 0
            filtered_wer = 0
            filtered_cer = 0
            filtered_distance = 0
            filtered_ref_tokens = 0
        
        excluded_samples = total_samples - filtered_samples
        
        final_result = {
            "all_samples": {
                "overall_wer": overall_wer,
                "overall_cer": overall_cer,
                "average_wer": overall_wer,
                "min_wer": np.min(wers),
                "max_wer": np.max(wers),
                "total_samples": total_samples,
                "total_distance": total_distance,
                "total_ref_tokens": total_ref_tokens,
            },
            "filtered_samples": {
                "overall_wer": filtered_wer,
                "overall_cer": filtered_cer,
                "average_wer": filtered_wer,
                "total_samples": filtered_samples,
                "total_distance": filtered_distance,
                "total_ref_tokens": filtered_ref_tokens,
            },
            "filter_settings": {
                "filter_high_wer": filter_high_wer,
                "wer_threshold": wer_threshold if filter_high_wer else None,
                "excluded_samples": excluded_samples,
            },
            "samples": results
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            print(f"详细结果已保存到: {output_file}")
        
        return final_result
    
    return None


def main():
    if len(sys.argv) < 3:
        print("用法:")
        print("  从文本文件计算WER/CER:")
        print("  python script.py input.txt output.json [lang] [--filter] [--threshold VALUE]")
        print("")
        print("参数说明:")
        print("  input.txt: 输入文本文件，每行格式: item_name predict_text gt_text")
        print("  output.json: 输出JSON文件")
        print("  lang: 语言类型，zh或en，默认为zh")
        print("  --filter: 启用高WER样本过滤")
        print("  --threshold: 设置WER阈值，默认1.0 (100%)")
        print("")
        print("示例:")
        print("  python script.py predictions.txt results.json zh")
        print("  python script.py predictions.txt results.json zh --filter")
        print("  python script.py predictions.txt results.json zh --filter --threshold 0.5")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    lang = "zh"
    filter_high_wer = False
    wer_threshold = 1.0
    
    # 解析可选参数
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--filter":
            filter_high_wer = True
            i += 1
        elif sys.argv[i] == "--threshold" and i+1 < len(sys.argv):
            wer_threshold = float(sys.argv[i+1])
            i += 2
        elif i == 3 and sys.argv[i] in ["zh", "en"]:
            lang = sys.argv[i]
            i += 1
        else:
            print(f"警告: 未知参数: {sys.argv[i]}")
            i += 1
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"语言: {lang}")
    if filter_high_wer:
        print(f"启用WER过滤，阈值: {wer_threshold}")
    
    # 使用方法1：快速计算
    result = calculate_statistics_from_text_file(
        input_file, 
        output_file, 
        lang, 
        filter_high_wer=filter_high_wer,
        wer_threshold=wer_threshold
    )
    
    # 或者使用方法2：详细计算
    # result = process_text_file_detailed(
    #     input_file, 
    #     output_file, 
    #     lang, 
    #     use_cer_only=True,
    #     filter_high_wer=filter_high_wer,
    #     wer_threshold=wer_threshold
    # )
    
    if result:
        print("\n计算完成！")
        print(f"结果已保存到: {output_file}")
    else:
        print("计算失败！")


if __name__ == "__main__":
    main()