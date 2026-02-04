import json
import re
import unicodedata
import csv
from typing import List, Dict, Tuple, Optional, Union
import editdistance as ed

# 导入标准化器（如果可用）
try:
    from cn_tn import TextNorm
    from whisper_normalizer.basic import BasicTextNormalizer
    from whisper_normalizer.english import EnglishTextNormalizer
    
    english_normalizer = EnglishTextNormalizer()
    chinese_normalizer = TextNorm(
        to_banjiao=False,
        to_upper=False,
        to_lower=False,
        remove_fillers=False,
        remove_erhua=False,
        check_chars=False,
        remove_space=False,
        cc_mode='',
    )
    basic_normalizer = BasicTextNormalizer()
    
    HAS_NORMALIZERS = True
except ImportError:
    print("警告: 未找到高级标准化器，使用基本标准化")
    HAS_NORMALIZERS = False


class EvaluationTokenizer:
    """评估用分词器（简化版本）"""
    
    SPACE = chr(32)
    
    def __init__(
        self,
        lowercase: bool = True,
        punctuation_removal: bool = True,
    ):
        self.lowercase = lowercase
        self.punctuation_removal = punctuation_removal
    
    @classmethod
    def remove_punctuation(cls, sent: str) -> str:
        """基于Unicode类别移除标点符号"""
        return cls.SPACE.join(
            t
            for t in sent.split(cls.SPACE)
            if not all(unicodedata.category(c)[0] == 'P' for c in t)
        )
    
    def tokenize(self, sent: str) -> str:
        """分词处理"""
        # 移除多余空格
        sent = re.sub(r'\s+', ' ', sent).strip()
        
        if self.punctuation_removal:
            sent = self.remove_punctuation(sent)
        
        if self.lowercase:
            sent = sent.lower()
        
        return sent


class BasicTextNormalizerSimple:
    """简单的文本标准化器（当高级标准化器不可用时使用）"""
    
    def __init__(self):
        pass
    
    def __call__(self, text: str) -> str:
        # 基本标准化：转换为小写，移除多余空格
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def normalize_text(text: str, language: str = 'en') -> str:
    """文本标准化"""
    
    if not HAS_NORMALIZERS:
        # 使用简单标准化器
        normalizer = BasicTextNormalizerSimple()
        return normalizer(text)
    
    # 使用高级标准化器
    if language == 'en':
        return english_normalizer(text)
    elif language == 'zh':
        return chinese_normalizer(text)
    else:
        return basic_normalizer(text)


def compute_wer_single(ref: str, hyp: str, 
                       language: str = 'en',
                       return_tokens: bool = False) -> Union[float, Tuple[float, List[str], List[str]]]:
    """
    计算单个句子的WER
    
    Args:
        ref: 参考文本（ground truth）
        hyp: 假设文本（模型输出）
        language: 语言（默认为英语）
        return_tokens: 是否返回分词结果
    
    Returns:
        WER值，如果return_tokens为True，同时返回分词结果
    """
    # 初始化分词器
    tokenizer = EvaluationTokenizer(
        lowercase=True,
        punctuation_removal=True,
    )
    
    # 文本标准化
    ref_normalized = normalize_text(ref, language)
    hyp_normalized = normalize_text(hyp, language)
    # print('ref_normalized ', ref_normalized)
    # print('ref ', ref)
    # assert 1==2
    # 分词
    ref_tokens = tokenizer.tokenize(ref_normalized).split()
    hyp_tokens = tokenizer.tokenize(hyp_normalized).split()
    
    # 计算编辑距离
    distance = ed.eval(ref_tokens, hyp_tokens)
    
    # 避免除零错误
    if len(ref_tokens) == 0:
        wer = 0.0 if len(hyp_tokens) == 0 else 1.0
    else:
        wer = distance / len(ref_tokens)
    
    if return_tokens:
        return wer, ref_tokens, hyp_tokens
    else:
        return wer


def parse_scp_line(line: str, delimiter: str = '\t') -> Tuple[str, str, str]:
    """
    解析SCP文件的一行
    
    格式: utterance_id <delimiter> ground_truth <delimiter> predicted_text
    
    Args:
        line: 输入行
        delimiter: 分隔符（默认为制表符）
    
    Returns:
        (utterance_id, ground_truth, predicted_text)
    """
    parts = line.strip().split(delimiter)
    
    if len(parts) < 3:
        raise ValueError(f"行格式不正确，期望至少3列，得到{len(parts)}列: {line}")
    
    utterance_id = parts[0].strip()
    
    # 合并剩余部分（以防文本中包含分隔符）
    if len(parts) == 3:
        predicted_text = parts[1].strip()
        ground_truth = parts[2].strip()
    else:
        # 如果有多列，第二列是ground_truth，其余合并为predicted_text
        # ground_truth = parts[1].strip()
        # predicted_text = delimiter.join(parts[2:]).strip()
        assert 1==2
    
    return utterance_id, ground_truth, predicted_text


def compute_wer_from_scp(scp_path: str, 
                         output_path: Optional[str] = None,
                         language: str = 'en',
                         delimiter: str = '\t',
                         exclude_high_wer: bool = False,
                         wer_threshold: float = 0.5,
                         verbose: bool = False) -> Dict:
    """
    从SCP文件计算WER
    
    Args:
        scp_path: SCP文件路径
        output_path: 输出结果文件路径（可选）
        language: 语言
        delimiter: 分隔符（默认为制表符）
        exclude_high_wer: 是否排除WER高于阈值的样本
        wer_threshold: WER阈值，高于此值的样本将被排除
        verbose: 是否打印详细信息
    
    Returns:
        包含WER统计信息的字典
    """
    
    # 读取SCP文件
    try:
        with open(scp_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {scp_path}")
        return {}
    
    total_lines = len(lines)
    print(f"读取 {total_lines} 行数据...")
    print(f"分隔符: '{delimiter}'")
    print(f"排除高WER样本: {exclude_high_wer}, 阈值: {wer_threshold}")
    
    # 初始化统计信息
    total_distance = 0
    total_ref_tokens = 0
    valid_samples_count = 0
    excluded_samples_count = 0
    
    # 存储每个样本的详细信息
    all_sample_details = []
    valid_sample_details = []
    
    # 解析每一行
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # 跳过空行
        if not line:
            continue
        
        try:
            # 解析行
            utterance_id, ground_truth, predicted_text = parse_scp_line(line, delimiter)
            
            # 计算WER
            wer, ref_tokens, hyp_tokens = compute_wer_single(
                ground_truth, predicted_text, language, return_tokens=True
            )
            
            # 计算编辑距离
            distance = ed.eval(ref_tokens, hyp_tokens)
            
            # 创建样本详情
            sample_detail = {
                'id': line_num - 1,
                'utterance_id': utterance_id,
                'wer': wer,
                'distance': distance,
                'ref_token_count': len(ref_tokens),
                'hyp_token_count': len(hyp_tokens),
                'ground_truth': ground_truth,
                'predicted_text': predicted_text,
                'ref_tokens': ref_tokens,
                'hyp_tokens': hyp_tokens,
            }
            
            all_sample_details.append(sample_detail)
            
            # 根据阈值决定是否添加到有效样本
            if not exclude_high_wer or wer <= wer_threshold:
                valid_sample_details.append(sample_detail)
                total_distance += distance
                total_ref_tokens += len(ref_tokens)
                valid_samples_count += 1
            else:
                excluded_samples_count += 1
                if verbose:
                    print(f"  排除行 {line_num} ({utterance_id}): WER = {wer:.4f} > {wer_threshold}")
            
            # 打印前几个样本的详细信息
            if verbose and line_num <= 5:
                print(f"\n样本 {line_num} ({utterance_id}):")
                print(f"  参考文本: {ground_truth}")
                print(f"  预测文本: {predicted_text}")
                print(f"  分词参考: {ref_tokens}")
                print(f"  分词预测: {hyp_tokens}")
                print(f"  WER: {wer:.4f} (距离: {distance}/{len(ref_tokens)})")
                
        except ValueError as e:
            print(f"警告: 行 {line_num} 解析失败: {e}")
            continue
        except Exception as e:
            print(f"警告: 处理行 {line_num} 时出错: {e}")
            continue
    
    # 计算统计信息
    total_samples = len(all_sample_details)
    
    # 计算总体WER（基于有效样本）
    if total_ref_tokens > 0:
        overall_wer = total_distance / total_ref_tokens
    else:
        overall_wer = 0.0
    
    # 计算样本级别的WER统计（基于有效样本）
    if valid_sample_details:
        wers = [detail['wer'] for detail in valid_sample_details]
        avg_wer = sum(wers) / len(wers)
        max_wer = max(wers)
        min_wer = min(wers)
    else:
        wers = []
        avg_wer = 0.0
        max_wer = 0.0
        min_wer = 0.0
    
    # 准备结果
    results = {
        'overall_wer': overall_wer,
        'average_wer': avg_wer,
        'min_wer': min_wer,
        'max_wer': max_wer,
        'total_samples': total_samples,
        'valid_samples': valid_samples_count,
        'excluded_samples': excluded_samples_count,
        'total_distance': total_distance,
        'total_ref_tokens': total_ref_tokens,
        'samples': valid_sample_details,  # 只包含有效样本
        'all_samples': all_sample_details,  # 包含所有样本
        'config': {
            'scp_file': scp_path,
            'language': language,
            'delimiter': delimiter,
            'exclude_high_wer': exclude_high_wer,
            'wer_threshold': wer_threshold,
        }
    }
    
    # 打印汇总信息
    print("\n" + "="*60)
    print("WER 计算结果汇总")
    print("="*60)
    print(f"总样本数: {total_samples}")
    if exclude_high_wer:
        print(f"有效样本数 (WER ≤ {wer_threshold}): {valid_samples_count}")
        print(f"排除样本数 (WER > {wer_threshold}): {excluded_samples_count}")
    print(f"总参考词数: {total_ref_tokens}")
    print(f"总编辑距离: {total_distance}")
    print(f"总体 WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    print(f"平均样本 WER: {avg_wer:.4f}")
    print(f"最小样本 WER: {min_wer:.4f}")
    print(f"最大样本 WER: {max_wer:.4f}")
    print("="*60)
    
    # 保存结果到文件
    if output_path:
        # 保存详细结果为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果已保存到: {output_path}")
        
        # 保存简化的CSV格式
        csv_path = output_path.replace('.json', '.csv')
        save_results_as_csv(results, csv_path)
        
        # 保存错误分析报告
        error_report_path = output_path.replace('.json', '_errors.txt')
        save_error_analysis(results, error_report_path)
    
    return results


def save_results_as_csv(results: Dict, csv_path: str):
    """将结果保存为CSV格式"""
    try:
        samples = results['samples']
        
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            # 写入标题行
            writer.writerow([
                'id', 'utterance_id', 'wer', 'distance', 
                'ref_token_count', 'hyp_token_count',
                'ground_truth', 'predicted_text'
            ])
            
            # 写入数据行
            for sample in samples:
                writer.writerow([
                    sample['id'],
                    sample['utterance_id'],
                    f"{sample['wer']:.4f}",
                    sample['distance'],
                    sample['ref_token_count'],
                    sample['hyp_token_count'],
                    sample['ground_truth'].replace('\n', ' '),
                    sample['predicted_text'].replace('\n', ' '),
                ])
        
        print(f"CSV格式结果已保存到: {csv_path}")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")


def save_error_analysis(results: Dict, report_path: str):
    """保存错误分析报告"""
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ASR WER 错误分析报告\n")
            f.write("="*60 + "\n\n")
            
            # 基本信息
            f.write("基本信息:\n")
            f.write(f"  总样本数: {results['total_samples']}\n")
            f.write(f"  有效样本数: {results['valid_samples']}\n")
            f.write(f"  排除样本数: {results['excluded_samples']}\n")
            f.write(f"  总体WER: {results['overall_wer']:.4f} ({results['overall_wer']*100:.2f}%)\n")
            f.write(f"  平均WER: {results['average_wer']:.4f}\n\n")
            
            # 按WER分组统计
            f.write("WER分布:\n")
            wer_groups = analyze_wer_distribution(results, return_dict=True)
            for group, count in wer_groups.items():
                percentage = (count / results['total_samples']) * 100
                f.write(f"  {group:30}: {count:4d} ({percentage:6.2f}%)\n")
            f.write("\n")
            
            # WER最高的样本
            f.write("WER最高的样本:\n")
            high_wer_samples = get_top_wer_samples(results, top_n=10)
            for i, sample in enumerate(high_wer_samples, 1):
                f.write(f"\n{i}. ID: {sample['id']}, Utterance: {sample['utterance_id']}\n")
                f.write(f"   WER: {sample['wer']:.4f} (距离: {sample['distance']}/{sample['ref_token_count']})\n")
                f.write(f"   参考: {sample['ground_truth']}\n")
                f.write(f"   预测: {sample['predicted_text']}\n")
        
        print(f"错误分析报告已保存到: {report_path}")
    except Exception as e:
        print(f"保存错误分析报告时出错: {e}")


def analyze_wer_distribution(results: Dict, return_dict: bool = False):
    """分析WER分布"""
    
    samples = results['all_samples']  # 使用所有样本进行分布分析
    
    # 按WER值分组
    wer_groups = {
        'perfect (WER=0)': 0,
        'excellent (0<WER<=0.1)': 0,
        'good (0.1<WER<=0.2)': 0,
        'fair (0.2<WER<=0.3)': 0,
        'poor (0.3<WER<=0.5)': 0,
        'bad (WER>0.5)': 0,
    }
    
    for sample in samples:
        wer = sample['wer']
        
        if wer == 0:
            wer_groups['perfect (WER=0)'] += 1
        elif wer <= 0.1:
            wer_groups['excellent (0<WER<=0.1)'] += 1
        elif wer <= 0.2:
            wer_groups['good (0.1<WER<=0.2)'] += 1
        elif wer <= 0.3:
            wer_groups['fair (0.2<WER<=0.3)'] += 1
        elif wer <= 0.5:
            wer_groups['poor (0.3<WER<=0.5)'] += 1
        else:
            wer_groups['bad (WER>0.5)'] += 1
    
    if not return_dict:
        print("\nWER分布分析:")
        print("-"*40)
        
        total_samples = len(samples)
        for group, count in wer_groups.items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"{group:30}: {count:4d} 样本 ({percentage:6.2f}%)")
    
    return wer_groups


def get_top_wer_samples(results: Dict, top_n: int = 10, include_excluded: bool = True):
    """获取WER最高的样本"""
    
    # 决定使用哪些样本
    if include_excluded:
        samples = results['all_samples']
    else:
        samples = results['samples']
    
    # 按WER排序（从高到低）
    sorted_samples = sorted(samples, key=lambda x: x['wer'], reverse=True)
    
    return sorted_samples[:top_n]


def print_top_wer_samples(results: Dict, top_n: int = 10, include_excluded: bool = True):
    """打印WER最高的样本"""
    
    top_samples = get_top_wer_samples(results, top_n, include_excluded)
    
    source_label = "所有样本" if include_excluded else "有效样本"
    print(f"\n{source_label}中WER最高的 {len(top_samples)} 个样本:")
    print("-"*80)
    
    for i, sample in enumerate(top_samples, 1):
        # 检查是否是被排除的样本
        excluded_note = ""
        if include_excluded and sample not in results['samples']:
            excluded_note = " [已排除]"
        
        print(f"\n{i}. ID: {sample['id']}, Utterance: {sample['utterance_id']}{excluded_note}")
        print(f"   WER: {sample['wer']:.4f} (距离: {sample['distance']}/{sample['ref_token_count']})")
        print(f"   参考: {sample['ground_truth']}")
        print(f"   预测: {sample['predicted_text']}")
    
    return top_samples


# 主函数
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='从SCP文件计算ASR任务的WER（词错误率）')
    parser.add_argument('scp_file', help='输入SCP文件路径')
    parser.add_argument('--output', '-o', default='scp_wer_results.json', 
                       help='输出结果文件路径（默认: scp_wer_results.json）')
    parser.add_argument('--language', '-l', default='en', 
                       choices=['en', 'zh', 'other'],
                       help='语言（默认: en）')
    parser.add_argument('--delimiter', '-d', default='\t',
                       help='分隔符（默认: 制表符，使用\\t表示）')
    parser.add_argument('--exclude-high-wer', action='store_true',
                       help='排除WER大于0.5的样本')
    parser.add_argument('--wer-threshold', type=float, default=0.5,
                       help='WER阈值（默认: 0.5）')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='打印详细信息')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='分析WER分布')
    parser.add_argument('--show-problems', '-p', action='store_true',
                       help='显示WER最高的样本')
    parser.add_argument('--max-problems', type=int, default=10,
                       help='显示的最大问题样本数（默认: 10）')
    parser.add_argument('--include-excluded', action='store_true',
                       help='在问题样本中包含被排除的样本')
    
    args = parser.parse_args()
    
    # 处理分隔符参数（支持特殊字符）
    delimiter = args.delimiter
    if delimiter == '\\t':
        delimiter = '\t'
    elif delimiter == '\\n':
        delimiter = '\n'
    elif delimiter == '\\r':
        delimiter = '\r'
    
    try:
        # 计算WER
        results = compute_wer_from_scp(
            scp_path=args.scp_file,
            output_path=args.output,
            language=args.language,
            delimiter=delimiter,
            exclude_high_wer=args.exclude_high_wer,
            wer_threshold=args.wer_threshold,
            verbose=args.verbose
        )
        
        if not results:
            print("错误: 未生成结果")
            return
        
        # 分析WER分布
        if args.analyze:
            analyze_wer_distribution(results)
        
        # 显示问题样本
        if args.show_problems:
            print_top_wer_samples(results, args.max_problems, args.include_excluded)
        
        print(f"\n✅ 处理完成!")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()
