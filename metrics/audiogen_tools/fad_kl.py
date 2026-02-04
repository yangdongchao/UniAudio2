import glob
import os
import shutil
import tempfile
import torch
import argparse
from audioldm_eval import EvaluationHelper
#from audiobox_aesthetics.infer import initialize_predictor
from tqdm import tqdm
import torchaudio

def calculate_averages(data):
    # 初始化计数器
    total_ce = 0
    total_cu = 0
    total_pc = 0
    total_pq = 0
    count = len(data)

    # 计算总和
    for item in data:
        total_ce += item['CE']
        total_cu += item['CU']
        total_pc += item['PC']
        total_pq += item['PQ']

    # 计算平均值
    avg_ce = total_ce / count
    avg_cu = total_cu / count
    avg_pc = total_pc / count
    avg_pq = total_pq / count

    return {
        'CE': avg_ce,
        'CU': avg_cu,
        'PC': avg_pc,
        'PQ': avg_pq
    }

def create_matched_temp_dirs(generation_dir, target_dir, file_extensions=None):
    """
    创建临时目录，只包含两个文件夹中都存在的音频文件
    
    参数:
        generation_dir: 生成的音频文件夹路径
        target_dir: 目标音频文件夹路径  
        file_extensions: 音频文件扩展名列表，默认支持常见的音频格式
    
    返回:
        tuple: (temp_gen_dir, temp_target_dir, file_count, temp_dir)
        返回两个临时目录路径和匹配的文件数量以及临时目录路径
    """
    if file_extensions is None:
        # 支持常见的音频格式
        file_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='audio_eval_')
    temp_gen_dir = os.path.join(temp_dir, 'generation')
    temp_target_dir = os.path.join(temp_dir, 'target')
    os.makedirs(temp_gen_dir, exist_ok=True)
    os.makedirs(temp_target_dir, exist_ok=True)
    
    # 获取两个文件夹中的音频文件名（不含扩展名）
    gen_files = {}
    target_files = {}
    
    # 扫描生成文件夹
    for ext in file_extensions:
        for file_path in glob.glob(os.path.join(generation_dir, f'*{ext}')):
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            gen_files[name_without_ext] = file_path
    
    # 扫描目标文件夹
    for ext in file_extensions:
        for file_path in glob.glob(os.path.join(target_dir, f'*{ext}')):
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            target_files[name_without_ext] = file_path
    
    # 找出两个文件夹中都存在的文件
    common_filenames = set(gen_files.keys()) & set(target_files.keys())
    
    if not common_filenames:
        print("警告: 两个文件夹中没有共同的文件名")
        # 清理临时目录
        shutil.rmtree(temp_dir)
        return None, None, 0, None
    
    print(f"找到 {len(common_filenames)} 个共同的文件")
    
    # 复制文件到临时目录
    matched_count = 0
    for filename in common_filenames:
        # 复制生成文件
        gen_src = gen_files[filename]
        # 保持原扩展名
        gen_ext = os.path.splitext(gen_src)[1]
        gen_dst = os.path.join(temp_gen_dir, f"{filename}{gen_ext}")
        shutil.copy2(gen_src, gen_dst)
        
        # 复制目标文件
        target_src = target_files[filename]
        target_ext = os.path.splitext(target_src)[1]
        target_dst = os.path.join(temp_target_dir, f"{filename}{target_ext}")
        shutil.copy2(target_src, target_dst)
        
        matched_count += 1
    
    return temp_gen_dir, temp_target_dir, matched_count, temp_dir

def cleanup_temp_dirs(temp_dir):
    """清理临时目录"""
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"已清理临时目录: {temp_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='音频生成结果评估')
    parser.add_argument(
        '--generation_result_path', 
        type=str, 
        required=True,
        help='生成的音频文件目录路径'
    )
    parser.add_argument(
        '--target_audio_path', 
        type=str, 
        required=True,
        help='目标音频文件目录路径'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='使用的设备，默认cuda'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='cnn14',
        choices=['cnn14', 'mert'],
        help='评估使用的backbone模型，默认cnn14'
    )
    parser.add_argument(
        '--limit_num',
        type=int,
        default=None,
        help='限制评估的音频对数，默认None（全部评估）'
    )
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取目录路径
    generation_result_path = args.generation_result_path
    target_audio_path = args.target_audio_path
    
    # 检查目录是否存在
    if not os.path.exists(generation_result_path):
        print(f"错误: 生成音频目录不存在: {generation_result_path}")
        return
    
    if not os.path.exists(target_audio_path):
        print(f"错误: 目标音频目录不存在: {target_audio_path}")
        return
    
    print(f"生成音频目录: {generation_result_path}")
    print(f"目标音频目录: {target_audio_path}")
    
    # 创建匹配的临时目录
    print("正在创建匹配的临时目录...")
    temp_gen_dir, temp_target_dir, matched_count, temp_dir = create_matched_temp_dirs(
        generation_result_path, 
        target_audio_path
    )
    
    if matched_count == 0:
        print("错误: 没有找到匹配的文件对，无法进行评估")
        return
    
    print(f"成功匹配 {matched_count} 对文件")
    print(f"临时生成目录: {temp_gen_dir}")
    print(f"临时目标目录: {temp_target_dir}")
    
    try:
        # 初始化评估器
        evaluator = EvaluationHelper(16000, device, backbone=args.backbone)
        
        # 使用临时目录进行评估
        print("开始评估...")
        metrics = evaluator.main(
            temp_gen_dir,
            temp_target_dir,
            limit_num=args.limit_num
        )
        
        print("\n评估结果:")
        print(metrics)
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 确保清理临时目录
        print("\n清理临时文件...")
        cleanup_temp_dirs(temp_dir)

if __name__ == "__main__":
    main()