#!/usr/bin/env python3
"""
合并文件夹中所有.txt文件的脚本
用法: python merge_txt_files.py <输入文件夹路径> <输出文件路径>
"""

import os
import sys
import argparse
from pathlib import Path

def merge_txt_files(input_dir, output_file, verbose=True):
    """
    合并文件夹中的所有.txt文件
    
    参数:
        input_dir: 包含.txt文件的文件夹路径
        output_file: 合并后的输出文件路径
        verbose: 是否显示详细信息
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    # 1. 检查输入文件夹是否存在
    if not input_path.exists() or not input_path.is_dir():
        print(f"错误: 输入文件夹不存在或不是目录 - {input_dir}")
        return False
    
    # 2. 获取所有.txt文件
    txt_files = list(input_path.glob("*.txt"))
    if not txt_files:
        print(f"警告: 在 {input_dir} 中未找到.txt文件")
        return False
    
    if verbose:
        print(f"找到 {len(txt_files)} 个.txt文件")
        print(f"输出文件: {output_file}")
        print("-" * 50)
    
    total_lines = 0
    processed_files = 0
    
    try:
        # 3. 打开输出文件
        with open(output_path, 'w', encoding='utf-8') as out_f:
            # 4. 按文件名排序后依次处理每个文件
            for txt_file in sorted(txt_files):
                if verbose:
                    print(f"处理: {txt_file.name}")
                
                file_lines = 0
                try:
                    with open(txt_file, 'r', encoding='utf-8') as in_f:
                        for line in in_f:
                            # 跳过空行
                            line = line.strip()
                            if line:
                                out_f.write(line + '\n')
                                file_lines += 1
                    
                    total_lines += file_lines
                    processed_files += 1
                    
                    if verbose:
                        print(f"  添加了 {file_lines} 行")
                
                except UnicodeDecodeError:
                    # 尝试其他编码
                    try:
                        with open(txt_file, 'r', encoding='gbk') as in_f:
                            for line in in_f:
                                line = line.strip()
                                if line:
                                    out_f.write(line + '\n')
                                    file_lines += 1
                        
                        total_lines += file_lines
                        processed_files += 1
                        
                        if verbose:
                            print(f"  使用GBK编码添加了 {file_lines} 行")
                    
                    except Exception as e:
                        print(f"  错误: 无法读取文件 {txt_file.name} - {e}")
                
                except Exception as e:
                    print(f"  错误: 处理文件 {txt_file.name} 时发生错误 - {e}")
        
        # 5. 打印统计信息
        if verbose:
            print("-" * 50)
            print(f"合并完成!")
            print(f"成功处理: {processed_files}/{len(txt_files)} 个文件")
            print(f"总行数: {total_lines}")
            print(f"输出文件: {output_path.absolute()}")
        
        return True
    
    except Exception as e:
        print(f"错误: 写入输出文件时发生错误 - {e}")
        return False

def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(
        description='合并文件夹中的所有.txt文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
        %(prog)s ./data ./merged_output.txt
        %(prog)s /home/user/documents /tmp/all_files.txt
                """
    )
    
    parser.add_argument('input_dir', help='包含.txt文件的文件夹路径')
    parser.add_argument('output_file', help='合并后的输出文件路径')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='安静模式，减少输出')
    
    args = parser.parse_args()
    
    # 运行合并函数
    success = merge_txt_files(
        input_dir=args.input_dir,
        output_file=args.output_file,
        verbose=not args.quiet
    )
    
    # 设置适当的退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
