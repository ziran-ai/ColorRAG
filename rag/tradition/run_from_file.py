#!/usr/bin/env python3
"""
从文件读取输入并运行RAG系统
支持批量处理多个用户需求
"""

import os
import sys
import json
import argparse
from datetime import datetime
from topic_rag_system import TopicRAGSystem

def read_inputs_from_file(file_path: str) -> list:
    """
    从文件读取用户输入
    
    支持的文件格式：
    1. 纯文本文件：每行一个用户需求
    2. JSON文件：包含用户需求的数组
    3. CSV文件：包含用户需求列的文件
    
    Args:
        file_path: 输入文件路径
        
    Returns:
        用户需求列表
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return []
    
    try:
        # 根据文件扩展名判断格式
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.json':
            # JSON格式
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'inputs' in data:
                    return data['inputs']
                else:
                    print("❌ JSON格式错误，应为数组或包含'inputs'键的对象")
                    return []
        
        elif ext == '.csv':
            # CSV格式
            import pandas as pd
            df = pd.read_csv(file_path)
            if 'user_input' in df.columns:
                return df['user_input'].tolist()
            elif 'input' in df.columns:
                return df['input'].tolist()
            elif 'text' in df.columns:
                return df['text'].tolist()
            else:
                print("❌ CSV文件应包含'user_input'、'input'或'text'列")
                return []
        
        else:
            # 纯文本格式：每行一个输入
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return [line.strip() for line in lines if line.strip()]
                
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return []

def save_results_to_file(results: list, output_file: str):
    """
    保存结果到文件
    
    Args:
        results: 结果列表
        output_file: 输出文件路径
    """
    try:
        # 创建输出目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存为JSON格式
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")

def run_rag_for_inputs(inputs: list, system: TopicRAGSystem, 
                      image_path: str = "test_image.jpg", top_k: int = 5) -> list:
    """
    为多个输入运行RAG系统
    
    Args:
        inputs: 用户输入列表
        system: RAG系统实例
        image_path: 图片路径
        top_k: 检索候选数量
        
    Returns:
        结果列表
    """
    results = []
    
    for i, user_input in enumerate(inputs, 1):
        print(f"\n{'='*60}")
        print(f"📝 处理第{i}个输入: {user_input}")
        print(f"{'='*60}")
        
        try:
            # 运行RAG系统
            result = system.run_full_pipeline(user_input, image_path, top_k)
            
            # 整理结果
            processed_result = {
                'input_id': i,
                'user_input': user_input,
                'generated_plan': result['new_plan'],
                'candidates': [
                    {
                        'description': candidate['description'],
                        'text_score': candidate['text_score'],
                        'color_score': candidate['color_score'],
                        'combined_score': candidate['combined_score']
                    }
                    for candidate in result['candidates']
                ],
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(processed_result)
            
            print(f"✅ 第{i}个输入处理完成")
            print(f"生成方案长度: {len(result['new_plan'])}")
            
        except Exception as e:
            print(f"❌ 处理第{i}个输入失败: {e}")
            results.append({
                'input_id': i,
                'user_input': user_input,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从文件读取输入并运行RAG系统')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径')
    parser.add_argument('--output', '-o', default='outputs/rag_results.json', help='输出文件路径')
    parser.add_argument('--image', default='test_image.jpg', help='图片路径')
    parser.add_argument('--top_k', type=int, default=5, help='检索候选数量')
    parser.add_argument('--api_key', default='sk-3c4ba59c8b094106995821395c7bc60e', help='DeepSeek API密钥')
    
    args = parser.parse_args()
    
    print("🚀 开始从文件读取输入并运行RAG系统")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"图片路径: {args.image}")
    
    # 读取输入
    inputs = read_inputs_from_file(args.input)
    if not inputs:
        print("❌ 没有读取到有效的输入")
        return
    
    print(f"✅ 读取到 {len(inputs)} 个输入")
    
    # 初始化RAG系统
    try:
        system = TopicRAGSystem(device='cpu', api_key=args.api_key)
        print("✅ RAG系统初始化成功")
    except Exception as e:
        print(f"❌ RAG系统初始化失败: {e}")
        return
    
    # 运行RAG系统
    results = run_rag_for_inputs(inputs, system, args.image, args.top_k)
    
    # 保存结果
    save_results_to_file(results, args.output)
    
    # 打印统计信息
    successful = len([r for r in results if 'error' not in r])
    failed = len([r for r in results if 'error' in r])
    
    print(f"\n📊 处理统计:")
    print(f"总输入数: {len(inputs)}")
    print(f"成功处理: {successful}")
    print(f"处理失败: {failed}")
    
    if successful > 0:
        avg_length = sum(len(r['generated_plan']) for r in results if 'generated_plan' in r) / successful
        print(f"平均生成长度: {avg_length:.0f} 字符")

if __name__ == "__main__":
    main() 