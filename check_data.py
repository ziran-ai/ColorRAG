#!/usr/bin/env python3
"""
数据格式检查脚本
"""

import pandas as pd
import numpy as np
import sys
import os

def check_data_format(file_path):
    """检查数据文件格式"""
    print(f"检查数据文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return False
    
    try:
        # 读取数据
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print("错误: 不支持的文件格式，请使用.xlsx或.csv文件")
            return False
        
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 检查颜色列
        color_columns = []
        for i in range(1, 6):
            for c in ['R', 'G', 'B']:
                col_name = f'Color_{i}_{c}'
                if col_name in df.columns:
                    color_columns.append(col_name)
        
        if not color_columns:
            print("错误: 找不到颜色列")
            print("期望的列名格式: Color_1_R, Color_1_G, Color_1_B, ..., Color_5_R, Color_5_G, Color_5_B")
            return False
        
        print(f"找到颜色列: {color_columns}")
        
        # 检查颜色数据
        color_data = df[color_columns].values
        print(f"颜色数据形状: {color_data.shape}")
        print(f"颜色数据范围: {color_data.min()} - {color_data.max()}")
        
        if color_data.max() > 255 or color_data.min() < 0:
            print("警告: 颜色值超出0-255范围")
        
        # 检查文本列
        text_columns = ['description', 'Description', 'text', 'Text', 'content', 'Content']
        text_column = None
        for col in text_columns:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            print("错误: 找不到文本描述列")
            print("期望的列名: description, Description, text, Text, content, Content")
            return False
        
        print(f"找到文本列: {text_column}")
        
        # 检查文本数据
        text_data = df[text_column].fillna('')
        print(f"文本数据样本:")
        for i in range(min(3, len(text_data))):
            print(f"  {i+1}: {text_data.iloc[i][:100]}...")
        
        # 检查评分列
        score_columns = ['Targets', 'targets', 'score', 'Score', 'rating', 'Rating']
        score_column = None
        for col in score_columns:
            if col in df.columns:
                score_column = col
                break
        
        if score_column:
            print(f"找到评分列: {score_column}")
            scores = df[score_column].values
            print(f"评分范围: {scores.min()} - {scores.max()}")
            print(f"评分分布: {np.percentile(scores, [25, 50, 75])}")
        else:
            print("警告: 未找到评分列，训练时将使用随机评分")
        
        # 检查缺失值
        missing_color = df[color_columns].isnull().sum().sum()
        missing_text = df[text_column].isnull().sum()
        
        print(f"颜色数据缺失值: {missing_color}")
        print(f"文本数据缺失值: {missing_text}")
        
        if missing_color > 0:
            print("警告: 颜色数据存在缺失值")
        
        # 数据质量检查
        print("\n数据质量检查:")
        
        # 检查重复行
        duplicates = df.duplicated().sum()
        print(f"重复行数: {duplicates}")
        
        # 检查空文本
        empty_text = (df[text_column].fillna('').str.strip() == '').sum()
        print(f"空文本行数: {empty_text}")
        
        # 检查颜色值是否合理
        invalid_colors = ((color_data < 0) | (color_data > 255)).sum()
        print(f"无效颜色值数量: {invalid_colors}")
        
        print("\n数据格式检查完成！")
        return True
        
    except Exception as e:
        print(f"错误: 读取数据文件时出错 - {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "palettes_descriptions.xlsx"
    
    success = check_data_format(file_path)
    
    if success:
        print("\n✅ 数据格式检查通过！可以开始训练。")
        print("\n下一步:")
        print("1. 运行: python run_training.py")
        print("2. 或者直接运行: python src/train_simple.py")
    else:
        print("\n❌ 数据格式检查失败！请修正数据格式后重试。")
        sys.exit(1)

if __name__ == "__main__":
    main() 