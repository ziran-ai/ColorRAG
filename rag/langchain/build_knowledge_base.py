#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建并保存知识库
基于Topic Model计算所有词库条目的向量表示
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import pickle
from datetime import datetime

# 添加项目路径
sys.path.append('/root/autodl-tmp/AETM')
from load_separate_models import load_separate_models

class KnowledgeBaseBuilder:
    """知识库构建器"""
    
    def __init__(self, model_dir='../../models', data_path='../../data/palettes_descriptions.xlsx', device='cpu'):
        """
        初始化知识库构建器
        
        Args:
            model_dir: 模型文件目录
            data_path: 词库数据路径
            device: 计算设备
        """
        self.device = device
        self.model_dir = model_dir
        self.data_path = data_path
        
        print("🚀 初始化知识库构建器...")
        
        # 1. 加载topic model
        print("📊 加载Topic Model...")
        self.model, self.vectorizer, self.vocab, self.theta = load_separate_models(model_dir)
        self.model.to(device)
        self.model.eval()
        print(f"   模型加载成功: {self.model.num_topics}个主题, {len(self.vocab)}个词汇")
        
        # 2. 加载词库数据
        print("📚 加载词库数据...")
        self.df_knowledge = pd.read_excel(data_path)
        print(f"   词库大小: {len(self.df_knowledge)} 条记录")
        
        print("✅ 知识库构建器初始化完成！")
    
    def _text_to_bow(self, text: str):
        """将文本转换为BOW向量"""
        try:
            bow_vector = self.vectorizer.transform([text])
            return bow_vector.toarray()[0]
        except Exception as e:
            print(f"   BOW转换失败: {e}")
            return np.zeros(len(self.vocab))
    
    def _topic_model_inference(self, bow_vector):
        """使用topic model进行推理"""
        try:
            with torch.no_grad():
                bow_tensor = torch.FloatTensor(bow_vector).unsqueeze(0).to(self.device)
                
                # 创建零颜色向量
                color_tensor = torch.zeros(1, self.model.color_dim).to(self.device)
                
                # 编码
                mu_color, logvar_color = self.model.encode_color(color_tensor)
                mu_text, logvar_text = self.model.encode_text(bow_tensor)
                
                # 高斯乘积
                mu, logvar = self.model.product_of_gaussians(mu_color, logvar_color, mu_text, logvar_text)
                
                # 重参数化
                delta = self.model.reparameterize(mu, logvar)
                theta = self.model.get_theta(delta)
                
                # 解码
                recon_color, recon_text = self.model.decode(theta)
                
                return recon_text.cpu().numpy(), recon_color.cpu().numpy()
                
        except Exception as e:
            print(f"   Topic model推理失败: {e}")
            return np.zeros((1, len(self.vocab))), np.zeros((1, 15))
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("🔍 开始构建知识库...")
        
        # 提取描述文本
        descriptions = self.df_knowledge['description'].fillna('').tolist()
        names = self.df_knowledge['names'].fillna('').astype(str).tolist()
        
        # 为每个描述计算topic向量和颜色向量
        knowledge_text_vectors = []
        knowledge_color_vectors = []
        knowledge_colors_rgb = []
        
        print(f"   正在为 {len(descriptions)} 个词库条目计算Topic Model向量...")
        
        for i, desc in enumerate(descriptions):
            try:
                # 将描述转换为BOW
                desc_bow = self._text_to_bow(desc)
                
                # 通过topic model推理
                with torch.no_grad():
                    recon_text_prob, recon_color_prob = self._topic_model_inference(desc_bow)
                    
                    # 直接使用重构的文本概率作为文本表示
                    knowledge_text_vectors.append(recon_text_prob.flatten())
                    knowledge_color_vectors.append(recon_color_prob.flatten())
                
                # 提取RGB颜色
                colors_rgb = []
                for j in range(1, 6):
                    r = self.df_knowledge.iloc[i][f'Color_{j}_R']
                    g = self.df_knowledge.iloc[i][f'Color_{j}_G'] 
                    b = self.df_knowledge.iloc[i][f'Color_{j}_B']
                    colors_rgb.append([float(r), float(g), float(b)])
                knowledge_colors_rgb.append(colors_rgb)
                
                if (i + 1) % 500 == 0:
                    print(f"   已处理 {i + 1}/{len(descriptions)} 条记录 ({(i+1)/len(descriptions)*100:.1f}%)")
                    
            except Exception as e:
                print(f"   警告：第{i}条记录处理失败: {e}")
                # 使用零向量作为占位符
                knowledge_text_vectors.append(np.zeros(len(self.vocab)))
                knowledge_color_vectors.append(np.zeros(15))
                knowledge_colors_rgb.append([[0, 0, 0]] * 5)
        
        # 转换为numpy数组
        knowledge_text_vectors = np.array(knowledge_text_vectors)
        knowledge_color_vectors = np.array(knowledge_color_vectors)
        
        print(f"   知识库构建完成:")
        print(f"   - 文本向量: {knowledge_text_vectors.shape}")
        print(f"   - 颜色向量: {knowledge_color_vectors.shape}")
        print(f"   - RGB颜色: {len(knowledge_colors_rgb)} 条记录")
        
        # 构建知识库字典
        knowledge_base = {
            'metadata': {
                'total_entries': len(descriptions),
                'text_vector_shape': knowledge_text_vectors.shape,
                'color_vector_shape': knowledge_color_vectors.shape,
                'vocab_size': len(self.vocab),
                'num_topics': self.model.num_topics,
                'color_dim': self.model.color_dim,
                'build_time': datetime.now().isoformat(),
                'model_dir': self.model_dir,
                'data_path': self.data_path
            },
            'data': {
                'descriptions': descriptions,
                'names': names,
                'knowledge_text_vectors': knowledge_text_vectors,
                'knowledge_color_vectors': knowledge_color_vectors,
                'knowledge_colors_rgb': knowledge_colors_rgb
            }
        }
        
        return knowledge_base
    
    def save_knowledge_base(self, knowledge_base, save_path='knowledge_base.pkl'):
        """保存知识库到文件"""
        print(f"💾 保存知识库到: {save_path}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # 保存为pickle文件
        with open(save_path, 'wb') as f:
            pickle.dump(knowledge_base, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # 获取文件大小
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        print(f"✅ 知识库保存成功！文件大小: {file_size:.1f} MB")
        
        # 保存元数据为JSON（便于查看）
        metadata_path = save_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base['metadata'], f, ensure_ascii=False, indent=2)
        print(f"📋 元数据保存到: {metadata_path}")
    
    def build_and_save(self, save_path='knowledge_base.pkl'):
        """构建并保存知识库"""
        print("🚀 开始构建并保存知识库...")
        
        # 构建知识库
        knowledge_base = self.build_knowledge_base()
        
        # 保存知识库
        self.save_knowledge_base(knowledge_base, save_path)
        
        print("🎉 知识库构建和保存完成！")
        return knowledge_base

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='构建并保存知识库')
    parser.add_argument('--model_dir', type=str, default='../../models', help='模型目录')
    parser.add_argument('--data_path', type=str, default='../../data/palettes_descriptions.xlsx', help='数据文件路径')
    parser.add_argument('--output', type=str, default='knowledge_base.pkl', help='输出文件路径')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    
    args = parser.parse_args()
    
    print("🚀 开始构建知识库")
    print(f"模型目录: {args.model_dir}")
    print(f"数据路径: {args.data_path}")
    print(f"输出路径: {args.output}")
    print(f"计算设备: {args.device}")
    
    # 检查输入文件
    if not os.path.exists(args.data_path):
        print(f"❌ 数据文件不存在: {args.data_path}")
        return
    
    # 构建知识库
    try:
        builder = KnowledgeBaseBuilder(
            model_dir=args.model_dir,
            data_path=args.data_path,
            device=args.device
        )
        
        knowledge_base = builder.build_and_save(args.output)
        
        print(f"\n📊 构建统计:")
        print(f"总条目数: {knowledge_base['metadata']['total_entries']}")
        print(f"文本向量维度: {knowledge_base['metadata']['text_vector_shape']}")
        print(f"颜色向量维度: {knowledge_base['metadata']['color_vector_shape']}")
        print(f"词汇表大小: {knowledge_base['metadata']['vocab_size']}")
        print(f"主题数量: {knowledge_base['metadata']['num_topics']}")
        
    except Exception as e:
        print(f"❌ 知识库构建失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
