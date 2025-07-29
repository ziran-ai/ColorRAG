#!/usr/bin/env python3
"""
Topic-RAG系统：基于多模态主题模型的检索增强生成系统

正确的RAG流程：
1. 模块一：图片理解 + 文本融合 + Topic Model推理 + 向量化
2. 模块二：检索相关方案 + 增强提示词 + 生成新方案
"""

import torch
import numpy as np
import pandas as pd
import joblib
import json
from PIL import Image
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import os
import re
import sys

# 添加路径
sys.path.append('..')
sys.path.append('../utils')

from utils.ali_qwen_vl import upload_image_to_imgbb, ali_qwen_vl_image_caption
from utils.doubao_vl import upload_image_to_imgbb, doubao_vl_image_caption
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 导入您的模型
import sys
sys.path.append('..')
from src.topic_model import MultiOmicsETM
from load_separate_models import load_separate_models

IMG_BB_API_KEY = "4961859f178a605de87876a6a75b3a38"
ALI_API_KEY = "sk-8638f33779a8435eb3afe874a9d881d1"
ALI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DOUBAO_API_KEY = "fc7a6e47-91f5-4ced-9498-75383418e1a5"
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

def _is_url(path):
    return re.match(r'^https?://', path) is not None

class DeepSeekAPI:
    """DeepSeek API客户端"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_text(self, prompt: str, model: str = "deepseek-chat") -> str:
        """生成文本"""
        try:
            url = f"{self.base_url}/chat/completions"
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
                "temperature": 0.7
            }
            
            print(f"正在调用DeepSeek文本生成API，模型: {model}")
            response = requests.post(url, headers=self.headers, json=data, timeout=60)
            
            if response.status_code != 200:
                print(f"API调用失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                return ""
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            return ""
    
    def analyze_image(self, image_path: str, prompt: str, model: str = "deepseek-vision") -> str:
        """分析图片"""
        try:
            # 检查图片文件是否存在
            if not os.path.exists(image_path):
                print(f"图片文件不存在: {image_path}")
                return "图片文件不存在"
            
            # 读取并编码图片
            with open(image_path, "rb") as f:
                import base64
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            url = f"{self.base_url}/chat/completions"
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            print(f"正在调用DeepSeek图片分析API，模型: {model}")
            response = requests.post(url, headers=self.headers, json=data, timeout=60)
            
            if response.status_code != 200:
                print(f"图片分析API调用失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                return "图片分析API调用失败"
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"DeepSeek图片分析失败: {e}")
            return "图片分析失败"

class TopicRAGSystem:
    """Topic-RAG系统主类"""
    
    def __init__(self, model_dir='../models', device='cpu', api_key=None):
        """
        初始化Topic-RAG系统
        
        Args:
            model_dir: 模型文件目录
            device: 计算设备 ('cpu' 或 'cuda')
            api_key: DeepSeek API密钥
        """
        self.device = device
        self.model_dir = model_dir
        
        # 加载模型和必要组件
        print("正在加载模型和组件...")
        self.model, self.vectorizer, self.vocab, self.theta = load_separate_models(model_dir)
        self.model.to(device)
        self.model.eval()
        
        # 加载原始数据
        self.df_plans = pd.read_excel('../data/palettes_descriptions.xlsx')
        
        # 构建检索数据库
        self._build_retrieval_database()
        
        # 初始化DeepSeek API
        self._init_deepseek(api_key)
        
        print("Topic-RAG系统初始化完成！")
    
    def _init_deepseek(self, api_key: str):
        """初始化DeepSeek API"""
        try:
            if api_key:
                self.deepseek = DeepSeekAPI(api_key)
                self.llm_available = True
                print("✅ DeepSeek API初始化成功")
            else:
                print("警告：未提供DeepSeek API密钥，LLM功能将不可用")
                self.llm_available = False
        except Exception as e:
            print(f"DeepSeek API初始化失败: {e}")
            self.llm_available = False
    
    def _build_retrieval_database(self):
        """构建检索数据库"""
        print("正在构建检索数据库...")
        
        # 使用训练好的theta矩阵作为主题表示（文本描述的词库）
        self.theta_vectors = self.theta.cpu().numpy()
        self.theta_vectors_normalized = normalize(self.theta_vectors, norm='l2', axis=1)
        
        # 构建颜色向量（从原始数据中提取）
        color_cols = [f'Color_{i}_{c}' for i in range(1, 6) for c in ['R', 'G', 'B']]
        self.color_vectors = self.df_plans[color_cols].values
        self.color_vectors_normalized = normalize(self.color_vectors, norm='l2', axis=1)
        
        # 存储原始文本描述（词库）
        self.text_descriptions = self.df_plans['description'].values
        
        print(f"检索数据库构建完成：{len(self.theta_vectors)}个方案")
        print(f"词库大小：{len(self.text_descriptions)}个文本描述")
    
    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        将文本转换为词袋向量
        
        Args:
            text: 输入文本
            
        Returns:
            BoW向量
        """
        # 使用训练好的TF-IDF向量化器
        bow_vector = self.vectorizer.transform([text]).toarray()
        return bow_vector
    
    def _image_understanding(self, image_path):
        """图片理解：本地图片自动上传imgbb并调用豆包API，返回图片描述，且把imgBB URL加进文本中"""
        if _is_url(image_path):
            image_url = image_path
        else:
            image_url = upload_image_to_imgbb(image_path, IMG_BB_API_KEY)
        desc = doubao_vl_image_caption(image_url, DOUBAO_API_KEY, DOUBAO_BASE_URL)
        return f"[imgBB URL]: {image_url}\n{desc}"
    
    def _text_fusion(self, user_text: str, image_analysis: str) -> str:
        """
        融合用户文本和图片分析结果
        
        Args:
            user_text: 用户文本需求
            image_analysis: 图片分析结果
            
        Returns:
            融合后的文本
        """
        if not self.llm_available:
            # 如果没有LLM，简单拼接
            return f"用户需求：{user_text}\n图片分析：{image_analysis}"
        
        try:
            # 构建融合提示词
            prompt = f"""请根据以下信息生成一个详细的设计方案描述：

用户需求：{user_text}
图片分析：{image_analysis}

请融合用户需求和图片分析，生成一段详细的设计方案描述，包括：
1. 整体设计风格（结合用户需求和图片特点）
2. 色彩搭配建议（基于图片色彩和用户偏好）
3. 设计元素特点（融合图片元素和用户需求）
4. 适用场景（明确具体的使用环境）
5. 情感氛围（描述设计营造的特定情感）

要求：语言专业，描述具体，突出色彩搭配，体现个性化。"""
            
            fused_text = self.deepseek.generate_text(prompt)
            
            if fused_text:
                print(f"文本融合结果: {fused_text[:100]}...")
                return fused_text
            else:
                print("文本融合返回空结果，使用原始文本")
                return user_text
                
        except Exception as e:
            print(f"文本融合失败: {e}，使用原始文本")
            return user_text
    
    def _topic_model_inference(self, text_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用Topic Model进行推理
        
        Args:
            text_vector: 文本BoW向量
            
        Returns:
            (重构的文本概率, 重构的颜色概率)
        """
        self.model.eval()
        with torch.no_grad():
            # 转换为tensor
            text_tensor = torch.FloatTensor(text_vector).to(self.device)
            
            # 使用文本编码器
            mu_text, logvar_text = self.model.encode_text(text_tensor)
            
            # 由于没有颜色输入，设置默认的颜色分布（高不确定性）
            mu_color = torch.zeros_like(mu_text)
            logvar_color = torch.ones_like(logvar_text) * 10
            
            # 高斯乘积
            mu, logvar = self.model.product_of_gaussians(mu_color, logvar_color, mu_text, logvar_text)
            
            # 获取主题比例
            theta = self.model.get_theta(mu)
            
            # 解码得到重构结果
            recon_color, recon_text = self.model.decode(theta)
            
            return recon_text.cpu().numpy(), recon_color.cpu().numpy()
    
    def module_one_process(self, user_text: str, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        模块一：图片理解 + 文本融合 + Topic Model推理 + 向量化
        
        Args:
            user_text: 用户文本需求
            image_path: 图片路径
            
        Returns:
            (重构的文本概率, 重构的颜色概率)
        """
        print("=== 模块一：图片理解 + 文本融合 + Topic Model推理 ===")
        
        # 步骤1.1: 图片理解
        print("步骤1.1: 正在使用DeepSeek Vision理解图片...")
        image_analysis = self._image_understanding(image_path)
        
        # 步骤1.2: 文本融合
        print("步骤1.2: 正在融合用户需求和图片分析...")
        fused_text = self._text_fusion(user_text, image_analysis)
        
        # 步骤1.3: 文本向量化
        print("步骤1.3: 正在将融合文本向量化...")
        bow_vector = self._text_to_bow(fused_text)
        
        # 步骤1.4: Topic Model推理
        print("步骤1.4: 正在通过Topic Model进行跨模态推理...")
        recon_text_prob, recon_color_prob = self._topic_model_inference(bow_vector)
        
        print(f"推理完成：文本概率向量形状 {recon_text_prob.shape}, 颜色概率向量形状 {recon_color_prob.shape}")
        
        return recon_text_prob, recon_color_prob
    
    def _retrieve_candidates(self, recon_text_prob: np.ndarray, recon_color_prob: np.ndarray, 
                           top_k: int = 10) -> List[Dict]:
        """
        检索相关候选方案 - 实现双重相似度计算
        
        Args:
            recon_text_prob: Topic Model输出的文本概率矩阵
            recon_color_prob: Topic Model输出的颜色概率矩阵
            top_k: 检索数量
            
        Returns:
            候选方案列表
        """
        print(f"正在检索Top-{top_k}个相关方案...")
        
        # 步骤1: 将文本概率向量转换为主题向量
        with torch.no_grad():
            text_tensor = torch.FloatTensor(recon_text_prob).to(self.device)
            mu_text, logvar_text = self.model.encode_text(text_tensor)
            theta_query = self.model.get_theta(mu_text).cpu().numpy()
        
        # 步骤2: 计算文本相似度（Topic Model输出的文本矩阵与词库的相似度）
        print("步骤2.1: 计算文本相似度...")
        theta_query_norm = normalize(theta_query, norm='l2', axis=1)
        text_similarities = cosine_similarity(theta_query_norm, self.theta_vectors_normalized).flatten()
        
        # 步骤3: 根据文本相似度选出Top-K个候选方案
        print("步骤2.2: 根据文本相似度选出候选方案...")
        top_text_indices = np.argsort(text_similarities)[-top_k*2:][::-1]  # 选出2倍数量用于后续筛选
        
        print(f"文本相似度范围: {text_similarities[top_text_indices].min():.3f} - {text_similarities[top_text_indices].max():.3f}")
        
        # 步骤4: 计算颜色相似度（Topic Model输出的颜色矩阵与候选方案的颜色相似度）
        print("步骤2.3: 计算颜色相似度...")
        recon_color_norm = normalize(recon_color_prob, norm='l2', axis=1)
        
        candidates = []
        for idx in top_text_indices:
            # 获取该候选方案的颜色向量
            candidate_color_vec = self.color_vectors_normalized[idx].reshape(1, -1)
            
            # 计算颜色相似度
            color_similarity = cosine_similarity(recon_color_norm, candidate_color_vec).item()
            
            # 综合得分（文本相似度权重0.6，颜色相似度权重0.4）
            combined_score = 0.6 * text_similarities[idx] + 0.4 * color_similarity
            
            candidates.append({
                'index': idx,
                'text_score': text_similarities[idx],
                'color_score': color_similarity,
                'combined_score': combined_score,
                'description': self.df_plans.iloc[idx]['description'],
                'colors': self.df_plans.iloc[idx][[f'Color_{i}_{c}' for i in range(1, 6) for c in ['R', 'G', 'B']]].values.reshape(-1, 3)
            })
        
        # 步骤5: 根据综合得分排序，选出最终的Top-K个方案
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        final_candidates = candidates[:top_k]
        
        print(f"最终检索结果：综合得分范围 {final_candidates[0]['combined_score']:.3f} - {final_candidates[-1]['combined_score']:.3f}")
        
        # 打印详细的相似度信息
        print("\n📊 检索详情:")
        for i, candidate in enumerate(final_candidates[:3], 1):
            print(f"方案{i}: 文本相似度={candidate['text_score']:.3f}, 颜色相似度={candidate['color_score']:.3f}, 综合得分={candidate['combined_score']:.3f}")
        
        return final_candidates
    
    def _generate_rag_prompt(self, user_text: str, image_analysis: str, 
                           top_candidates: List[Dict]) -> str:
        """
        构建RAG增强提示词
        
        Args:
            user_text: 用户原始需求
            image_analysis: 图片分析结果
            top_candidates: 检索到的候选方案
            
        Returns:
            增强后的提示词
        """
        # 构建参考方案文本
        reference_text = "参考设计方案：\n\n"
        for i, candidate in enumerate(top_candidates[:3], 1):
            colors = candidate['colors']
            color_desc = ""
            for j, color in enumerate(colors, 1):
                color_desc += f"颜色{j}: RGB({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})\n"
            
            reference_text += f"方案{i}：\n描述：{candidate['description']}\n配色：\n{color_desc}\n"
        
        # 构建增强提示词
        prompt = f"""你是一位专业的设计美学专家。请根据以下信息生成一个全新的、个性化的设计方案：

用户需求：{user_text}

{reference_text}

请基于以上参考方案，结合用户需求，生成一个全新的设计方案。要求：

1. 设计风格：明确说明整体设计风格特点，要与参考方案有所区别
2. 色彩搭配：提供5种颜色的RGB值，并说明色彩搭配原理和与参考方案的区别
3. 设计元素：描述主要设计元素和布局特点，体现创新性
4. 适用场景：说明适用的具体场景，可以扩展参考方案的应用范围
5. 情感氛围：描述设计营造的情感氛围，突出个性化特点

重要：请确保生成的设计方案是全新的，与参考方案有明显区别，同时满足用户需求。
请用中文回答，语言专业优美，突出创新性和个性化特点。"""
        
        return prompt
    
    def module_two_process(self, recon_text_prob: np.ndarray, recon_color_prob: np.ndarray, 
                          user_text: str, image_analysis: str, top_k: int = 10) -> Dict[str, Any]:
        """
        模块二：检索增强生成
        
        Args:
            recon_text_prob: 重构的文本概率
            recon_color_prob: 重构的颜色概率
            user_text: 用户原始需求
            image_analysis: 图片分析结果
            top_k: 检索候选数量
            
        Returns:
            生成的新方案
        """
        print("=== 模块二：检索增强生成 ===")
        
        # 步骤2.1: 检索相关候选方案
        print("步骤2.1: 正在检索相关候选方案...")
        candidates = self._retrieve_candidates(recon_text_prob, recon_color_prob, top_k)
        
        # 步骤2.2: 构建RAG增强提示词
        print("步骤2.2: 正在构建RAG增强提示词...")
        rag_prompt = self._generate_rag_prompt(user_text, image_analysis, candidates)
        
        # 步骤2.3: 生成新方案
        print("步骤2.3: 正在生成新方案...")
        if self.llm_available:
            new_plan = self.deepseek.generate_text(rag_prompt)
        else:
            new_plan = "LLM不可用，无法生成新方案"
        
        return {
            'new_plan': new_plan,
            'candidates': candidates[:3],  # 返回前3个候选
            'rag_prompt': rag_prompt
        }
    
    def run_full_pipeline(self, user_text: str, image_path: str, top_k: int = 10) -> Dict[str, Any]:
        """
        运行完整的Topic-RAG流程
        
        Args:
            user_text: 用户文本需求
            image_path: 图片路径
            top_k: 检索候选数量
            
        Returns:
            完整的处理结果
        """
        print("🚀 启动Topic-RAG系统...")
        print(f"用户需求: {user_text}")
        print(f"图片路径: {image_path}")
        
        # 模块一：图片理解 + 文本融合 + Topic Model推理
        recon_text_prob, recon_color_prob = self.module_one_process(user_text, image_path)
        
        # 获取图片分析结果（用于模块二）
        image_analysis = self._image_understanding(image_path)
        
        # 模块二：检索增强生成
        result = self.module_two_process(recon_text_prob, recon_color_prob, user_text, image_analysis, top_k)
        
        print("\n" + "="*50)
        print("✨ 生成的新设计方案 ✨")
        print("="*50)
        print(result['new_plan'])
        print("="*50)
        
        return result

def main():
    """主函数：演示系统使用"""
    
    # 使用您提供的API密钥
    api_key = "sk-3c4ba59c8b094106995821395c7bc60e"
    
    # 初始化系统
    system = TopicRAGSystem(device='cpu', api_key=api_key)
    
    # 示例使用
    user_text = "我想要一个现代简约风格的配色方案，适合办公环境"
    image_path = "test_image.jpg"  # 替换为实际图片路径
    
    # 运行完整流程
    result = system.run_full_pipeline(user_text, image_path)
    
    # 打印检索详情
    print("\n📊 检索到的参考方案:")
    for i, candidate in enumerate(result['candidates'], 1):
        print(f"{i}. 综合得分: {candidate['combined_score']:.3f}")
        print(f"   描述: {candidate['description'][:100]}...")

if __name__ == "__main__":
    main() 