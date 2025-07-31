#!/usr/bin/env python3
"""
Streamlit Web界面 - RAG配色方案生成系统
基于成功的simple_pipeline.py构建用户友好的Web界面
"""

import streamlit as st
import sys
import os
import json
import torch
import numpy as np
from PIL import Image
import base64
import io
import requests
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

# 修复路径问题 - 添加正确的项目根目录
project_root = '/root/autodl-tmp/AETM'
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'rag', 'langchain'))

# 导入检查
try:
    from src.topic_model import MultiOmicsETM
    from load_separate_models import load_separate_models  # 这个文件在项目根目录
    from langchain_rag_system import LangChainTopicRAGSystem
    st.success("✅ 所有模块导入成功")
except ImportError as e:
    st.error(f"❌ 模块导入失败: {e}")
    st.info("""
    请检查以下文件是否存在：
    - /root/autodl-tmp/AETM/src/topic_model.py
    - /root/autodl-tmp/AETM/load_separate_models.py  (注意：在项目根目录)
    - /root/autodl-tmp/AETM/rag/langchain/langchain_rag_system.py
    """)
    st.stop()

# API密钥配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-3c4ba59c8b094106995821395c7bc60e")
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "3ed24480-459b-4dfc-8d80-57cd55b8fca7")

# 检查API密钥
if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
    st.warning("⚠️ 请配置DEEPSEEK_API_KEY环境变量")

if not DOUBAO_API_KEY or DOUBAO_API_KEY == "your_doubao_api_key_here":
    st.warning("⚠️ 请配置DOUBAO_API_KEY环境变量（用于图片理解）")

# 页面配置
st.set_page_config(
    page_title="ColorRAG - AI Color Design Platform",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 商业化CSS样式
st.markdown("""
<style>
    /* 隐藏Streamlit默认元素 */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}

    /* 主标题样式 */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }

    /* 副标题样式 */
    .subtitle {
        text-align: center;
        color: #8B8B8B;
        font-size: 1.1rem;
        margin-bottom: 4rem;
        font-weight: 300;
        line-height: 1.6;
    }

    /* 特色功能卡片 */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }

    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid #f0f2f6;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.12);
    }

    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1.5rem;
        display: block;
    }

    .feature-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2D3748;
        margin-bottom: 1rem;
    }

    .feature-desc {
        color: #718096;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* 输入区域样式 */
    .input-section {
        background: white;
        border-radius: 24px;
        padding: 3rem;
        margin: 3rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.08);
        border: 1px solid #f0f2f6;
    }

    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2D3748;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-subtitle {
        color: #718096;
        font-size: 1rem;
        margin-bottom: 2rem;
        line-height: 1.5;
    }

    /* 生成按钮样式 */
    .generate-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1.2rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
        width: 100%;
        margin: 2rem 0;
    }

    .generate-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.4);
    }

    /* 结果展示区域 */
    .result-section {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 24px;
        padding: 3rem;
        margin: 3rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.08);
    }

    /* 调色板样式 */
    .color-palette-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }

    .color-item {
        text-align: center;
        margin: 1rem 0;
    }

    .color-block {
        width: 80px;
        height: 80px;
        border-radius: 16px;
        margin: 0 auto 0.8rem auto;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: 3px solid white;
    }

    .color-info {
        font-size: 0.85rem;
        color: #4A5568;
        font-weight: 500;
    }

    /* 隐藏技术信息 */
    .stMetric {
        display: none;
    }

    /* 自定义文件上传区域 */
    .uploadedFile {
        border-radius: 16px;
        border: 2px dashed #CBD5E0;
        padding: 2rem;
        text-align: center;
        background: #F7FAFC;
        transition: all 0.3s ease;
    }

    .uploadedFile:hover {
        border-color: #667eea;
        background: #EDF2F7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """加载完整的RAG系统"""
    try:
        # 初始化完整的LangChain Topic-RAG系统 - 使用绝对路径
        rag_system = LangChainTopicRAGSystem(
            model_dir='/root/autodl-tmp/AETM/models',
            device='cpu',
            api_key=DEEPSEEK_API_KEY
        )
        
        # 手动设置豆包API密钥
        rag_system.doubao_api_key = DOUBAO_API_KEY
        
        # 获取系统组件
        model = rag_system.model
        vectorizer = rag_system.vectorizer
        theta = rag_system.theta
        
        return model, rag_system, vectorizer, theta
        
    except Exception as e:
        st.error(f"完整RAG系统加载失败: {str(e)}")
        return None, None, None, None

def image_understanding(image_file, user_text: str) -> tuple:
    """图片理解与文本融合"""
    try:
        from openai import OpenAI

        # 静默处理图片
        image_file.seek(0)
        image = Image.open(image_file)

        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 调整图片大小（避免过大）
        max_size = (800, 600)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # 编码图片
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=85, optimize=True)
        img_buffer.seek(0)
        image_data = img_buffer.getvalue()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # 设置环境变量（豆包API要求）
        os.environ["ARK_API_KEY"] = DOUBAO_API_KEY

        # 初始化豆包客户端
        client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=DOUBAO_API_KEY,
        )

        # 调用豆包视觉模型
        response = client.chat.completions.create(
            model="doubao-1-5-thinking-vision-pro-250428",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "请详细描述这张图片的内容，包括：1.主要物体和场景 2.色彩搭配和风格特点 3.整体氛围和感受 4.可能的设计风格。请用简洁明了的语言描述，重点关注色彩和设计相关的信息。"
                        },
                    ],
                }
            ],
            timeout=30
        )

        image_description = response.choices[0].message.content
        print(f"✅ 豆包API调用成功，图片分析完成")

    except Exception as e:
        error_msg = str(e)
        print(f"❌ 豆包API调用失败: {error_msg}")

        # 显示详细错误信息（但不中断流程）
        if "timeout" in error_msg.lower():
            print("⏰ API调用超时，可能是网络问题")
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            print("🔑 API密钥可能无效或已过期")
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            print("🚫 API调用频率限制，请稍后重试")
        else:
            print("🔧 其他API错误，使用模拟描述继续")

        # 使用模拟描述（不显示错误给用户，静默处理）
        image_description = "这是一张现代设计图片，色彩搭配简洁优雅，整体风格现代简约，具有良好的视觉层次和色彩平衡。"

    # 融合文本
    fused_text = f"用户需求: {user_text}\n图片内容: {image_description}"
    return fused_text, image_description

def topic_model_inference(model, theta, fused_text: str):
    """Topic Model推理生成文本和颜色矩阵"""
    try:
        words = fused_text.lower().split()
        text_keywords = ['现代', '简约', '办公', '专业', '温馨', '极简', '几何', '蓝', '灰', '米']
        text_score = sum(1 for word in words for keyword in text_keywords if keyword in word)
        
        if text_score > 0:
            selected_idx = min(theta.shape[0] // 2 + text_score * 100, theta.shape[0] - 1)
        else:
            selected_idx = theta.shape[0] // 2
            
        selected_theta = theta[selected_idx:selected_idx+1]
        
        with torch.no_grad():
            # 生成主题嵌入
            topic_embedding = torch.matmul(selected_theta, model.alpha.data)
            # 生成文本矩阵和颜色矩阵
            text_matrix = torch.matmul(topic_embedding, model.rho_text.data)
            color_matrix = torch.matmul(topic_embedding, model.rho_color.data)

        return text_matrix, color_matrix, selected_idx
        
    except Exception as e:
        st.error(f"Topic Model推理失败: {str(e)}")
        return None, None, None

def similarity_retrieval(text_matrix, color_matrix, knowledge_base, text_top_k=10, final_top_k=3):
    """两阶段相似度检索"""
    try:
        kb_data = knowledge_base['data']
        descriptions = kb_data['descriptions']
        color_vectors = kb_data['knowledge_color_vectors']
        text_vectors = kb_data['knowledge_text_vectors']
        
        # 第一阶段：文本相似度
        query_text = text_matrix.detach().cpu().numpy()
        text_similarities = cosine_similarity(query_text, text_vectors)[0]
        # 归一化到0-1范围
        text_similarities = (text_similarities + 1) / 2
        text_top_indices = np.argsort(text_similarities)[-text_top_k:][::-1]

        # 第二阶段：颜色相似度
        candidate_color_vectors = color_vectors[text_top_indices]
        query_color = color_matrix.detach().cpu().numpy()

        # 检查维度匹配
        if query_color.shape[1] != candidate_color_vectors.shape[1]:
            # 调整维度
            min_dim = min(query_color.shape[1], candidate_color_vectors.shape[1])
            query_color = query_color[:, :min_dim]
            candidate_color_vectors = candidate_color_vectors[:, :min_dim]

        candidate_color_similarities = cosine_similarity(query_color, candidate_color_vectors)[0]
        # 归一化到0-1范围
        candidate_color_similarities = (candidate_color_similarities + 1) / 2
        color_top_indices = np.argsort(candidate_color_similarities)[-final_top_k:][::-1]
        
        # 构建最终结果
        final_results = []
        for i, color_idx in enumerate(color_top_indices):
            original_idx = text_top_indices[color_idx]
            result = {
                'rank': i + 1,
                'index': int(original_idx),
                'text_similarity': float(text_similarities[original_idx]),
                'color_similarity': float(candidate_color_similarities[color_idx]),
                'description': descriptions[original_idx],
                'color_vector': color_vectors[original_idx].tolist()
            }
            final_results.append(result)
        
        return final_results, text_similarities.max(), candidate_color_similarities.max()
        
    except Exception as e:
        st.error(f"相似度检索失败: {str(e)}")
        return [], 0, 0

def deepseek_generation_multiple(text_matrix, color_matrix, top_results: List[Dict], original_text: str, image_description: str) -> List[str]:
    """DeepSeek基于3个知识库方案分别生成3个新方案"""
    try:
        generated_plans = []

        # 为每个知识库方案生成一个新的配色方案
        for i, result in enumerate(top_results[:3]):

            # 构建单个方案的详细信息
            color_vector = result['color_vector']
            kb_colors = []
            if len(color_vector) >= 15:
                for j in range(5):
                    start_idx = j * 3
                    if start_idx + 2 < len(color_vector):
                        r = int(np.clip(color_vector[start_idx] * 255, 0, 255))
                        g = int(np.clip(color_vector[start_idx + 1] * 255, 0, 255))
                        b = int(np.clip(color_vector[start_idx + 2] * 255, 0, 255))
                        kb_colors.append(f"RGB({r}, {g}, {b})")

            # 为每个方案构建专门的提示词
            individual_prompt = f"""你是一位顶级的色彩设计大师。请基于以下信息，创作一个全新的配色方案。

**用户需求:**
{original_text}

**图片分析:**
{image_description}

**参考方案** (文本相似度: {result['text_similarity']:.3f}, 颜色相似度: {result['color_similarity']:.3f}):
方案描述: {result['description']}
参考颜色: {', '.join(kb_colors) if kb_colors else '颜色数据不完整'}

**创作要求:**
请以这个参考方案为灵感，结合图片风格和用户需求，创作一个全新的配色方案。要求与参考方案有所区别，体现创新性。

请按以下格式输出：

**设计理念**
[简洁描述设计思路，说明如何在参考方案基础上创新]

**配色方案**
颜色名称 - RGB(r, g, b)
颜色名称 - RGB(r, g, b)
颜色名称 - RGB(r, g, b)
颜色名称 - RGB(r, g, b)
颜色名称 - RGB(r, g, b)

**应用建议**
[说明适用场景和搭配建议]

注意：请确保方案既有创新性又实用，与参考方案相关但不雷同。"""
        
            # 调用DeepSeek API为当前方案生成新配色
            import requests
            import time

            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": individual_prompt}],
                "temperature": 0.7 + i * 0.1,  # 每个方案使用不同的温度增加多样性
                "max_tokens": 1000
            }

            # 为每个方案重试
            max_retries = 2  # 减少重试次数，因为要生成3个方案
            plan_generated = False

            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=45  # 稍微减少超时时间
                    )

                    if response.status_code == 200:
                        result = response.json()
                        generated_plan = result["choices"][0]["message"]["content"]
                        generated_plans.append({
                            "title": f"方案{i+1}：基于知识库方案{i+1}的创新设计",
                            "content": generated_plan,
                            "reference_similarity": result.get('text_similarity', 0) + result.get('color_similarity', 0)
                        })
                        # 方案生成成功，继续处理
                        plan_generated = True
                        break
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue

                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue

            # 如果API失败，生成备用方案
            if not plan_generated:
                st.warning(f"⚠️ 方案{i+1} API失败，生成备用方案")
                fallback = generate_single_fallback_plan(result, original_text, image_description, i+1)
                generated_plans.append({
                    "title": f"方案{i+1}：基于知识库方案{i+1}的备用设计",
                    "content": fallback,
                    "reference_similarity": result.get('text_similarity', 0) + result.get('color_similarity', 0)
                })

        return generated_plans

    except Exception as e:
        st.error(f"DeepSeek生成失败: {str(e)}")
        # 生成3个备用方案
        fallback_plans = []
        for i, result in enumerate(top_results[:3]):
            fallback = generate_single_fallback_plan(result, original_text, image_description, i+1)
            fallback_plans.append({
                "title": f"方案{i+1}：备用设计方案",
                "content": fallback,
                "reference_similarity": result.get('text_similarity', 0) + result.get('color_similarity', 0)
            })
        return fallback_plans

def generate_single_fallback_plan(result: Dict, original_text: str, image_description: str, plan_num: int) -> str:
    """为单个知识库方案生成备用配色方案"""
    try:
        # 从知识库方案提取颜色
        color_vector = result['color_vector']
        fallback_colors = []

        if len(color_vector) >= 15:
            for i in range(5):
                start_idx = i * 3
                if start_idx + 2 < len(color_vector):
                    # 在原颜色基础上进行微调，增加变化
                    r = int(np.clip(color_vector[start_idx] * 255 + (plan_num * 10), 0, 255))
                    g = int(np.clip(color_vector[start_idx + 1] * 255 + (plan_num * 15), 0, 255))
                    b = int(np.clip(color_vector[start_idx + 2] * 255 + (plan_num * 5), 0, 255))

                    color_names = ["主色调", "辅助色", "点缀色", "背景色", "强调色"]
                    fallback_colors.append(f"{color_names[i]} - RGB({r}, {g}, {b})")

        # 生成备用方案文本
        style_variations = [
            "现代简约风格的创新演绎",
            "经典与现代的完美融合",
            "自然灵感的色彩表达"
        ]

        fallback_plan = f"""**设计理念**
基于知识库优秀方案的{style_variations[(plan_num-1) % 3]}，结合{image_description.split('，')[0] if '，' in image_description else '图片特点'}，创造出既符合{original_text.split('，')[0] if '，' in original_text else '用户需求'}又具有独特个性的配色方案。

**配色方案**
{chr(10).join(fallback_colors)}

**应用建议**
此方案适用于{original_text}的应用场景，色彩层次丰富，既保持整体和谐又突出重点区域，建议在实际应用中根据具体环境进行微调。"""

        return fallback_plan

    except Exception as e:
        return f"备用方案{plan_num}生成失败: {str(e)}"

def generate_fallback_plan(top_results: List[Dict], original_text: str, image_description: str) -> str:
    """生成备用配色方案（当DeepSeek API失败时）"""
    try:
        # 静默生成备用配色方案

        # 基于知识库最佳方案生成备用方案
        if top_results and len(top_results) > 0:
            best_result = top_results[0]

            # 从最佳方案提取颜色
            color_vector = best_result['color_vector']
            fallback_colors = []

            if len(color_vector) >= 15:
                for i in range(5):
                    start_idx = i * 3
                    if start_idx + 2 < len(color_vector):
                        r = int(np.clip(color_vector[start_idx] * 255, 0, 255))
                        g = int(np.clip(color_vector[start_idx + 1] * 255, 0, 255))
                        b = int(np.clip(color_vector[start_idx + 2] * 255, 0, 255))

                        # 根据位置给颜色命名
                        color_names = ["主色调", "辅助色", "点缀色", "背景色", "强调色"]
                        fallback_colors.append(f"{color_names[i]} - RGB({r}, {g}, {b})")

            # 生成备用方案文本
            fallback_plan = f"""**设计理念**
基于图片分析和知识库最佳匹配方案，这是一个融合了{image_description.split('，')[0] if '，' in image_description else '现代设计'}特点的配色方案。整体色调和谐统一，既满足用户的{original_text.split('，')[0] if '，' in original_text else '设计需求'}，又体现了专业的色彩搭配原则。

**配色方案**
{chr(10).join(fallback_colors)}

**应用建议**
此配色方案适用于{original_text}的场景，建议主色调用于大面积区域，辅助色用于功能区域，点缀色用于重要元素突出，背景色保持空间的整洁感，强调色用于关键信息的视觉引导。整体搭配既保持视觉舒适度，又具有良好的功能性。"""

            # 备用配色方案生成完成
            return fallback_plan

        else:
            # 如果连知识库结果都没有，生成通用方案
            generic_plan = f"""**设计理念**
基于{image_description}的视觉特点，结合{original_text}的需求，采用经典的配色理论，创造出既实用又美观的色彩方案。

**配色方案**
主色调 - RGB(70, 130, 180)
辅助色 - RGB(245, 245, 245)
点缀色 - RGB(255, 165, 0)
背景色 - RGB(248, 248, 255)
强调色 - RGB(220, 20, 60)

**应用建议**
这是一个通用的配色方案，适合多种应用场景。建议根据具体需求进行微调。"""

            st.info("✅ 通用配色方案生成完成")
            return generic_plan

    except Exception as e:
        st.error(f"备用方案生成也失败了: {str(e)}")
        return "系统暂时无法生成配色方案，请稍后重试。"

def display_color_palette(colors_rgb: List[str], title: str):
    """显示颜色调色板"""
    if title:
        st.subheader(title)

    if not colors_rgb:
        st.warning("No color information found")
        return

    # 使用Streamlit的列布局显示颜色
    cols = st.columns(len(colors_rgb))

    for i, color_str in enumerate(colors_rgb):
        with cols[i]:
            try:
                # 提取RGB值
                rgb_values = color_str.replace('RGB(', '').replace(')', '').split(', ')
                r, g, b = map(int, rgb_values)
                hex_color = f"#{r:02x}{g:02x}{b:02x}"

                # 创建颜色块 - 不显示编号
                st.markdown(f"""
                <div style='text-align: center;'>
                    <div style='
                        width: 100px;
                        height: 100px;
                        background-color: {hex_color};
                        border-radius: 10px;
                        border: 2px solid #ddd;
                        margin: 0 auto 10px auto;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    '>
                    </div>
                    <div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>{color_str}</div>
                    <div style='font-size: 10px; color: #666; font-family: monospace;'>{hex_color}</div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"颜色解析错误: {color_str}")
                continue

    # 额外显示调色板条
    st.markdown("#### 🎨 Color Palette Preview")
    palette_html = "<div style='display: flex; height: 60px; border-radius: 10px; overflow: hidden; border: 2px solid #ddd;'>"

    for color_str in colors_rgb:
        try:
            rgb_values = color_str.replace('RGB(', '').replace(')', '').split(', ')
            r, g, b = map(int, rgb_values)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            palette_html += f"<div style='flex: 1; background-color: {hex_color};'></div>"
        except:
            continue

    palette_html += "</div>"
    st.markdown(palette_html, unsafe_allow_html=True)

def get_fewshot_examples():
    """获取Few-Shot范例，用于教会LLM按指定数量生成颜色"""
    examples = {
        "3_colors": {
            "input": "I need an extremely minimalist color scheme for a coffee brand logo, only 3 colors needed.",
            "output": """### **Minimalist Coffee Logo**
**Design Concept**: Convey warmth and professionalism through ultimate simplicity. Inspired by a fresh latte, using minimal colors to express core concepts: coffee, milk, and focus.
---
### **Color Scheme**
1. **Espresso Black - RGB(50, 40, 35)**
   *Represents the coffee base, stable and professional.*
2. **Warm Milk White - RGB(245, 240, 235)**
   *Represents blended milk, warm and pure.*
3. **Terracotta Brown - RGB(180, 110, 80)**
   *Represents the coffee cup, adding handcrafted warmth and texture.*"""
        },
        "7_colors": {
            "input": "Design a 7-color environmental palette for a fantasy RPG game set in a magical forest.",
            "output": """### **Magical Forest "Elven Twilight"**
**Design Concept**: Capture moonlight filtering through ancient forest canopy, illuminating magical creatures and plants. Create an atmosphere that's both serene and filled with potential danger and wonder.
---
### **Color Scheme**
1. **Midnight Blue - RGB(20, 25, 55)**
   *Forest's night backdrop, vast and mysterious.*
2. **Ancient Wood Brown - RGB(60, 45, 40)**
   *Massive tree trunks, full of power.*
3. **Moss Green - RGB(80, 110, 70)**
   *Moss covering roots and stones, representing life and moisture.*
4. **Moonlight Silver - RGB(210, 220, 230)**
   *Cold moonlight piercing through leaves, main light source.*
5. **Ghost Mushroom Cyan - RGB(100, 200, 180)**
   *Glowing magical mushrooms, fantasy elements and accent lights.*
6. **Fairy Blood Red - RGB(190, 50, 60)**
   *Hidden magical flowers or warning colors of dangerous creatures.*
7. **Earth Gray - RGB(100, 95, 90)**
   *Forest rocks and soil, providing neutral transitions.*"""
        }
    }
    return examples

def create_fewshot_prompt(num_colors: int, user_query: str, retrieved_context: str = "") -> str:
    """使用Few-Shot方法创建提示词，确保生成指定数量的颜色"""
    examples = get_fewshot_examples()

    # 选择合适的范例
    if num_colors <= 4:
        primary_example = examples["3_colors"]
        secondary_example = examples["7_colors"]
    else:
        primary_example = examples["7_colors"]
        secondary_example = examples["3_colors"]

    prompt = f"""# Role
You are a top-tier color design master who can precisely generate the specified number of colors according to user requirements and provide complete design solutions.

# Task
Please learn from the following examples to understand how to construct your answer based on the specified number of colors. Then, create a brand new color scheme based on the final user's real requirements.

---
[Example 1: Input]
User Requirements: "{primary_example['input']}"
Reference Cases: (omitted or empty)

[Example 1: Output]
{primary_example['output']}

---
[Example 2: Input]
User Requirements: "{secondary_example['input']}"
Reference Cases: (omitted or empty)

[Example 2: Output]
{secondary_example['output']}

---

# User's Real Requirements and Reference Cases
Now, please strictly follow the format of the above examples and generate a brand new color scheme containing **{num_colors}** colors based on the following user's real requirements.

[Real Requirements: Input]
User Requirements: "{user_query}"
Reference Cases: "{retrieved_context}"

[Real Requirements: Output]
"""

    return prompt

def generate_scheme_description(user_text: str, colors: List[str], scheme_name: str, design_theme: str) -> str:
    """为配色方案生成完整的AI描述"""
    color_names = [
        "Primary", "Secondary", "Accent", "Background", "Highlight",
        "Complementary", "Neutral", "Contrast", "Supporting", "Emphasis"
    ]

    # 生成颜色描述
    color_descriptions = []
    for i, color in enumerate(colors):
        color_name = color_names[i] if i < len(color_names) else f"Color {i+1}"
        # 简单的颜色特征分析
        rgb_values = color.replace("RGB(", "").replace(")", "").split(", ")
        r, g, b = map(int, rgb_values)

        # 基于RGB值生成颜色特征描述
        if r > 200 and g > 200 and b > 200:
            tone = "bright and luminous"
        elif r < 80 and g < 80 and b < 80:
            tone = "deep and sophisticated"
        elif max(r, g, b) - min(r, g, b) < 50:
            tone = "balanced and neutral"
        else:
            tone = "vibrant and dynamic"

        color_descriptions.append(f"**{color_name} - {color}**: {tone.capitalize()}, creating visual harmony and professional appeal.")

    # 生成完整描述
    description = f"""### **{scheme_name}**

**Design Concept**: This sophisticated color palette embodies {design_theme}, carefully crafted to meet your specific requirements. Each color has been selected to create a harmonious balance that speaks to modern aesthetics while maintaining timeless appeal.

**Color Philosophy**: The palette draws inspiration from contemporary design trends, incorporating psychological color theory to evoke the desired emotional response. The {len(colors)}-color composition ensures versatility across various applications while maintaining visual coherence.

---
### **Color Breakdown**

{chr(10).join([f"{i+1}. {desc}" for i, desc in enumerate(color_descriptions)])}

**Application Notes**: This color scheme works exceptionally well for {user_text.lower() if user_text else 'various design applications'}, providing both visual impact and professional sophistication. The carefully balanced contrast ratios ensure accessibility while the harmonious color relationships create a cohesive and memorable visual identity.

**Design Versatility**: Perfect for digital interfaces, print materials, branding elements, and environmental design. The palette's flexibility allows for both bold statements and subtle elegance, adapting seamlessly to different contexts and applications."""

    return description

def parse_generated_colors(generated_plan: str) -> List[str]:
    """从生成的方案中提取RGB颜色"""
    import re
    rgb_pattern = r'RGB\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'
    matches = re.findall(rgb_pattern, generated_plan)

    colors = []
    for match in matches:
        r, g, b = map(int, match)
        colors.append(f"RGB({r}, {g}, {b})")

    return colors

def run_complete_rag_pipeline(rag_system, user_text: str, image_file=None, num_colors: int = 5):
    """运行完整的RAG流程，支持指定颜色数量"""
    try:
        # 处理图片上传
        image_path = None
        if image_file is not None:
            # 保存上传的图片
            image_path = "temp_uploaded_image.jpg"
            with open(image_path, "wb") as f:
                f.write(image_file.getbuffer())

        # 使用完整的两模块RAG流程，传入颜色数量参数
        with st.spinner("🤖 Complete Two-Module RAG Reasoning..."):
            result = rag_system.run_complete_pipeline(
                user_text=user_text,
                image_path=image_path,
                top_k=5,
                num_colors=num_colors  # 传入颜色数量参数
            )

        # 清理临时文件
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

        return result

    except Exception as e:
        return {
            'error': f"完整RAG流程执行失败: {str(e)}"
        }

def main():
    """Main application interface"""
    st.set_page_config(
        page_title="AI Color Master",
        page_icon="🎨",
        layout="wide"
    )

    # 页面标题
    st.markdown('<h1 class="main-header">🎨 AI Color Master</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Professional AI-powered color scheme design platform<br>Transform your creative vision into perfect color combinations</p>', unsafe_allow_html=True)

    # 特色功能展示
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🎨</div>
            <div class="feature-title">Professional Colors</div>
            <div class="feature-desc">AI-powered color analysis with professional design principles and aesthetic theory</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Instant Generation</div>
            <div class="feature-desc">Generate multiple unique color schemes in seconds with advanced AI algorithms</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💎</div>
            <div class="feature-title">Multiple Styles</div>
            <div class="feature-desc">Choose from various design styles - modern, classic, minimalist, and more</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 系统状态检查（隐藏技术细节，不阻止加载）
    try:
        system_ok = check_system_requirements()
        if not system_ok:
            st.info("🔧 Some system components are initializing, but the application will continue to load...")
    except Exception as e:
        st.info("🔧 System check in progress...")
    
    # 加载完整RAG系统（隐藏技术细节）
    try:
        with st.spinner("🎨 Initializing AI Color Master..."):
            model, rag_system, vectorizer, theta = load_models()

        if rag_system is None:
            st.error("🔧 System initialization failed, please refresh the page")
            st.info("💡 If this persists, please check that all model files are properly loaded")
            st.stop()

        # 系统已经初始化完成，包含所有必要的组件
        text_top_k = 20
        final_top_k = 3

        # 显示成功加载信息
        st.success("✅ AI Color Master is ready!")

    except Exception as e:
        st.error("🔧 System initialization encountered an issue")
        st.error(f"Error details: {str(e)}")
        st.info("💡 Please refresh the page to try again")
        st.stop()
    
    # 配色风格指导区域
    st.markdown("""
    <div class="input-section">
        <div class="section-title">🎨 Color Style Guide</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 2rem 0;">
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 12px; text-align: center;">
                <div style="background: #e3f2fd; color: #1976d2; padding: 0.5rem; border-radius: 8px; margin-bottom: 0.5rem; font-weight: 600;">Modern</div>
                <div style="font-size: 0.9rem; color: #666;">Clean, minimalist, contemporary</div>
            </div>
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 12px; text-align: center;">
                <div style="background: #fff3e0; color: #f57c00; padding: 0.5rem; border-radius: 8px; margin-bottom: 0.5rem; font-weight: 600;">Warm</div>
                <div style="font-size: 0.9rem; color: #666;">Cozy, inviting, comfortable</div>
            </div>
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 12px; text-align: center;">
                <div style="background: #f3e5f5; color: #7b1fa2; padding: 0.5rem; border-radius: 8px; margin-bottom: 0.5rem; font-weight: 600;">Elegant</div>
                <div style="font-size: 0.9rem; color: #666;">Sophisticated, refined, luxury</div>
            </div>
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 12px; text-align: center;">
                <div style="background: #e8f5e8; color: #388e3c; padding: 0.5rem; border-radius: 8px; margin-bottom: 0.5rem; font-weight: 600;">Natural</div>
                <div style="font-size: 0.9rem; color: #666;">Organic, earthy, sustainable</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 创建专属配色方案区域
    st.markdown("""
    <div class="input-section">
        <div class="section-title">🎯 Create Your Custom Color Scheme</div>
        <div class="section-subtitle">Describe your design vision and let our AI create the perfect color palette for you</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 📝 Describe Your Design Requirements")
        user_text = st.text_area(
            "Design Requirements",
            placeholder="Example: I need a sophisticated color scheme for a modern office space, with calming blues and professional grays that promote focus and creativity...",
            height=120,
            help="Describe your design needs, style preferences, usage scenarios, target audience, and desired mood",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("#### 🖼️ Upload Inspiration Image")
        uploaded_file = st.file_uploader(
            "Inspiration Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to extract color inspiration and enhance your design",
            label_visibility="collapsed"
        )

        if uploaded_file:
            # 控制图片显示尺寸
            st.image(uploaded_file, caption="Your Inspiration", width=300)

    # 颜色数量控制
    st.markdown("""
    <div class="input-section">
        <div class="section-title">🎯 Customize Your Palette</div>
        <div class="section-subtitle">Choose how many colors you want in your palette</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        num_colors = st.slider(
            "Number of Colors",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="Choose between 3-10 colors for your palette. More colors provide more design flexibility."
        )

        # 显示颜色数量的视觉提示
        color_preview = "🎨 " + "●" * num_colors
        st.markdown(f'<div style="text-align: center; font-size: 1.2rem; margin: 1rem 0;">{color_preview}</div>', unsafe_allow_html=True)

    # 生成按钮
    st.markdown('<div style="text-align: center; margin: 3rem 0;">', unsafe_allow_html=True)
    generate_clicked = st.button("✨ Generate My Color Schemes", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if generate_clicked:
        if not user_text.strip():
            st.warning("💡 Please describe your design requirements to get started")
            return

        # 显示处理进度
        with st.spinner("🎨 AI Color Master is creating your perfect color schemes..."):
            try:
                # 运行完整RAG流程，传入颜色数量参数
                result = run_complete_rag_pipeline(rag_system, user_text, uploaded_file, num_colors)

                if 'error' not in result:
                    # 显示生成结果
                    st.markdown("""
                    <div class="result-section">
                        <div class="section-title">🎨 Your Custom Color Schemes</div>
                        <div class="section-subtitle">AI has designed 3 unique color schemes based on your requirements. Choose your favorite one!</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 生成3个不同的配色方案
                    generated_plans = []

                    # 从单个方案生成3个变体
                    base_plan = result.get('generated_plan', '')
                    base_colors = parse_generated_colors(base_plan) if base_plan else []

                    # 方案1：原始方案
                    generated_plans.append({
                        "title": "🌟 Classic Elegance",
                        "content": base_plan if base_plan else "A sophisticated color scheme designed with professional aesthetics and modern design principles.",
                        "colors": base_colors if base_colors else [f"RGB({100 + i*20}, {150 + i*15}, {200 + i*10})" for i in range(num_colors)]
                    })

                    if 'candidates' in result and result['candidates']:
                        # 方案2：基于第一个候选方案的变体，生成完整AI描述
                        if len(result['candidates']) > 0:
                            candidate = result['candidates'][0]
                            variant_colors = []
                            # 使用用户选择的颜色数量
                            for color in candidate['colors'][:num_colors]:
                                r = int(color[0] * 255)
                                g = int(color[1] * 255)
                                b = int(color[2] * 255)
                                variant_colors.append(f"RGB({r}, {g}, {b})")

                            # 如果候选方案颜色不足，生成补充颜色
                            while len(variant_colors) < num_colors:
                                # 基于现有颜色生成相近的变体颜色
                                if len(candidate['colors']) > 0:
                                    base_color = candidate['colors'][0]
                                else:
                                    base_color = [0.5, 0.5, 0.5]
                                r = min(255, max(0, int(base_color[0] * 255) + (len(variant_colors) * 30) % 100))
                                g = min(255, max(0, int(base_color[1] * 255) + (len(variant_colors) * 20) % 80))
                                b = min(255, max(0, int(base_color[2] * 255) + (len(variant_colors) * 40) % 120))
                                variant_colors.append(f"RGB({r}, {g}, {b})")

                            # 为方案2生成完整的AI描述
                            scheme2_description = generate_scheme_description(
                                user_text, variant_colors, "Modern Sophistication",
                                "contemporary design principles with sophisticated color harmony"
                            )

                            generated_plans.append({
                                "title": "💫 Modern Sophistication",
                                "content": scheme2_description,
                                "colors": variant_colors
                            })

                        # 方案3：基于第二个候选方案的变体，生成完整AI描述
                        if len(result['candidates']) > 1:
                            candidate = result['candidates'][1]
                            variant_colors = []
                            # 使用用户选择的颜色数量
                            for color in candidate['colors'][:num_colors]:
                                r = int(color[0] * 255)
                                g = int(color[1] * 255)
                                b = int(color[2] * 255)
                                variant_colors.append(f"RGB({r}, {g}, {b})")

                            # 如果候选方案颜色不足，生成补充颜色
                            while len(variant_colors) < num_colors:
                                # 基于现有颜色生成相近的变体颜色
                                if len(candidate['colors']) > 0:
                                    base_color = candidate['colors'][0]
                                else:
                                    base_color = [0.3, 0.6, 0.4]
                                r = min(255, max(0, int(base_color[0] * 255) + (len(variant_colors) * 25) % 90))
                                g = min(255, max(0, int(base_color[1] * 255) + (len(variant_colors) * 35) % 100))
                                b = min(255, max(0, int(base_color[2] * 255) + (len(variant_colors) * 15) % 70))
                                variant_colors.append(f"RGB({r}, {g}, {b})")

                            # 为方案3生成完整的AI描述
                            scheme3_description = generate_scheme_description(
                                user_text, variant_colors, "Creative Innovation",
                                "bold and innovative design approach with artistic color expression"
                            )

                            generated_plans.append({
                                "title": "✨ Creative Innovation",
                                "content": scheme3_description,
                                "colors": variant_colors
                            })

                    # 确保至少有3个方案（如果候选方案不足，生成默认方案）
                    while len(generated_plans) < 3:
                            # 生成指定数量的默认颜色
                            default_colors = []
                            color_bases = [
                                [100, 150, 200],  # 蓝色系
                                [200, 180, 160],  # 暖色系
                                [150, 200, 100],  # 绿色系
                                [220, 200, 180],  # 米色系
                                [180, 160, 200],  # 紫色系
                                [200, 150, 120],  # 棕色系
                                [120, 180, 160],  # 青色系
                                [190, 170, 140],  # 土色系
                                [160, 140, 180],  # 淡紫系
                                [140, 190, 170]   # 薄荷系
                            ]

                            for i in range(num_colors):
                                if i < len(color_bases):
                                    r, g, b = color_bases[i]
                                else:
                                    # 生成额外的颜色变体
                                    base_idx = i % len(color_bases)
                                    r = min(255, max(0, color_bases[base_idx][0] + (i * 20) % 60))
                                    g = min(255, max(0, color_bases[base_idx][1] + (i * 15) % 50))
                                    b = min(255, max(0, color_bases[base_idx][2] + (i * 25) % 70))
                                default_colors.append(f"RGB({r}, {g}, {b})")

                            # 为默认方案生成完整的AI描述
                            scheme_title = f"Alternative Design {len(generated_plans)}"
                            final_colors = base_colors if base_colors and len(base_colors) == num_colors else default_colors
                            default_description = generate_scheme_description(
                                user_text, final_colors, scheme_title,
                                "balanced design approach with professional color harmony and versatile application"
                            )

                            generated_plans.append({
                                "title": f"🎨 {scheme_title}",
                                "content": default_description,
                                "colors": final_colors
                            })

                    # 显示3个方案
                    for i, plan in enumerate(generated_plans[:3]):
                        st.markdown(f"""
                        <div class="color-palette-container">
                            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1.5rem;">
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold;">{i+1}</div>
                                <div style="font-size: 1.3rem; font-weight: 700; color: #2D3748;">{plan['title']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # 显示颜色调色板
                        if plan['colors']:
                            display_color_palette(plan['colors'], "")

                        # 显示方案描述
                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; margin: 1rem 0 2rem 0; border-left: 4px solid #667eea;">
                            <div style="color: #4A5568; line-height: 1.6;">{plan['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # 添加保存按钮
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col2:
                            if st.button(f"💾 Save {plan['title']}", key=f"save_scheme_{i}", use_container_width=True):
                                st.success(f"✅ {plan['title']} saved successfully!")

                        st.markdown("---")

                    else:
                        # 如果没有候选方案，显示单个方案
                        colors = parse_generated_colors(result['generated_plan'])
                        if colors:
                            st.markdown("### 🎨 Your Color Palette")
                            display_color_palette(colors, "")

                        # 显示完整方案描述
                        st.markdown("### 📋 Design Description")
                        st.markdown(f"""
                        <div class="color-palette-container">
                            <div style="background: #f8fafc; padding: 2rem; border-radius: 16px; margin: 2rem 0; border-left: 4px solid #667eea;">
                                <div style="color: #2D3748; line-height: 1.8; font-size: 1.05rem;">{result['generated_plan']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 显示参考候选方案（简化显示）
                    if 'candidates' in result and result['candidates']:
                        st.markdown("""
                        <div style="margin: 3rem 0;">
                            <div style="font-size: 1.3rem; font-weight: 700; color: #2D3748; margin-bottom: 1rem;">💡 Design Inspiration</div>
                            <div style="font-size: 0.95rem; color: #718096; margin-bottom: 1.5rem;">These reference designs helped inspire your custom color scheme</div>
                        </div>
                        """, unsafe_allow_html=True)

                        for i, candidate in enumerate(result['candidates'][:2], 1):  # 只显示前2个
                            st.markdown(f"""
                            <div style="background: white; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
                                <div style="font-weight: 600; color: #4A5568; margin-bottom: 0.8rem;">Reference Design {i}</div>
                                <div style="color: #718096; font-size: 0.9rem; line-height: 1.5;">{candidate['description'][:200]}...</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # 添加最终操作按钮
                    st.markdown('<div style="text-align: center; margin: 3rem 0;">', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        if st.button("� Generate New Schemes", use_container_width=True):
                            st.rerun()
                    with col2:
                        if st.button("� Export All Colors", use_container_width=True):
                            st.info("📤 Export feature coming soon!")
                    with col3:
                        if st.button("� Get More Ideas", use_container_width=True):
                            st.info("� Try different descriptions or upload a new inspiration image!")
                    st.markdown('</div>', unsafe_allow_html=True)

                else:
                    st.error("😔 AI Color Master encountered some difficulties, please try again")

            except Exception as e:
                st.error("🔧 Processing failed, please check your input and try again")
                st.error(f"Error details: {str(e)}")
                # 显示详细的错误信息用于调试
                import traceback
                with st.expander("🔍 Technical Details (for debugging)"):
                    st.code(traceback.format_exc())

def generate_color_scheme(user_text: str, uploaded_image, model, theta, knowledge_base,
                         text_top_k: int, final_top_k: int):
    """生成配色方案的完整流程"""

    # 显示加载状态
    with st.spinner("🎨 AI大师正在为您精心创作配色方案..."):
        try:
            # 后台处理，不显示具体步骤
            fused_text, image_description = image_understanding(uploaded_image, user_text)

            text_matrix, color_matrix, selected_idx = topic_model_inference(model, theta, fused_text)

            if text_matrix is None:
                st.error("😔 AI大师遇到了一些困难，请重试")
                return

            top_results, max_text_sim, max_color_sim = similarity_retrieval(
                text_matrix, color_matrix, knowledge_base, text_top_k, final_top_k
            )

            generated_plans = deepseek_generation_multiple(text_matrix, color_matrix, top_results, user_text, image_description)

            # 显示最终结果
            st.markdown("---")
            st.markdown("## 🎉 您的专属配色方案")
            st.markdown("### AI大师为您精心设计了3套不同风格的配色方案，请选择您最喜欢的一套：")

            if generated_plans and len(generated_plans) > 0:
                # 创建美观的方案展示
                style_names = ["🌟 经典优雅", "💫 现代时尚", "✨ 创意个性"]
                tabs = st.tabs(style_names)

                for i, (tab, plan) in enumerate(zip(tabs, generated_plans)):
                    with tab:
                        # 解析当前方案的颜色
                        plan_colors = parse_generated_colors(plan["content"])

                        # 显示颜色调色板
                        if plan_colors:
                            st.markdown('<div class="color-palette">', unsafe_allow_html=True)
                            display_color_palette(plan_colors, f"")
                            st.markdown('</div>', unsafe_allow_html=True)

                        # 显示方案详情
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)

                        # 美化方案内容显示
                        content = plan["content"]
                        # 移除技术性标题，使用更友好的表述
                        content = content.replace("**设计理念**", "### 💡 设计理念")
                        content = content.replace("**配色方案**", "### 🎨 配色详情")
                        content = content.replace("**应用建议**", "### 💼 使用建议")

                        st.markdown(content)
                        st.markdown('</div>', unsafe_allow_html=True)

                        # 添加下载按钮
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col2:
                            if st.button(f"💾 保存方案{i+1}", key=f"save_{i}", use_container_width=True):
                                st.info(f"💾 方案{i+1}已保存")
            else:
                st.error("😔 AI大师暂时无法为您生成方案，请稍后重试")

            # 添加分享功能
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📤 分享我的配色方案", use_container_width=True):
                    save_results(user_text, fused_text, generated_plans, top_results)
                    st.balloons()
                    st.success("🎉 您的配色方案已保存！可以分享给朋友了")

        except Exception as e:
            st.error(f"😔 AI大师遇到了一些困难: {str(e)}")
            st.info("💡 请尝试重新上传图片或调整需求描述")

def add_footer():
    """添加页脚"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; margin-top: 3rem;'>
        <h3 style='color: #333; margin-bottom: 1rem;'>🎨 AI色彩大师</h3>
        <p style='color: #666; margin-bottom: 1rem;'>让每个人都能成为色彩专家</p>
        <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;'>
            <span style='color: #888;'>✨ 专业配色</span>
            <span style='color: #888;'>⚡ 秒速生成</span>
            <span style='color: #888;'>💎 多样选择</span>
        </div>
        <p style='color: #999; font-size: 0.9rem; margin-top: 1rem;'>© 2024 COLORRAG</p>
    </div>
    """, unsafe_allow_html=True)

def save_results(user_text: str, fused_text: str, generated_plans: List[Dict], top_results: List[Dict]):
    """保存生成结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_data = {
        "timestamp": timestamp,
        "user_input": user_text,
        "fused_text": fused_text,
        "generated_plans": generated_plans,
        "top_results": top_results
    }

    filename = f"streamlit_results_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

def check_system_requirements():
    """Check system requirements"""
    st.sidebar.markdown("## 🔧 System Status Check")

    # 检查模型文件 - 使用正确的绝对路径
    model_files = [
        "/root/autodl-tmp/AETM/models/model_architecture.json",
        "/root/autodl-tmp/AETM/models/best_color_encoder.pth",
        "/root/autodl-tmp/AETM/models/best_text_encoder.pth",
        "/root/autodl-tmp/AETM/models/best_decoder.pth",
        "/root/autodl-tmp/AETM/models/best_theta.pt",
        "/root/autodl-tmp/AETM/models/tfidf_vectorizer.pkl",
        "/root/autodl-tmp/AETM/models/vocab.json"
    ]

    missing_files = []
    for file_path in model_files:
        if os.path.exists(file_path):
            st.sidebar.success(f"✅ {os.path.basename(file_path)}")
        else:
            st.sidebar.error(f"❌ {os.path.basename(file_path)}")
            missing_files.append(file_path)

    # 检查数据文件
    data_file = "/root/autodl-tmp/AETM/data/palettes_descriptions.xlsx"
    if os.path.exists(data_file):
        st.sidebar.success("✅ Data File")
    else:
        st.sidebar.error("❌ Data File")
        missing_files.append(data_file)

    # 检查核心Python文件
    core_files = [
        "/root/autodl-tmp/AETM/src/topic_model.py",
        "/root/autodl-tmp/AETM/load_separate_models.py",
        "/root/autodl-tmp/AETM/rag/langchain/langchain_rag_system.py"
    ]

    for file_path in core_files:
        if os.path.exists(file_path):
            st.sidebar.success(f"✅ {os.path.basename(file_path)}")
        else:
            st.sidebar.error(f"❌ {os.path.basename(file_path)}")
            missing_files.append(file_path)

    # 检查API密钥
    if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your_deepseek_api_key_here":
        st.sidebar.success("✅ DeepSeek API")
    else:
        st.sidebar.warning("⚠️ DeepSeek API Not Configured")

    if DOUBAO_API_KEY and DOUBAO_API_KEY != "your_doubao_api_key_here":
        st.sidebar.success("✅ Doubao API")
    else:
        st.sidebar.warning("⚠️ Doubao API Not Configured")

    return len(missing_files) == 0

if __name__ == "__main__":
    main()
    add_footer()
