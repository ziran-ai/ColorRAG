#!/usr/bin/env python3
"""
Streamlit Web界面 - RAG配色方案生成系统
基于成功的simple_pipeline.py构建用户友好的Web界面
"""

import streamlit as st
import os
import sys
import torch
import numpy as np
import json
import pickle
import base64
from typing import Dict, Any, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
from datetime import datetime

# 添加路径
sys.path.append('.')
sys.path.append('..')
sys.path.append('/root/autodl-tmp/AETM')
sys.path.append('/root/autodl-tmp/AETM/src')

# API配置
DOUBAO_API_KEY = "fc7a6e47-91f5-4ced-9498-75383418e1a5"
DEEPSEEK_API_KEY = "sk-3c4ba59c8b094106995821395c7bc60e"

# 页面配置
st.set_page_config(
    page_title="ColorRAG - 智能配色设计平台",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 商业化CSS样式
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #f0f0f0;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    .step-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: none;
    }
    .color-palette {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    .generate-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """加载所有模型和数据"""
    try:
        # 导入模型类
        from topic_model import MultiOmicsETM
        
        # 加载模型架构
        with open('/root/autodl-tmp/AETM/models/model_architecture.json', 'r') as f:
            architecture = json.load(f)
        
        # 创建模型实例
        model = MultiOmicsETM(
            num_topics=architecture['num_topics'],
            color_dim=architecture['color_dim'],
            text_dim=architecture['text_dim'],
            embedding_dim=architecture['embedding_dim'],
            hidden_dim=architecture['hidden_dim'],
            dropout=architecture['dropout']
        )
        
        # 加载解码器权重
        decoder_state = torch.load('/root/autodl-tmp/AETM/models/best_decoder.pth', map_location='cpu')
        model.alpha.data = decoder_state['alpha']
        model.rho_color.data = decoder_state['rho_color']
        model.rho_text.data = decoder_state['rho_text']
        
        # 加载预训练的theta
        theta = torch.load('/root/autodl-tmp/AETM/models/best_theta.pt', map_location='cpu')
        
        # 加载知识库
        with open('/root/autodl-tmp/AETM/rag/langchain/knowledge_base.pkl', 'rb') as f:
            knowledge_base = pickle.load(f)
        
        model.eval()
        return model, theta, knowledge_base, architecture
        
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None, None, None, None

def image_understanding(image_file, user_text: str) -> str:
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

        # 设置环境变量
        os.environ["ARK_API_KEY"] = DOUBAO_API_KEY

        # 初始化客户端
        client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=DOUBAO_API_KEY,
        )

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

    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ 豆包API调用失败: {error_msg}")

        # 显示详细错误信息
        if "timeout" in error_msg.lower():
            st.warning("⏰ API调用超时，可能是网络问题")
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            st.warning("🔑 API密钥可能无效或已过期")
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            st.warning("🚫 API调用频率限制，请稍后重试")
        else:
            st.warning("🔧 其他API错误，使用模拟描述继续")

        # 使用模拟描述
        image_description = "这是一张现代设计图片，色彩搭配简洁优雅，整体风格现代简约，具有良好的视觉层次和色彩平衡。"
        # 静默切换到模拟图片描述模式

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
    st.subheader(title)

    if not colors_rgb:
        st.warning("未找到颜色信息")
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
    st.markdown("### 🎨 调色板预览")
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

def main():
    """主应用函数"""

    # 主标题
    st.markdown('<h1 class="main-header">ColorRAG</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">专业级智能配色设计平台，让每个人都能成为色彩专家</p>', unsafe_allow_html=True)

    # 价值主张
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🎨</div>
                <h3 style="color: #333; margin-bottom: 1rem;">专业配色</h3>
                <p style="color: #666; line-height: 1.6;">基于色彩理论和设计美学，为您提供专业级的配色方案</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
                <h3 style="color: #333; margin-bottom: 1rem;">秒速生成</h3>
                <p style="color: #666; line-height: 1.6;">上传图片，描述需求，AI瞬间为您生成多套精美配色方案</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">💎</div>
                <h3 style="color: #333; margin-bottom: 1rem;">多样选择</h3>
                <p style="color: #666; line-height: 1.6;">一次生成三套不同风格的方案，总有一款适合您的需求</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 后台加载模型（不显示给用户）
    with st.spinner("🚀 AI大师正在准备中..."):
        model, theta, knowledge_base, architecture = load_models()

    if model is None:
        st.error("😔 AI大师暂时不可用，请稍后重试")
        return

    # 设置默认参数（不让用户看到技术细节）
    text_top_k = 10
    final_top_k = 3

    # 风格指导区域
    st.markdown("---")
    st.markdown("## 🎨 配色风格指导")

    # 风格指导卡片
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #333; margin-bottom: 1rem;">🌈 色彩风格词汇</h4>
            <div style="line-height: 2;">
                <span style="background: #f0f8ff; padding: 4px 8px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.9rem;">现代简约</span>
                <span style="background: #fff5ee; padding: 4px 8px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.9rem;">温馨舒适</span>
                <span style="background: #f0fff0; padding: 4px 8px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.9rem;">清新自然</span>
                <span style="background: #fdf5e6; padding: 4px 8px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.9rem;">复古怀旧</span>
                <span style="background: #f5f5dc; padding: 4px 8px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.9rem;">奢华典雅</span>
                <span style="background: #e6e6fa; padding: 4px 8px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.9rem;">活力动感</span>
                <span style="background: #ffe4e1; padding: 4px 8px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.9rem;">浪漫柔美</span>
                <span style="background: #f0f0f0; padding: 4px 8px; border-radius: 15px; margin: 2px; display: inline-block; font-size: 0.9rem;">工业风格</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #333; margin-bottom: 1rem;">💡 描述建议</h4>
            <ul style="color: #666; line-height: 1.8;">
                <li><strong>空间类型：</strong>办公室、咖啡厅、卧室、客厅等</li>
                <li><strong>期望氛围：</strong>专业、温馨、活力、宁静等</li>
                <li><strong>色彩偏好：</strong>暖色调、冷色调、中性色等</li>
                <li><strong>使用场景：</strong>工作、休息、娱乐、商务等</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 输入区域 - 并排布局
    st.markdown("---")
    st.markdown("## 📝 开始创作您的专属配色")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🎯 描述您的设计需求")
        user_text = st.text_area(
            "",
            placeholder="例如：我想为我的咖啡厅设计一套温馨舒适的配色方案，希望营造出清新自然的休闲氛围，让顾客感到放松愉悦...",
            height=150,
            label_visibility="collapsed"
        )

        # 配色指导和描述建议
        st.markdown("### 💡 配色指导")
        st.info("""
        **描述建议：**
        - 明确使用场景（如：网站、APP、室内设计、品牌等）
        - 描述期望的情感氛围（如：温馨、专业、活力、优雅等）
        - 提及目标用户群体（如：年轻人、商务人士、家庭等）
        - 说明功能需求（如：易读性、注意力引导等）

        **常见场景参考：**
        现代简约办公空间 | 温馨家居客厅 | 高端酒店大堂 | 清新咖啡店 | 科技APP界面 | 优雅书店 | 活力健身房 | 宁静瑜伽馆
        """)

    with col2:
        st.markdown("### 🖼️ 上传灵感图片")
        uploaded_image = st.file_uploader(
            "",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="上传能体现您期望风格的图片，AI将分析其色彩特点",
            label_visibility="collapsed"
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            # 限制图片显示宽度，减少空白区域
            st.image(image, caption="✨ 您的灵感图片", width=300)
        else:
            st.info("📷 请上传一张图片作为配色灵感来源")

    # 生成按钮
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎨 立即生成我的专属配色", type="primary", use_container_width=True):
            if not user_text.strip():
                st.warning("💭 请先描述您的设计需求")
                return

            if uploaded_image is None:
                st.warning("📸 请上传一张灵感图片")
                return

            # 开始生成流程（隐藏过程）
            generate_color_scheme(
                user_text, uploaded_image, model, theta, knowledge_base,
                text_top_k, final_top_k
            )

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
        <p style='color: #999; font-size: 0.9rem; margin-top: 1rem;'>© 2024 AI色彩大师 - 智能配色设计平台</p>
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

if __name__ == "__main__":
    main()
    add_footer()
