#!/usr/bin/env python3
"""
LangChain RAG系统Web平台
基于Streamlit构建，集成LangChain RAG系统
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import re
import tempfile
import base64

# 添加路径
sys.path.append('./langchain')

try:
    from simple_langchain_rag import SimpleLangChainRAG
    print("✅ 成功导入LangChain RAG系统")
except ImportError as e:
    print(f"❌ LangChain RAG系统导入失败: {e}")
    st.error(f"LangChain RAG系统导入失败: {e}")
    st.stop()

# 页面配置
st.set_page_config(
    page_title="智能配色设计助手 - LangChain版",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .color-box {
        width: 100px;
        height: 60px;
        margin: 5px;
        border-radius: 8px;
        display: inline-block;
        border: 2px solid #ddd;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class LangChainRAGWebApp:
    """LangChain RAG Web应用"""
    
    def __init__(self):
        self.rag_system = None
        self.load_system()
    
    def load_system(self):
        """加载LangChain RAG系统"""
        try:
            with st.spinner("🚀 正在加载LangChain RAG系统..."):
                self.rag_system = SimpleLangChainRAG(data_path='../data/palettes_descriptions.xlsx')
            st.success("✅ LangChain RAG系统加载成功！")
        except Exception as e:
            st.error(f"❌ 系统加载失败: {e}")
            self.rag_system = None
    
    def is_chinese(self, text):
        """检测是否包含中文"""
        return re.search('[\u4e00-\u9fff]', text) is not None
    
    def rgb_to_hex(self, rgb):
        """RGB转HEX"""
        if isinstance(rgb, list) and len(rgb) == 3:
            r, g, b = [int(c * 255) if c <= 1 else int(c) for c in rgb]
            return f"#{r:02x}{g:02x}{b:02x}"
        return "#000000"
    
    def display_color_palette(self, colors, title="配色方案"):
        """显示配色方案"""
        st.markdown(f"### {title}")
        
        # 创建颜色条
        cols = st.columns(5)
        for i, color in enumerate(colors):
            with cols[i]:
                hex_color = self.rgb_to_hex(color)
                rgb_str = f"RGB({int(color[0]*255 if color[0]<=1 else color[0])}, {int(color[1]*255 if color[1]<=1 else color[1])}, {int(color[2]*255 if color[2]<=1 else color[2])})"
                
                st.markdown(f"""
                <div style="
                    background-color: {hex_color};
                    height: 80px;
                    border-radius: 8px;
                    border: 2px solid #ddd;
                    margin-bottom: 10px;
                "></div>
                <div style="text-align: center; font-size: 0.8rem;">
                    <strong>颜色 {i+1}</strong><br>
                    {hex_color}<br>
                    {rgb_str}
                </div>
                """, unsafe_allow_html=True)
    
    def parse_color_schemes(self, generated_text):
        """解析生成的多个配色方案"""
        import re
        
        schemes = []
        
        # 多种匹配模式
        patterns = [
            # 模式1：标准格式
            r'### 方案[一二三]：(.+?)\n\*\*设计理念：\*\*(.+?)\n\*\*配色方案：\*\*(.+?)\n\*\*应用建议：\*\*(.+?)\n\*\*创新点：\*\*(.+?)(?=\n### |$)',
            # 模式2：没有星号的格式
            r'### 方案[一二三]：(.+?)\n设计理念：(.+?)\n配色方案：(.+?)\n应用建议：(.+?)\n创新点：(.+?)(?=\n### |$)',
            # 模式3：更宽松的格式
            r'方案[一二三]：(.+?)(?:\n|$)(.+?)(?:配色方案|颜色)(.+?)(?:应用|建议)(.+?)(?:创新|特点)(.+?)(?=方案|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, generated_text, re.DOTALL | re.IGNORECASE)
            if matches:
                break
        else:
            # 如果都不匹配，尝试简单分割
            sections = generated_text.split('###')
            matches = []
            for section in sections[1:]:  # 跳过第一个空部分
                if '方案' in section:
                    matches.append((section, '', '', '', ''))
        
        # 提取RGB颜色的通用函数
        def extract_colors(text):
            rgb_pattern = r'RGB\((\d+),\s*(\d+),\s*(\d+)\)'
            rgb_matches = re.findall(rgb_pattern, text)
            
            colors = []
            for rgb_match in rgb_matches:
                r, g, b = map(int, rgb_match)
                colors.append([r/255.0, g/255.0, b/255.0])
            
            # 如果没有找到RGB，尝试查找其他数字模式
            if not colors:
                number_pattern = r'(\d{1,3})[,\s]+(\d{1,3})[,\s]+(\d{1,3})'
                number_matches = re.findall(number_pattern, text)
                for match in number_matches:
                    r, g, b = map(int, match)
                    if all(0 <= x <= 255 for x in [r, g, b]):
                        colors.append([r/255.0, g/255.0, b/255.0])
            
            # 如果还是没有颜色，生成一些默认颜色
            if not colors:
                default_colors = [
                    [0.2, 0.4, 0.8],  # 蓝色
                    [0.8, 0.2, 0.2],  # 红色
                    [0.2, 0.8, 0.2],  # 绿色
                    [0.8, 0.8, 0.2],  # 黄色
                    [0.6, 0.6, 0.6]   # 灰色
                ]
                colors = default_colors
            
            # 确保有5个颜色
            while len(colors) < 5:
                colors.append([0.5, 0.5, 0.5])
            
            return colors[:5]
        
        for i, match in enumerate(matches):
            if len(match) >= 5:
                scheme_name = match[0].strip()
                design_concept = match[1].strip() if match[1] else f"方案{i+1}的设计理念"
                color_section = match[2].strip() if match[2] else match[0]  # 如果没有单独的颜色部分，使用整个文本
                application = match[3].strip() if match[3] else "适用于多种场景"
                innovation = match[4].strip() if match[4] else "独特的配色组合"
            else:
                # 简化处理
                scheme_name = f"方案{i+1}"
                design_concept = "现代设计理念"
                color_section = match[0] if match else ""
                application = "多场景应用"
                innovation = "创新配色"
            
            colors = extract_colors(color_section + str(match))
            
            schemes.append({
                'name': scheme_name,
                'design_concept': design_concept,
                'colors': colors,
                'color_section': color_section,
                'application': application,
                'innovation': innovation
            })
        
        return schemes

    def display_retrieved_knowledge(self, retrieved_docs):
        """显示检索到的知识"""
        st.markdown("### 🔍 检索到的专业知识")

        for i, doc in enumerate(retrieved_docs, 1):
            with st.expander(f"参考方案 {i}: {doc['name']}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**描述:**")
                    st.write(doc['description'])

                with col2:
                    st.markdown("**配色:**")
                    self.display_color_palette(doc['colors'], f"参考配色 {i}")

    def run_rag_generation(self, user_input, uploaded_image):
        """运行RAG生成"""
        if not self.rag_system:
            st.error("❌ RAG系统未加载")
            return None

        if not user_input.strip():
            st.warning("⚠️ 请输入设计需求")
            return None

        if uploaded_image is None:
            st.warning("⚠️ 请上传参考图片")
            return None

        try:
            # 保存上传的图片到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_image.getvalue())
                temp_image_path = tmp_file.name

            # 显示进度
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🔄 正在处理用户需求...")
            progress_bar.progress(20)

            status_text.text("🖼️ 正在分析图片...")
            progress_bar.progress(40)

            status_text.text("🔍 正在检索相关知识...")
            progress_bar.progress(60)

            status_text.text("🎨 正在生成配色方案...")
            progress_bar.progress(80)

            # 运行RAG流程
            result = self.rag_system.run_rag_pipeline(
                user_input=user_input,
                image_path=temp_image_path
            )

            progress_bar.progress(100)
            status_text.text("✅ 生成完成！")

            # 清理临时文件
            os.unlink(temp_image_path)

            return result

        except Exception as e:
            st.error(f"❌ 生成失败: {e}")
            return None

    def main(self):
        """主界面"""
        # 标题
        st.markdown('<h1 class="main-header">🎨 智能配色设计助手</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">基于LangChain的专业配色方案生成系统</p>', unsafe_allow_html=True)

        # 侧边栏
        with st.sidebar:
            st.markdown("## ⚙️ 系统设置")

            # 系统状态
            if self.rag_system:
                st.success("🟢 LangChain RAG系统已就绪")
            else:
                st.error("🔴 系统未加载")
                if st.button("🔄 重新加载系统"):
                    self.load_system()
                    st.rerun()

            st.markdown("---")
            st.markdown("## 📖 使用说明")
            st.markdown("""
            1. **详细描述需求**: 包含风格、色调、场景、氛围等维度
            2. **选择快速标签**: 可选择预设标签快速构建描述
            3. **上传灵感图片**: 提供你喜欢的设计或色彩参考
            4. **生成配色方案**: 获得基于AI分析的专业配色
            5. **查看详细结果**: 包含配色、检索知识和应用建议
            """)

            st.markdown("---")
            st.markdown("## 🎯 系统特点")
            st.markdown("""
            - ✅ **智能理解**: 深度分析文本需求和图片灵感
            - ✅ **专业知识库**: 基于10,702条专业配色数据
            - ✅ **多维度输入**: 支持风格、色调、场景等维度描述
            - ✅ **个性化定制**: 完全根据你的需求生成独特方案
            - ✅ **LangChain架构**: 模块化、可扩展的AI系统
            """)

        # 主内容区
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<h2 class="sub-header">📝 输入设计需求</h2>', unsafe_allow_html=True)

            # 用户输入
            user_input = st.text_area(
                "请详细描述你的配色需求:",
                placeholder="例如: 我想为咖啡店设计一套温暖舒适的配色方案，风格要现代简约，色调偏暖，适合营造放松的氛围...",
                height=120
            )

            # 输入建议指导
            st.markdown("**💡 建议描述以下维度:**")
            with st.expander("📋 查看详细输入建议", expanded=False):
                st.markdown("""
                **🎨 设计风格维度:**
                - 现代简约、古典奢华、工业风、北欧风、日式禅意、复古怀旧等

                **🌈 色调偏好:**
                - 暖色调、冷色调、中性色调、高饱和度、低饱和度、单色系、对比色等

                **🏢 应用场景:**
                - 办公空间、餐厅、咖啡店、酒店、住宅、商店、网站、APP、包装设计等

                **😊 情感氛围:**
                - 专业严肃、温暖舒适、活力充沛、宁静放松、奢华高端、亲和友好等

                **🎯 目标用户:**
                - 商务人士、年轻群体、家庭用户、高端客户、儿童等

                **📱 使用媒介:**
                - 室内装修、网页设计、移动应用、印刷品、包装、品牌标识等

                **✨ 示例完整描述:**
                "我需要为一家面向年轻白领的精品咖啡店设计配色方案。风格要现代简约但不失温暖，色调偏暖但不过于鲜艳，要营造专业而放松的氛围。主要用于店面装修和品牌设计，希望顾客感受到品质感和舒适感。"
                """)

            # 快速标签选择
            st.markdown("**🏷️ 快速标签选择 (可多选):**")
            col1_1, col1_2, col1_3 = st.columns(3)

            with col1_1:
                st.markdown("**风格:**")
                style_tags = st.multiselect(
                    "选择风格标签",
                    ["现代简约", "古典奢华", "工业风", "北欧风", "日式禅意", "复古怀旧", "未来科技"],
                    key="style_tags"
                )

            with col1_2:
                st.markdown("**色调:**")
                tone_tags = st.multiselect(
                    "选择色调标签",
                    ["暖色调", "冷色调", "中性色调", "高饱和度", "低饱和度", "单色系", "对比色"],
                    key="tone_tags"
                )

            with col1_3:
                st.markdown("**场景:**")
                scene_tags = st.multiselect(
                    "选择应用场景",
                    ["办公空间", "餐厅咖啡", "酒店民宿", "住宅家居", "零售商店", "网站APP", "包装设计"],
                    key="scene_tags"
                )

            # 自动生成标签描述
            if style_tags or tone_tags or scene_tags:
                tag_description = "基于选择的标签: "
                if style_tags:
                    tag_description += f"风格({', '.join(style_tags)}) "
                if tone_tags:
                    tag_description += f"色调({', '.join(tone_tags)}) "
                if scene_tags:
                    tag_description += f"场景({', '.join(scene_tags)})"

                if st.button("📝 将标签添加到描述中"):
                    if user_input:
                        user_input = f"{user_input}\n\n{tag_description}"
                    else:
                        user_input = tag_description
                    st.rerun()

        with col2:
            st.markdown('<h2 class="sub-header">🖼️ 上传灵感参考图片</h2>', unsafe_allow_html=True)

            st.markdown("""
            **📸 上传你的灵感图片:**
            - 可以是你喜欢的设计作品、自然风景、艺术作品等
            - 系统会分析图片的色彩、风格和情感氛围
            - 作为配色方案生成的重要参考依据
            """)

            uploaded_image = st.file_uploader(
                "选择图片文件 (支持 PNG, JPG, JPEG):",
                type=['png', 'jpg', 'jpeg'],
                help="上传一张能体现你期望设计风格或色彩感觉的图片"
            )

            if uploaded_image:
                col2_1, col2_2 = st.columns([2, 1])
                with col2_1:
                    st.image(uploaded_image, caption="你上传的灵感参考图片", use_container_width=True)
                with col2_2:
                    st.info("""
                    **图片将用于:**
                    - 🎨 色彩分析
                    - 🎭 风格识别
                    - 💭 情感氛围理解
                    - 🔍 设计元素提取
                    """)

            # 生成按钮
            st.markdown('<h2 class="sub-header">🎨 生成配色方案</h2>', unsafe_allow_html=True)

            if st.button("🚀 生成专业配色方案", type="primary", use_container_width=True):
                if user_input and uploaded_image:
                    result = self.run_rag_generation(user_input, uploaded_image)

                    if result and result.get('success'):
                        # 保存结果到session state
                        st.session_state['rag_result'] = result
                        st.rerun()
                else:
                    st.warning("⚠️ 请输入需求并上传图片")

        # 显示结果
        if 'rag_result' in st.session_state:
            result = st.session_state['rag_result']

            st.markdown("---")
            st.markdown('<h2 class="sub-header">✨ 生成结果</h2>', unsafe_allow_html=True)

            # 结果标签页
            tab1, tab2, tab3, tab4 = st.tabs(["🎨 配色方案", "📋 方案详情", "🔍 检索知识", "📊 详细信息"])

            with tab1:
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown("### 🎯 生成的配色方案")

                # 解析多个配色方案
                schemes = self.parse_color_schemes(result['generated_solution'])

                if schemes:
                    # 为每个方案创建标签页
                    scheme_tabs = st.tabs([f"方案{i+1}: {scheme['name']}" for i, scheme in enumerate(schemes)])

                    for i, (scheme_tab, scheme) in enumerate(zip(scheme_tabs, schemes)):
                        with scheme_tab:
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.markdown(f"**设计理念：** {scheme['design_concept']}")
                                st.markdown(f"**应用建议：** {scheme['application']}")
                                st.markdown(f"**创新点：** {scheme['innovation']}")

                            with col2:
                                self.display_color_palette(scheme['colors'], f"方案{i+1}配色")
                else:
                    # 如果解析失败，显示原始文本
                    st.markdown(result['generated_solution'])

                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown("### 📋 完整方案详情")
                st.markdown(result['generated_solution'])
                st.markdown('</div>', unsafe_allow_html=True)

            with tab3:
                self.display_retrieved_knowledge(result['retrieved_documents'])

            with tab4:
                st.markdown("### 📋 处理详情")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**原始需求:**")
                    st.write(result['user_input'])

                    st.markdown("**英文需求:**")
                    st.write(result['user_query_english'])

                with col2:
                    st.markdown("**图片分析:**")
                    with st.expander("查看详细分析"):
                        st.write(result['image_analysis'])

                # 下载结果
                result_json = json.dumps(result, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📥 下载完整结果",
                    data=result_json,
                    file_name=f"color_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# 运行应用
if __name__ == "__main__":
    app = LangChainRAGWebApp()
    app.main()
