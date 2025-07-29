#!/usr/bin/env python3
"""
RAG系统Gradio可视化界面
简化版本，易于部署
"""

import gradio as gr
import sys
import os
import json
from datetime import datetime
import numpy as np
from PIL import Image

# 添加路径并导入RAG系统
sys.path.append('..')
sys.path.append('./tradition')

try:
    from tradition.topic_rag_system import TopicRAGSystem
    print("✅ 成功导入RAG系统模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    raise e

class RAGGradioApp:
    def __init__(self):
        self.system = None
        self.load_system()
    
    def load_system(self):
        """加载RAG系统"""
        try:
            api_key = "sk-3c4ba59c8b094106995821395c7bc60e"
            self.system = TopicRAGSystem(device='cpu', api_key=api_key)
            print("✅ RAG系统加载成功")
        except Exception as e:
            print(f"❌ RAG系统加载失败: {e}")
    
    def generate_design(self, user_input, image, top_k):
        """生成设计方案"""
        if not self.system:
            return "❌ 系统未加载，请检查配置", None, None
        
        if not user_input.strip():
            return "❌ 请输入设计需求", None, None
        
        try:
            # 处理图片
            image_path = "temp_image.jpg"
            if image is not None:
                image.save(image_path)
            else:
                image_path = "test_image.jpg"
            
            # 生成设计方案
            result = self.system.run_full_pipeline(user_input, image_path, top_k)
            
            # 提取生成方案
            generated_plan = result.get('new_plan', '生成失败')
            
            # 提取候选方案信息
            candidates_info = ""
            if 'candidates' in result:
                candidates_info = "📋 参考候选方案:\n\n"
                for i, candidate in enumerate(result['candidates'][:3], 1):
                    candidates_info += f"**候选方案 {i}** (得分: {candidate.get('combined_score', 0):.3f})\n"
                    candidates_info += f"描述: {candidate['description'][:100]}...\n\n"
            
            # 生成统计信息
            stats_info = f"""
📊 生成统计:
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 检索候选数: {top_k}
- 候选方案数: {len(result.get('candidates', []))}
- 生成方案长度: {len(generated_plan)} 字符
"""
            
            return generated_plan, candidates_info, stats_info
            
        except Exception as e:
            return f"❌ 生成失败: {str(e)}", None, None
    
    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="Topic-RAG设计助手", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎨 Topic-RAG设计助手")
            gr.Markdown("基于多模态主题模型的智能设计配色方案生成系统")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # 输入区域
                    user_input = gr.Textbox(
                        label="设计需求描述",
                        placeholder="例如：我想要一个现代简约风格的配色方案，适合办公环境，需要体现专业性和创新性...",
                        lines=4
                    )
                    
                    image_input = gr.Image(
                        label="上传参考图片（可选）",
                        type="pil"
                    )
                    
                    top_k = gr.Slider(
                        minimum=3,
                        maximum=10,
                        value=5,
                        step=1,
                        label="检索候选数量"
                    )
                    
                    generate_btn = gr.Button("🚀 生成设计方案", variant="primary")
                
                with gr.Column(scale=3):
                    # 输出区域
                    output_plan = gr.Markdown(
                        label="生成的设计方案",
                        value="请输入设计需求并点击生成按钮"
                    )
                    
                    candidates_output = gr.Markdown(
                        label="参考候选方案",
                        value=""
                    )
                    
                    stats_output = gr.Markdown(
                        label="生成统计",
                        value=""
                    )
            
            # 示例区域
            with gr.Accordion("💡 使用示例", open=False):
                gr.Markdown("""
### 示例输入：
1. **现代办公风格**: "我想要一个现代简约风格的配色方案，适合办公环境，需要体现专业性和创新性"
2. **古典奢华风格**: "我想要一个古典奢华的配色方案，适合高端酒店大堂，需要体现尊贵和优雅"
3. **温馨家居风格**: "我想要一个温暖舒适的配色方案，适合家居客厅，需要营造温馨的家庭氛围"
4. **创意工作室风格**: "我想要一个充满活力的配色方案，适合创意工作室，需要激发创造力和灵感"

### 使用技巧：
- 详细描述设计风格和适用场景
- 说明情感需求和氛围要求
- 可以上传参考图片获得更好的建议
- 调整检索候选数量以平衡质量和速度
                """)
            
            # 系统信息
            with gr.Accordion("ℹ️ 系统信息", open=False):
                gr.Markdown("""
### 系统特点：
- **多模态融合**: 结合文本和图像信息
- **智能检索**: 基于Topic Model的相似度检索
- **个性化生成**: 根据用户需求生成独特方案
- **实时反馈**: 即时生成和显示结果

### 技术架构：
- Topic Model: 多模态主题模型
- 检索系统: 双重相似度计算
- 生成系统: DeepSeek LLM
- 界面框架: Gradio
                """)
            
            # 绑定事件
            generate_btn.click(
                fn=self.generate_design,
                inputs=[user_input, image_input, top_k],
                outputs=[output_plan, candidates_output, stats_output]
            )
            
            # 回车键触发生成
            user_input.submit(
                fn=self.generate_design,
                inputs=[user_input, image_input, top_k],
                outputs=[output_plan, candidates_output, stats_output]
            )
        
        return interface

def main():
    """主函数"""
    app = RAGGradioApp()
    interface = app.create_interface()
    
    # 启动应用
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )

if __name__ == "__main__":
    main() 