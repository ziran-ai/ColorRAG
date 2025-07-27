import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import gradio as gr
import json
import requests
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from src.topic_model import MultiOmicsETM

# 加载环境变量
load_dotenv()

# 配置DeepSeek API
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    print("警告: 未找到DEEPSEEK_API_KEY环境变量，多模态理解模块将无法工作")

# DeepSeek API 配置
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"  # 根据实际提供的模型名称调整

# 调用DeepSeek API的辅助函数
def call_deepseek_api(prompt, images=None):
    """
    调用DeepSeek API进行文本或多模态生成
    
    Args:
        prompt: 提示词
        images: 图片列表，每个元素是一个dict，包含url或base64
        
    Returns:
        str: 生成的文本
    """
    if not DEEPSEEK_API_KEY:
        return "未配置DeepSeek API密钥，无法进行文本生成"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    # 构建消息
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    # 如果有图片，添加到消息中
    if images:
        for img_data in images:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": img_data
            })
    
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # 如果请求失败，抛出异常
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    except Exception as e:
        return f"调用DeepSeek API时出错: {str(e)}"

# 加载训练好的模型和相关数据
def load_model_and_data():
    """加载模型和相关数据"""
    print("加载模型和数据...")
    
    # 加载TF-IDF向量器
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"找不到向量器文件: {vectorizer_path}")
    
    vectorizer = joblib.load(vectorizer_path)
    
    # 加载训练好的模型
    model_path = 'models/moetm_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    # 加载模型配置
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['config']
    
    # 创建模型实例
    model = MultiOmicsETM(
        num_topics=model_config['num_topics'],
        color_dim=model_config['color_dim'],
        text_dim=model_config['text_dim'],
        embedding_dim=model_config['embedding_dim'],
        hidden_dim=model_config['hidden_dim'],
        dropout=model_config['dropout']
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    
    # 加载嵌入矩阵
    alpha = torch.load('models/alpha.pt', map_location='cpu')
    rho_color = torch.load('models/rho_color.pt', map_location='cpu')
    rho_text = torch.load('models/rho_text.pt', map_location='cpu')
    
    # 加载原始数据集，用于检索
    data_path = 'data/palettes_with_descriptions.xlsx'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件: {data_path}")
    
    df = pd.read_excel(data_path)
    
    # 处理颜色数据
    color_columns = []
    for i in range(1, 6):
        for c in ['R', 'G', 'B']:
            color_columns.append(f'Color_{i}_{c}')
    
    # 提取颜色数据并归一化到[0, 1]
    color_data = df[color_columns].values.astype(np.float32)
    color_data = np.clip(color_data, 0, 1)  # 数据已经在0-1区间内
    
    # 处理文本数据
    text_vectors = vectorizer.transform(df['description']).toarray().astype(np.float32)
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'df': df,
        'color_data': color_data,
        'text_vectors': text_vectors,
        'alpha': alpha,
        'rho_color': rho_color,
        'rho_text': rho_text
    }

# 模块一：多模态输入理解
def process_multimodal_input(text_input, image):
    """
    处理用户的文本和图像输入，融合为一段详细的描述
    
    Args:
        text_input: 用户输入的文本
        image: 用户上传的图片
    
    Returns:
        str: 融合后的详细描述文本
    """
    if not DEEPSEEK_API_KEY:
        return f"无法处理多模态输入: 未配置API密钥。将只使用文本输入: {text_input}"
    
    try:
        # 准备图片
        if image is not None:
            import base64
            # 将图片转换为base64
            image_base64 = base64.b64encode(image).decode('utf-8')
            image_data = {"data": image_base64, "mime_type": "image/jpeg"}
            
            # 创建提示词
            prompt = f"""
            任务：分析图片和文本，生成详细的设计方案描述。
            
            第一步：详细描述这张图片的内容、风格、关键元素和整体氛围。
            
            第二步：结合上述图片描述和以下用户需求：'{text_input}'，生成一段综合性的、详细的设计方案描述，包含场景、情感、风格和关键对象。
            
            注意：
            1. 描述应该详细具体，使用丰富的形容词
            2. 描述应该包含色彩相关的词语和氛围
            3. 输出应该只包含最终的描述文本，不要包含其他解释
            """
            
            # 调用API生成描述
            detailed_description = call_deepseek_api(prompt, [image_data])
        else:
            # 如果没有图片，只处理文本
            prompt = f"""
            任务：基于用户的简短描述，生成一个详细的设计方案描述。
            
            用户需求：'{text_input}'
            
            请生成一段综合性的、详细的设计方案描述，包含场景、情感、风格和色彩氛围。
            
            注意：
            1. 描述应该详细具体，使用丰富的形容词
            2. 描述应该包含色彩相关的词语和氛围
            3. 输出应该只包含最终的描述文本，不要包含其他解释
            """
            
            # 调用API生成描述
            detailed_description = call_deepseek_api(prompt)
        
        return detailed_description
    
    except Exception as e:
        return f"处理多模态输入时出错: {str(e)}。将只使用文本输入: {text_input}"

def preprocess_text_for_model(detailed_description, vectorizer):
    """
    将详细描述文本处理成模型需要的输入格式
    
    Args:
        detailed_description: 详细的描述文本
        vectorizer: 训练好的TF-IDF向量器
    
    Returns:
        numpy.ndarray: 文本的TF-IDF向量
    """
    # 使用与训练时相同的向量器转换文本
    text_vector = vectorizer.transform([detailed_description]).toarray().astype(np.float32)
    return text_vector[0]  # 返回一维向量

# 模块二：moETM跨模态推理和RAG检索
def infer_and_retrieve(text_vector, model_data, top_k=10):
    """
    根据文本向量进行跨模态推理和检索
    
    Args:
        text_vector: 处理后的文本向量
        model_data: 包含模型和数据的字典
        top_k: 检索的候选数量
    
    Returns:
        tuple: (最佳方案索引, 综合得分, 相似度列表)
    """
    model = model_data['model']
    
    # 创建颜色占位符（全零向量）
    color_placeholder = torch.zeros((1, model.color_dim), dtype=torch.float32)
    
    # 转换文本向量为张量
    text_tensor = torch.tensor(text_vector, dtype=torch.float32).unsqueeze(0)
    
    # 使用模型的文本编码器
    with torch.no_grad():
        # 获取文本编码
        mu_text, logvar_text = model.encode_text(text_tensor)
        
        # 由于颜色模态缺失，将颜色部分设为低置信度的均值0
        mu_color = torch.zeros_like(mu_text)
        logvar_color = torch.ones_like(logvar_text) * 10  # 高方差（低置信度）
        
        # 合并两个模态的分布
        mu, logvar = model.product_of_gaussians(mu_color, logvar_color, mu_text, logvar_text)
        
        # 不需要随机采样，直接使用均值
        delta = mu
        
        # 获取主题比例
        theta = model.get_theta(delta)
        
        # 解码得到重构的颜色和文本表示
        recon_color, recon_text = model.decode(theta)
    
    # 步骤a: 基于文本的初步检索
    text_similarities = cosine_similarity(recon_text.detach().numpy(), model_data['text_vectors'])
    text_similarities = text_similarities[0]  # 降维到一维
    
    # 获取Top-K候选方案的索引
    top_indices = np.argsort(text_similarities)[-top_k:][::-1]
    
    # 步骤b: 基于颜色的重排序
    retrieved_colors = model_data['color_data'][top_indices]
    color_similarities = cosine_similarity(recon_color.detach().numpy(), retrieved_colors)
    color_similarities = color_similarities[0]  # 降维到一维
    
    # 步骤c: 综合评分与最终选择
    w_text, w_color = 0.6, 0.4  # 文本和颜色的权重
    combined_scores = w_text * text_similarities[top_indices] + w_color * color_similarities
    
    # 找出最佳方案
    best_idx = np.argmax(combined_scores)
    best_global_idx = top_indices[best_idx]
    best_score = combined_scores[best_idx]
    
    return best_global_idx, best_score, {
        'top_indices': top_indices,
        'text_similarities': text_similarities[top_indices],
        'color_similarities': color_similarities,
        'combined_scores': combined_scores
    }

# 模块三：结果生成
def generate_result_content(best_scheme, model_data):
    """
    根据最佳方案生成结果内容
    
    Args:
        best_scheme: 最佳方案的数据
        model_data: 包含模型和数据的字典
    
    Returns:
        str: 生成的结果内容
    """
    if not DEEPSEEK_API_KEY:
        # 如果没有API密钥，生成简单的描述
        colors_text = ""
        for i in range(1, 6):
            r = int(best_scheme[f'Color_{i}_R'] * 255)
            g = int(best_scheme[f'Color_{i}_G'] * 255)
            b = int(best_scheme[f'Color_{i}_B'] * 255)
            colors_text += f"颜色{i}: RGB({r}, {g}, {b})\n"
        
        return f"""
        设计方案描述：{best_scheme['description']}
        
        配色方案：
        {colors_text}
        """
    
    try:
        # 提取颜色信息
        colors = []
        for i in range(1, 6):
            r = int(best_scheme[f'Color_{i}_R'] * 255)
            g = int(best_scheme[f'Color_{i}_G'] * 255)
            b = int(best_scheme[f'Color_{i}_B'] * 255)
            colors.append(f"RGB({r}, {g}, {b})")
        
        # 创建提示词
        prompt = f"""
        你是一位专业的设计方案解说员。请根据以下核心描述：
        
        '{best_scheme['description']}'
        
        和这套配色方案的五个核心颜色：
        1. {colors[0]}
        2. {colors[1]}
        3. {colors[2]}
        4. {colors[3]}
        5. {colors[4]}
        
        生成一段优美的介绍文字，包含：
        1. 这套配色方案的视觉效果和情感氛围
        2. 适合应用的场景和领域
        3. 配色方案中颜色之间的和谐关系
        
        最后，清晰地列出这五个颜色的RGB数值。
        
        使用中文回答，注重专业性和文学性。
        """
        
        # 调用API生成内容
        generated_content = call_deepseek_api(prompt)
        
        return generated_content
    
    except Exception as e:
        # 出错时返回简单描述
        colors_text = ""
        for i in range(1, 6):
            r = int(best_scheme[f'Color_{i}_R'] * 255)
            g = int(best_scheme[f'Color_{i}_G'] * 255)
            b = int(best_scheme[f'Color_{i}_B'] * 255)
            colors_text += f"颜色{i}: RGB({r}, {g}, {b})\n"
        
        return f"""
        生成结果内容时出错: {str(e)}
        
        设计方案描述：{best_scheme['description']}
        
        配色方案：
        {colors_text}
        """

# 主应用函数
def color_generator_app(text_input, image=None):
    """
    颜色生成应用的主函数
    
    Args:
        text_input: 用户输入的文本
        image: 用户上传的图片
    
    Returns:
        str: 生成的结果
    """
    # 加载模型和数据
    try:
        model_data = load_model_and_data()
    except Exception as e:
        return f"加载模型和数据时出错: {str(e)}"
    
    # 步骤1: 多模态输入理解
    detailed_description = process_multimodal_input(text_input, image)
    
    # 步骤2: 文本预处理
    text_vector = preprocess_text_for_model(detailed_description, model_data['vectorizer'])
    
    # 步骤3: 跨模态推理和RAG检索
    best_idx, best_score, similarity_info = infer_and_retrieve(text_vector, model_data)
    
    # 获取最佳方案
    best_scheme = model_data['df'].iloc[best_idx]
    
    # 步骤4: 生成结果内容
    result_content = generate_result_content(best_scheme, model_data)
    
    # 创建详细的输出报告
    output = f"""
    【用户输入】
    文本: {text_input}
    图片: {"已上传" if image is not None else "无"}
    
    【详细描述】
    {detailed_description}
    
    【检索结果】
    最佳方案ID: {best_idx}
    综合评分: {best_score:.4f}
    
    【生成内容】
    {result_content}
    """
    
    return output

# 使用Gradio创建Web界面
def create_web_ui():
    """创建Gradio Web界面"""
    with gr.Blocks() as app:
        gr.Markdown("# 智能配色方案生成器")
        gr.Markdown("上传一张图片和输入需求描述，AI将为您生成最佳配色方案")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="输入您的需求描述", 
                    placeholder="例如：我需要一个适合儿童玩具的明亮、活泼的配色方案"
                )
                image_input = gr.Image(label="上传参考图片（可选）")
                submit_btn = gr.Button("生成配色方案")
            
            with gr.Column():
                output = gr.Textbox(label="生成结果", lines=20)
        
        submit_btn.click(color_generator_app, inputs=[text_input, image_input], outputs=output)
    
    return app

# 主函数
if __name__ == "__main__":
    app = create_web_ui()
    app.launch() 