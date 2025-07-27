#!/usr/bin/env python3
"""
加载分别保存的编码器和解码器模型
"""

import torch
import joblib
import json
import numpy as np
from src.topic_model import MultiOmicsETM

def load_separate_models(model_dir='models'):
    """加载分别保存的编码器和解码器"""
    
    # 加载模型架构信息
    with open(f'{model_dir}/model_architecture.json', 'r') as f:
        arch = json.load(f)
    
    # 创建模型实例
    model = MultiOmicsETM(
        num_topics=arch['num_topics'],
        color_dim=arch['color_dim'],
        text_dim=arch['text_dim'],
        embedding_dim=arch['embedding_dim'],
        hidden_dim=arch['hidden_dim'],
        dropout=arch['dropout']
    )
    
    # 加载颜色编码器
    color_encoder_state = torch.load(f'{model_dir}/best_color_encoder.pth')
    model.color_encoder.load_state_dict(color_encoder_state['color_encoder'])
    model.color_mean.load_state_dict(color_encoder_state['color_mean'])
    model.color_logvar.load_state_dict(color_encoder_state['color_logvar'])
    
    # 加载文本编码器
    text_encoder_state = torch.load(f'{model_dir}/best_text_encoder.pth')
    model.text_encoder.load_state_dict(text_encoder_state['text_encoder'])
    model.text_mean.load_state_dict(text_encoder_state['text_mean'])
    model.text_logvar.load_state_dict(text_encoder_state['text_logvar'])
    
    # 加载解码器
    decoder_state = torch.load(f'{model_dir}/best_decoder.pth')
    model.alpha = decoder_state['alpha']
    model.rho_color = decoder_state['rho_color']
    model.rho_text = decoder_state['rho_text']
    model.lambda_color = decoder_state['lambda_color']
    model.lambda_text = decoder_state['lambda_text']
    model.score_predictor.load_state_dict(decoder_state['score_predictor'])
    
    # 加载其他必要文件
    vectorizer = joblib.load(f'{model_dir}/tfidf_vectorizer.pkl')
    vocab = json.load(open(f'{model_dir}/vocab.json', 'r'))
    theta = torch.load(f'{model_dir}/best_theta.pt')
    
    return model, vectorizer, vocab, theta

def load_traditional_model(model_dir='models'):
    """加载传统方式保存的模型"""
    
    # 加载模型架构信息
    with open(f'{model_dir}/model_architecture.json', 'r') as f:
        arch = json.load(f)
    
    # 创建模型实例
    model = MultiOmicsETM(
        num_topics=arch['num_topics'],
        color_dim=arch['color_dim'],
        text_dim=arch['text_dim'],
        embedding_dim=arch['embedding_dim'],
        hidden_dim=arch['hidden_dim'],
        dropout=arch['dropout']
    )
    
    # 加载完整模型
    model.load_state_dict(torch.load(f'{model_dir}/best_model.pth'))
    
    # 加载其他必要文件
    vectorizer = joblib.load(f'{model_dir}/tfidf_vectorizer.pkl')
    vocab = json.load(open(f'{model_dir}/vocab.json', 'r'))
    theta = torch.load(f'{model_dir}/theta.pt')
    
    return model, vectorizer, vocab, theta

def encode_text_only(model, text_vector, device='cpu'):
    """仅使用文本编码器进行编码"""
    model.eval()
    with torch.no_grad():
        text_vector = torch.FloatTensor(text_vector).unsqueeze(0).to(device)
        
        # 使用文本编码器
        mu_text, logvar_text = model.encode_text(text_vector)
        
        # 由于没有颜色信息，我们使用文本编码器的输出作为最终分布
        # 或者可以设置一个默认的颜色分布
        mu_color = torch.zeros_like(mu_text)
        logvar_color = torch.ones_like(logvar_text) * 10  # 高方差表示不确定性
        
        # 高斯乘积
        mu, logvar = model.product_of_gaussians(mu_color, logvar_color, mu_text, logvar_text)
        
        # 重参数化
        delta = model.reparameterize(mu, logvar)
        theta = model.get_theta(delta)
        
        return theta.cpu().numpy()

def encode_color_only(model, color_vector, device='cpu'):
    """仅使用颜色编码器进行编码"""
    model.eval()
    with torch.no_grad():
        color_vector = torch.FloatTensor(color_vector).unsqueeze(0).to(device)
        
        # 使用颜色编码器
        mu_color, logvar_color = model.encode_color(color_vector)
        
        # 由于没有文本信息，我们使用颜色编码器的输出作为最终分布
        # 或者可以设置一个默认的文本分布
        mu_text = torch.zeros_like(mu_color)
        logvar_text = torch.ones_like(logvar_color) * 10  # 高方差表示不确定性
        
        # 高斯乘积
        mu, logvar = model.product_of_gaussians(mu_color, logvar_color, mu_text, logvar_text)
        
        # 重参数化
        delta = model.reparameterize(mu, logvar)
        theta = model.get_theta(delta)
        
        return theta.cpu().numpy()

def decode_to_color(model, theta, device='cpu'):
    """从主题分布解码到颜色"""
    model.eval()
    with torch.no_grad():
        theta = torch.FloatTensor(theta).to(device)
        
        # 颜色解码
        color_logits = torch.matmul(theta, torch.matmul(model.alpha, model.rho_color)) + model.lambda_color
        recon_color = torch.sigmoid(color_logits)
        
        return recon_color.cpu().numpy()

def decode_to_text(model, theta, vectorizer, device='cpu'):
    """从主题分布解码到文本"""
    model.eval()
    with torch.no_grad():
        theta = torch.FloatTensor(theta).to(device)
        
        # 文本解码
        text_logits = torch.matmul(theta, torch.matmul(model.alpha, model.rho_text)) + model.lambda_text
        recon_text = torch.softmax(text_logits, dim=1)
        
        # 转换为文本（简化版本）
        feature_names = vectorizer.get_feature_names_out()
        top_indices = torch.topk(recon_text, k=10, dim=1)[1].cpu().numpy()
        
        decoded_texts = []
        for indices in top_indices:
            words = [feature_names[i] for i in indices]
            decoded_texts.append(' '.join(words))
        
        return decoded_texts

def predict_score(model, theta, device='cpu'):
    """预测评分"""
    model.eval()
    with torch.no_grad():
        theta = torch.FloatTensor(theta).to(device)
        pred_score = model.predict_score(theta)
        return pred_score.cpu().numpy()

def main():
    """示例用法"""
    print("加载模型...")
    
    try:
        # 尝试加载分别保存的模型
        model, vectorizer, vocab, theta = load_separate_models()
        print("✅ 成功加载分别保存的编码器和解码器")
        save_method = "separate"
    except FileNotFoundError:
        try:
            # 尝试加载传统方式保存的模型
            model, vectorizer, vocab, theta = load_traditional_model()
            print("✅ 成功加载传统方式保存的模型")
            save_method = "traditional"
        except FileNotFoundError:
            print("❌ 未找到模型文件，请先运行训练脚本")
            return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"模型架构: {model.num_topics} 个主题, {model.embedding_dim} 维嵌入")
    print(f"词汇表大小: {len(vocab)}")
    print(f"Theta矩阵形状: {theta.shape}")
    
    # 示例：文本到颜色的生成
    print("\n=== 文本到颜色生成示例 ===")
    
    # 准备示例文本
    sample_text = "warm sunset colors with orange and red tones"
    text_vector = vectorizer.transform([sample_text]).toarray().astype(np.float32)
    
    # 编码
    if save_method == "separate":
        theta_sample = encode_text_only(model, text_vector, device)
    else:
        # 对于传统方式，需要完整的编码过程
        with torch.no_grad():
            text_tensor = torch.FloatTensor(text_vector).to(device)
            # 创建零颜色向量
            color_tensor = torch.zeros(1, model.color_dim).to(device)
            # 完整编码
            mu_color, logvar_color = model.encode_color(color_tensor)
            mu_text, logvar_text = model.encode_text(text_tensor)
            mu, logvar = model.product_of_gaussians(mu_color, logvar_color, mu_text, logvar_text)
            delta = model.reparameterize(mu, logvar)
            theta_sample = model.get_theta(delta).cpu().numpy()
    
    # 解码到颜色
    generated_colors = decode_to_color(model, theta_sample, device)
    
    print(f"输入文本: {sample_text}")
    print(f"生成的颜色 (RGB, 0-1):")
    for i, color in enumerate(generated_colors[0].reshape(-1, 3)):
        rgb_255 = (color * 255).astype(int)
        print(f"  颜色 {i+1}: RGB({rgb_255[0]}, {rgb_255[1]}, {rgb_255[2]})")
    
    # 预测评分
    score = predict_score(model, theta_sample, device)
    print(f"预测评分: {score[0][0]:.3f}")
    
    # 示例：颜色到文本的生成
    print("\n=== 颜色到文本生成示例 ===")
    
    # 准备示例颜色
    sample_colors = np.array([0.8, 0.6, 0.4, 0.7, 0.5, 0.3, 0.9, 0.4, 0.2, 0.6, 0.3, 0.1, 0.5, 0.2, 0.0])
    
    # 编码
    if save_method == "separate":
        theta_sample = encode_color_only(model, sample_colors, device)
    else:
        # 对于传统方式，需要完整的编码过程
        with torch.no_grad():
            color_tensor = torch.FloatTensor(sample_colors).unsqueeze(0).to(device)
            # 创建零文本向量
            text_tensor = torch.zeros(1, model.text_dim).to(device)
            # 完整编码
            mu_color, logvar_color = model.encode_color(color_tensor)
            mu_text, logvar_text = model.encode_text(text_tensor)
            mu, logvar = model.product_of_gaussians(mu_color, logvar_color, mu_text, logvar_text)
            delta = model.reparameterize(mu, logvar)
            theta_sample = model.get_theta(delta).cpu().numpy()
    
    # 解码到文本
    generated_texts = decode_to_text(model, theta_sample, vectorizer, device)
    
    print(f"输入颜色: {sample_colors}")
    print(f"生成的文本描述: {generated_texts[0]}")
    
    # 预测评分
    score = predict_score(model, theta_sample, device)
    print(f"预测评分: {score[0][0]:.3f}")

if __name__ == "__main__":
    main() 