import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import json
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from src.topic_model import MultiOmicsETM

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练多模态主题模型')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='palettes_descriptions.xlsx',
                        help='带有描述的调色板数据路径')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='测试集比例')
    
    # 模型参数
    parser.add_argument('--num_topics', type=int, default=50,
                        help='主题数量 K')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='嵌入空间维度 L')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='编码器隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout比率')
    parser.add_argument('--max_features', type=int, default=2000,
                        help='TF-IDF最大特征数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--kl_weight', type=float, default=0.01,
                        help='KL散度损失权重')
    parser.add_argument('--score_weight', type=float, default=1.0,
                        help='评分预测损失权重')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='早停轮数')
    parser.add_argument('--lr_scheduler_patience', type=int, default=5,
                        help='学习率调度器的耐心值')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
                        help='学习率调度器的缩减因子')
    
    # 输出参数
    parser.add_argument('--save_dir', type=str, default='models',
                        help='模型保存目录')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='输出保存目录')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志记录间隔（批次）')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='评估间隔（轮数）')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def preprocess_data(data_path, max_features=2000, test_size=0.1, seed=42):
    """
    预处理数据，包括读取数据、处理颜色和文本数据、划分训练集和测试集
    
    Args:
        data_path: 数据文件路径
        max_features: TF-IDF最大特征数
        test_size: 测试集比例
        seed: 随机种子
        
    Returns:
        dict: 包含处理后的数据和相关信息
    """
    print(f"读取数据: {data_path}")
    
    # 读取数据
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("不支持的文件格式，请提供.xlsx或.csv文件")
    
    print(f"数据集大小: {len(df)}")
    print(f"数据列: {list(df.columns)}")
    
    # 处理颜色数据 - 查找颜色列
    color_columns = []
    for i in range(1, 6):
        for c in ['R', 'G', 'B']:
            col_name = f'Color_{i}_{c}'
            if col_name in df.columns:
                color_columns.append(col_name)
    
    # 如果找不到标准格式的颜色列，尝试其他可能的格式
    if not color_columns:
        # 查找包含RGB的列
        for col in df.columns:
            if 'R' in col and 'G' in col and 'B' in col:
                color_columns.append(col)
    
    if not color_columns:
        raise ValueError("找不到颜色列，请检查数据格式")
    
    print(f"找到颜色列: {color_columns}")
    
    # 提取颜色数据并归一化到[0, 1]
    color_data = df[color_columns].values.astype(np.float32)
    color_data = np.clip(color_data, 0, 1)  # 数据已经在0-1区间内
    
    # 处理文本数据
    text_column = None
    for col in ['description', 'Description', 'text', 'Text', 'content', 'Content']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        raise ValueError("找不到文本描述列，请检查数据格式")
    
    print(f"使用文本列: {text_column}")
    
    # 使用TF-IDF向量化文本
    print(f"使用TF-IDF向量化文本 (max_features={max_features})")
    vectorizer = TfidfVectorizer(
        max_df=0.9, 
        min_df=5, 
        max_features=max_features,
        stop_words='english'  # 移除英文停用词
    )
    text_vectors = vectorizer.fit_transform(df[text_column].fillna('')).toarray().astype(np.float32)
    
    # 保存vectorizer，后续推理时需要
    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
    # 处理评分数据（如果有）
    score_column = None
    for col in ['Targets', 'targets', 'score', 'Score', 'rating', 'Rating']:
        if col in df.columns:
            score_column = col
            break
    
    if score_column:
        print(f"使用评分列: {score_column}")
        scores = df[score_column].values.astype(np.float32).reshape(-1, 1)
        # 归一化评分到[0, 1]
        score_min, score_max = scores.min(), scores.max()
        scores = (scores - score_min) / (score_max - score_min)
    else:
        print("警告: 数据中缺少评分列，将使用随机评分")
        scores = np.random.rand(len(df), 1).astype(np.float32)
    
    # 划分训练集和测试集
    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=seed
    )
    
    # 创建训练集和测试集
    train_color = color_data[train_indices]
    train_text = text_vectors[train_indices]
    train_scores = scores[train_indices]
    
    test_color = color_data[test_indices]
    test_text = text_vectors[test_indices]
    test_scores = scores[test_indices]
    
    # 创建AnnData对象用于评估
    adata = ad.AnnData(X=np.concatenate([color_data, text_vectors], axis=1))
    adata.obs['cell_types'] = df[text_column].str[:20].fillna('unknown').values
    adata.obs['cell_types'] = adata.obs['cell_types'].astype('category')
    if score_column:
        adata.obs['targets'] = df[score_column].values
    
    # 保存词汇表
    vocab = vectorizer.vocabulary_
    with open('models/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"颜色数据维度: {color_data.shape}")
    print(f"文本数据维度: {text_vectors.shape}")
    print(f"词汇表大小: {len(vocab)}")
    
    # 返回处理后的数据
    return {
        'train_color': train_color,
        'train_text': train_text,
        'train_scores': train_scores,
        'test_color': test_color,
        'test_text': test_text,
        'test_scores': test_scores,
        'color_dim': color_data.shape[1],
        'text_dim': text_vectors.shape[1],
        'vectorizer': vectorizer,
        'adata': adata,
        'df': df,
        'vocab': vocab
    }

def compute_loss(model_output, targets, kl_weight=0.1, score_weight=0.5):
    """
    计算总损失
    
    Args:
        model_output: 模型输出的元组 (recon_color, recon_text, pred_score, theta, mu, logvar)
        targets: 目标值的元组 (color_data, text_data, scores)
        kl_weight: KL散度损失权重
        score_weight: 评分预测损失权重
        
    Returns:
        dict: 包含总损失和各部分损失
    """
    recon_color, recon_text, pred_score, _, mu, logvar = model_output
    color_data, text_data, scores = targets
    
    # 计算颜色重构损失 (MSE)
    color_loss = nn.functional.mse_loss(recon_color, color_data)
    
    # 计算文本重构损失 (交叉熵)
    # 确保文本数据是归一化的
    text_data_norm = text_data / (text_data.sum(dim=1, keepdim=True) + 1e-10)
    # 使用交叉熵损失
    text_loss = -torch.mean(torch.sum(text_data_norm * torch.log(recon_text + 1e-10), dim=1))
    
    # 计算KL散度损失 (VAE正则项)
    kl_loss = 0.5 * torch.mean(torch.sum(mu.pow(2) + torch.exp(logvar) - logvar - 1, dim=1))
    
    # 计算评分预测损失 (MSE)
    score_loss = nn.functional.mse_loss(pred_score, scores)
    
    # 计算总损失
    total_loss = color_loss + text_loss + kl_weight * kl_loss + score_weight * score_loss
    
    return {
        'total': total_loss,
        'color': color_loss.item(),
        'text': text_loss.item(),
        'kl': kl_loss.item(),
        'score': score_loss.item()
    }

def train_epoch(model, train_loader, optimizer, device, kl_weight, score_weight):
    """训练一个轮次"""
    model.train()
    total_loss = 0
    color_loss_sum = 0
    text_loss_sum = 0
    kl_loss_sum = 0
    score_loss_sum = 0
    
    for batch_idx, (color_data, text_data, scores) in enumerate(train_loader):
        # 将数据移到设备上
        color_data = color_data.to(device)
        text_data = text_data.to(device)
        scores = scores.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        model_output = model(color_data, text_data)
        
        # 计算损失
        loss_dict = compute_loss(
            model_output, (color_data, text_data, scores),
            kl_weight=kl_weight, score_weight=score_weight
        )
        
        # 反向传播
        loss_dict['total'].backward()
        optimizer.step()
        
        # 累加损失
        total_loss += loss_dict['total'].item()
        color_loss_sum += loss_dict['color']
        text_loss_sum += loss_dict['text']
        kl_loss_sum += loss_dict['kl']
        score_loss_sum += loss_dict['score']
    
    # 计算平均损失
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_color_loss = color_loss_sum / num_batches
    avg_text_loss = text_loss_sum / num_batches
    avg_kl_loss = kl_loss_sum / num_batches
    avg_score_loss = score_loss_sum / num_batches
    
    return {
        'total': avg_loss,
        'color': avg_color_loss,
        'text': avg_text_loss,
        'kl': avg_kl_loss,
        'score': avg_score_loss
    }

def evaluate_model(model, test_loader, device, kl_weight, score_weight):
    """评估模型"""
    model.eval()
    total_loss = 0
    color_loss_sum = 0
    text_loss_sum = 0
    kl_loss_sum = 0
    score_loss_sum = 0
    
    with torch.no_grad():
        for color_data, text_data, scores in test_loader:
            # 将数据移到设备上
            color_data = color_data.to(device)
            text_data = text_data.to(device)
            scores = scores.to(device)
            
            # 前向传播
            model_output = model(color_data, text_data)
            
            # 计算损失
            loss_dict = compute_loss(
                model_output, (color_data, text_data, scores),
                kl_weight=kl_weight, score_weight=score_weight
            )
            
            # 累加损失
            total_loss += loss_dict['total'].item()
            color_loss_sum += loss_dict['color']
            text_loss_sum += loss_dict['text']
            kl_loss_sum += loss_dict['kl']
            score_loss_sum += loss_dict['score']
    
    # 计算平均损失
    num_batches = len(test_loader)
    avg_loss = total_loss / num_batches
    avg_color_loss = color_loss_sum / num_batches
    avg_text_loss = text_loss_sum / num_batches
    avg_kl_loss = kl_loss_sum / num_batches
    avg_score_loss = score_loss_sum / num_batches
    
    return {
        'total': avg_loss,
        'color': avg_color_loss,
        'text': avg_text_loss,
        'kl': avg_kl_loss,
        'score': avg_score_loss
    }

def extract_embeddings(model, data_loader, device):
    """提取嵌入向量"""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for color_data, text_data, _ in data_loader:
            color_data = color_data.to(device)
            text_data = text_data.to(device)
            
            # 获取主题分布
            mu_color, logvar_color = model.encode_color(color_data)
            mu_text, logvar_text = model.encode_text(text_data)
            mu, logvar = model.product_of_gaussians(mu_color, logvar_color, mu_text, logvar_text)
            delta = model.reparameterize(mu, logvar)
            theta = model.get_theta(delta)
            
            embeddings.append(theta.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)

def calculate_clustering_metrics(embeddings, labels, n_clusters=10):
    """计算聚类指标"""
    # 使用K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # 计算ARI
    ari = adjusted_rand_score(labels, cluster_labels)
    
    # 计算NMI
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    
    # 计算ASW (Silhouette Score)
    asw = silhouette_score(embeddings, cluster_labels)
    
    return {
        'ARI': ari,
        'NMI': nmi,
        'ASW': asw
    }

def calculate_topic_coherence(topic_word_matrix, top_k=10):
    """计算主题一致性"""
    coherence_scores = []
    
    for topic_idx in range(topic_word_matrix.shape[0]):
        # 获取前top_k个词
        top_indices = np.argsort(topic_word_matrix[topic_idx])[-top_k:]
        top_weights = topic_word_matrix[topic_idx][top_indices]
        
        # 计算一致性分数（基于权重分布）
        coherence = np.mean(top_weights)
        coherence_scores.append(coherence)
    
    return np.mean(coherence_scores)

def calculate_topic_sparsity(topic_matrix, threshold=0.01):
    """计算主题稀疏性"""
    # 计算非零元素的比例
    sparsity = 1.0 - (np.count_nonzero(topic_matrix > threshold) / topic_matrix.size)
    return sparsity

def calculate_topic_specificity(topic_matrix):
    """计算主题特异性"""
    # 计算每个主题的熵
    entropy = -np.sum(topic_matrix * np.log(topic_matrix + 1e-10), axis=1)
    # 特异性是熵的倒数
    specificity = 1.0 / (entropy + 1e-10)
    return np.mean(specificity)

def calculate_cross_modal_alignment(topic_color_matrix, topic_text_matrix):
    """计算跨模态对齐"""
    # 计算两个矩阵之间的相关性
    correlation = np.corrcoef(topic_color_matrix.flatten(), topic_text_matrix.flatten())[0, 1]
    return correlation if not np.isnan(correlation) else 0.0

def evaluate_topics(model, adata, epoch, output_dir):
    """评估主题质量"""
    print(f"评估第 {epoch} 轮的主题质量...")
    
    # 获取主题-词矩阵和主题-颜色矩阵
    topic_word_matrix = model.get_topic_word_matrix()
    topic_color_matrix = model.get_topic_color_matrix()
    
    # 计算各种指标
    metrics = {}
    
    # 主题一致性
    metrics['Topic_Coherence_Mod1'] = calculate_topic_coherence(topic_color_matrix)
    metrics['Topic_Coherence_Mod2'] = calculate_topic_coherence(topic_word_matrix)
    
    # 主题稀疏性
    metrics['Topic_Sparsity_Mod1'] = calculate_topic_sparsity(topic_color_matrix)
    metrics['Topic_Sparsity_Mod2'] = calculate_topic_sparsity(topic_word_matrix)
    
    # 主题特异性
    metrics['Topic_Specificity_Mod1'] = calculate_topic_specificity(topic_color_matrix)
    metrics['Topic_Specificity_Mod2'] = calculate_topic_specificity(topic_word_matrix)
    
    # 跨模态对齐
    metrics['Cross_Modal_Alignment'] = calculate_cross_modal_alignment(topic_color_matrix, topic_word_matrix)
    
    # 主题离散性（基于主题分布的方差）
    topic_distributions = topic_word_matrix / (topic_word_matrix.sum(axis=1, keepdims=True) + 1e-10)
    topic_variance = np.var(topic_distributions, axis=1)
    metrics['Topic_Discreteness'] = np.mean(topic_variance)
    
    # 聚类指标（如果有标签）
    if 'cell_types' in adata.obs:
        # 提取嵌入向量
        embeddings = adata.obsm.get('delta', None)
        if embeddings is not None:
            labels = adata.obs['cell_types'].cat.codes
            clustering_metrics = calculate_clustering_metrics(embeddings, labels)
            metrics.update(clustering_metrics)
    
    # 其他指标（简化版本）
    metrics['B_kBET'] = 0.0  # 需要批次信息
    metrics['B_ASW'] = 0.0   # 需要批次信息
    metrics['B_GC'] = 0.0    # 需要批次信息
    metrics['B_ebm'] = 0.0   # 需要批次信息
    metrics['ASW_2'] = metrics.get('ASW', 0.0)  # 使用ASW作为ASW_2
    
    print(f"主题评估指标: {metrics}")
    
    return metrics

def plot_loss_curves(train_losses, val_losses, output_dir):
    """绘制损失曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('训练损失曲线', fontsize=16)
    
    loss_types = ['total', 'color', 'text', 'kl', 'score']
    titles = ['总损失', '颜色损失', '文本损失', 'KL损失', '评分损失']
    
    for i, (loss_type, title) in enumerate(zip(loss_types, titles)):
        row = i // 3
        col = i % 3
        
        axes[row, col].plot(train_losses[loss_type], label='训练', color='blue')
        axes[row, col].plot(val_losses[loss_type], label='验证', color='red')
        axes[row, col].set_title(title)
        axes[row, col].set_xlabel('轮次')
        axes[row, col].set_ylabel('损失')
        axes[row, col].legend()
        axes[row, col].grid(True)
    
    # 隐藏多余的子图
    axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("开始数据预处理...")
    # 预处理数据
    data = preprocess_data(
        args.data_path,
        max_features=args.max_features,
        test_size=args.test_size,
        seed=args.seed
    )
    
    print("创建数据加载器...")
    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(data['train_color']),
        torch.FloatTensor(data['train_text']),
        torch.FloatTensor(data['train_scores'])
    )
    
    test_dataset = TensorDataset(
        torch.FloatTensor(data['test_color']),
        torch.FloatTensor(data['test_text']),
        torch.FloatTensor(data['test_scores'])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print("创建模型...")
    # 创建模型
    model = MultiOmicsETM(
        num_topics=args.num_topics,
        color_dim=data['color_dim'],
        text_dim=data['text_dim'],
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    
    # 将模型移到设备上
    device = torch.device(args.device)
    model = model.to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"使用设备: {device}")
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=args.lr_scheduler_factor, 
        patience=args.lr_scheduler_patience, 
        verbose=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = {'total': [], 'color': [], 'text': [], 'kl': [], 'score': []}
    val_losses = {'total': [], 'color': [], 'text': [], 'kl': [], 'score': []}
    topic_metrics_history = []
    
    # 将参数保存为JSON
    with open(os.path.join(args.output_dir, 'params.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print("开始训练...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练一个轮次
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            kl_weight=args.kl_weight, score_weight=args.score_weight
        )
        
        # 评估模型
        val_loss = evaluate_model(
            model, test_loader, device,
            kl_weight=args.kl_weight, score_weight=args.score_weight
        )
        
        # 记录损失
        for key in train_loss:
            train_losses[key].append(train_loss[key])
            val_losses[key].append(val_loss[key])
        
        # 使用学习率调度器
        scheduler.step(val_loss['total'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印进度
        epoch_time = time.time() - start_time
        print(f"轮次 {epoch+1}/{args.epochs} - 时间: {epoch_time:.2f}s - LR: {current_lr:.6f}")
        print(f"  训练损失: {train_loss['total']:.4f} - 验证损失: {val_loss['total']:.4f}")
        print(f"  颜色损失: {train_loss['color']:.4f} - 文本损失: {train_loss['text']:.4f}")
        print(f"  KL损失: {train_loss['kl']:.4f} - 评分损失: {train_loss['score']:.4f}")
        
        # 评估主题质量
        if (epoch + 1) % args.eval_interval == 0:
            # 提取嵌入向量
            train_theta = extract_embeddings(model, train_loader, device)
            
            # 将嵌入向量添加到AnnData对象中
            adata_subset = data['adata'].copy()
            adata_subset.obsm['delta'] = train_theta
            
            # 评估主题质量
            topic_metrics = evaluate_topics(model, adata_subset, epoch + 1, args.output_dir)
            topic_metrics_history.append(topic_metrics)
            
            # 保存主题评估结果
            with open(os.path.join(args.output_dir, f'topic_metrics_epoch_{epoch+1}.json'), 'w') as f:
                json.dump(topic_metrics, f, indent=4)
        
        # 早停
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            patience_counter = 0
            
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            
            # 保存重要矩阵
            # 创建matrices文件夹并保存关键矩阵
            matrices_dir = os.path.join(args.save_dir, 'matrices')
            os.makedirs(matrices_dir, exist_ok=True)
            
            torch.save(model.alpha.detach().cpu(), os.path.join(matrices_dir, 'alpha.pt'))
            torch.save(model.rho_color.detach().cpu(), os.path.join(matrices_dir, 'rho_color.pt'))
            torch.save(model.rho_text.detach().cpu(), os.path.join(matrices_dir, 'rho_text.pt'))
            torch.save(model.lambda_color.detach().cpu(), os.path.join(matrices_dir, 'lambda_color.pt'))
            torch.save(model.lambda_text.detach().cpu(), os.path.join(matrices_dir, 'lambda_text.pt'))
            
            # 保存theta矩阵（用于后续生成）
            train_theta = extract_embeddings(model, train_loader, device)
            torch.save(torch.FloatTensor(train_theta), os.path.join(args.save_dir, 'theta.pt'))
            
            print(f"轮次 {epoch+1}: 保存最佳模型，验证损失: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping:
                print(f"轮次 {epoch+1}: 早停，已经 {args.early_stopping} 轮没有改善")
                break
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))
    
    # 保存重要矩阵
    torch.save(model.alpha.detach().cpu(), os.path.join(args.save_dir, 'final_alpha.pt'))
    torch.save(model.rho_color.detach().cpu(), os.path.join(args.save_dir, 'final_rho_color.pt'))
    torch.save(model.rho_text.detach().cpu(), os.path.join(args.save_dir, 'final_rho_text.pt'))
    
    # 保存最终theta矩阵
    final_theta = extract_embeddings(model, train_loader, device)
    torch.save(torch.FloatTensor(final_theta), os.path.join(args.save_dir, 'final_theta.pt'))
    
    # 保存模型架构信息
    model_architecture = {
        'num_topics': args.num_topics,
        'color_dim': data['color_dim'],
        'text_dim': data['text_dim'],
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout
    }
    with open(os.path.join(args.save_dir, 'model_architecture.json'), 'w') as f:
        json.dump(model_architecture, f, indent=4)
    
    # 绘制损失曲线
    plot_loss_curves(train_losses, val_losses, args.output_dir)
    
    # 保存损失历史
    loss_history = {
        'train': train_losses,
        'val': val_losses
    }
    with open(os.path.join(args.output_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f, indent=4)
    
    # 保存主题评估历史
    with open(os.path.join(args.output_dir, 'topic_metrics_history.json'), 'w') as f:
        json.dump(topic_metrics_history, f, indent=4)
    
    print("训练完成！")
    print(f"模型和矩阵已保存到: {args.save_dir}")
    print(f"训练日志和评估结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 