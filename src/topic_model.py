import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiOmicsETM(nn.Module):
    """
    多组学嵌入式主题模型 (Multi-Omics Embedded Topic Model)
    
    参数:
        num_topics: 主题数量 K
        color_dim: 颜色特征维度 (15 = 5个颜色 × 3个RGB通道)
        text_dim: 文本特征维度 (TF-IDF向量的长度)
        embedding_dim: 嵌入空间维度 L
        hidden_dim: 编码器隐藏层维度
        dropout: Dropout比率
    """
    def __init__(self, num_topics, color_dim, text_dim, embedding_dim=32, 
                 hidden_dim=128, dropout=0.1):
        super(MultiOmicsETM, self).__init__()
        
        self.num_topics = num_topics
        self.color_dim = color_dim
        self.text_dim = text_dim
        self.embedding_dim = embedding_dim
        
        # 颜色编码器 (Color Encoder)
        self.color_encoder = nn.Sequential(
            nn.Linear(color_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 颜色均值和对数方差
        self.color_mean = nn.Linear(hidden_dim // 2, num_topics)
        self.color_logvar = nn.Linear(hidden_dim // 2, num_topics)
        
        # 文本编码器 (Text Encoder)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 文本均值和对数方差
        self.text_mean = nn.Linear(hidden_dim // 2, num_topics)
        self.text_logvar = nn.Linear(hidden_dim // 2, num_topics)
        
        # 评分预测头 (Score Prediction Head)
        self.score_predictor = nn.Sequential(
            nn.Linear(num_topics, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 可学习的嵌入矩阵
        # alpha: 主题嵌入矩阵 (K × L)
        self.alpha = nn.Parameter(torch.randn(num_topics, embedding_dim))
        
        # rho_color: 颜色特征嵌入矩阵 (L × 15)
        self.rho_color = nn.Parameter(torch.randn(embedding_dim, color_dim))
        
        # rho_text: 文本特征嵌入矩阵 (L × text_dim)
        self.rho_text = nn.Parameter(torch.randn(embedding_dim, text_dim))
        
        # lambda: 偏置项
        self.lambda_color = nn.Parameter(torch.zeros(1, color_dim))
        self.lambda_text = nn.Parameter(torch.zeros(1, text_dim))
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 特殊初始化嵌入矩阵
        nn.init.xavier_uniform_(self.alpha)
        nn.init.xavier_uniform_(self.rho_color)
        nn.init.xavier_uniform_(self.rho_text)
    
    def encode_color(self, x_color):
        """颜色编码器"""
        h = self.color_encoder(x_color)
        mu = self.color_mean(h)
        logvar = self.color_logvar(h)
        return mu, logvar
    
    def encode_text(self, x_text):
        """文本编码器"""
        h = self.text_encoder(x_text)
        mu = self.text_mean(h)
        logvar = self.text_logvar(h)
        return mu, logvar
    
    def product_of_gaussians(self, mu1, logvar1, mu2, logvar2):
        """
        高斯分布的乘积 (Product of Gaussians)
        
        将两个高斯分布合并为一个更精确的联合高斯分布
        
        公式:
        (sigma_n*)^2 = ((sigma_n_color)^2 * (sigma_n_text)^2) / ((sigma_n_color)^2 + (sigma_n_text)^2)
        mu_n* = (sigma_n*)^2 * ((mu_n_color / (sigma_n_color)^2) + (mu_n_text / (sigma_n_text)^2))
        """
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        
        # 计算联合方差
        joint_var = (var1 * var2) / (var1 + var2 + 1e-10)
        
        # 计算联合均值
        joint_mu = joint_var * (mu1 / (var1 + 1e-10) + mu2 / (var2 + 1e-10))
        
        # 转换回对数方差
        joint_logvar = torch.log(joint_var + 1e-10)
        
        return joint_mu, joint_logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def get_theta(self, delta):
        """将潜在向量转换为主题比例"""
        return F.softmax(delta, dim=1)
    
    def decode(self, theta):
        """
        解码器: 从主题比例重构原始数据
        
        Args:
            theta: 主题比例向量 (batch_size, num_topics)
            
        Returns:
            tuple: (重构的颜色数据, 重构的文本数据)
        """
        # 主题嵌入矩阵乘以特征嵌入矩阵
        color_logits = torch.matmul(theta, torch.matmul(self.alpha, self.rho_color)) + self.lambda_color
        text_logits = torch.matmul(theta, torch.matmul(self.alpha, self.rho_text)) + self.lambda_text
        
        # 颜色数据使用Sigmoid激活函数 (归一化到[0,1]区间)
        recon_color = torch.sigmoid(color_logits)
        
        # 文本数据使用Softmax激活函数 (转换为概率分布)
        recon_text = F.softmax(text_logits, dim=1)
        
        return recon_color, recon_text
    
    def predict_score(self, theta):
        """预测评分"""
        return self.score_predictor(theta)
    
    def forward(self, x_color, x_text):
        """
        前向传播
        
        Args:
            x_color: 颜色特征 (batch_size, color_dim)
            x_text: 文本特征 (batch_size, text_dim)
            
        Returns:
            tuple: (重构的颜色, 重构的文本, 预测的评分, 主题比例, 均值, 对数方差)
        """
        # 编码
        mu_color, logvar_color = self.encode_color(x_color)
        mu_text, logvar_text = self.encode_text(x_text)
        
        # 高斯分布的乘积
        mu, logvar = self.product_of_gaussians(mu_color, logvar_color, mu_text, logvar_text)
        
        # 重参数化采样
        delta = self.reparameterize(mu, logvar)
        
        # 获取主题比例
        theta = self.get_theta(delta)
        
        # 解码
        recon_color, recon_text = self.decode(theta)
        
        # 预测评分
        pred_score = self.predict_score(theta)
        
        return recon_color, recon_text, pred_score, theta, mu, logvar
    
    def get_topic_word_matrix(self):
        """获取主题-词矩阵，用于解释主题"""
        return torch.matmul(self.alpha, self.rho_text).detach().cpu().numpy()
    
    def get_topic_color_matrix(self):
        """获取主题-颜色矩阵，用于解释主题"""
        return torch.matmul(self.alpha, self.rho_color).detach().cpu().numpy() 