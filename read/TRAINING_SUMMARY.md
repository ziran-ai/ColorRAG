# 多模态主题模型训练流程总结

## 🎯 训练目标

训练一个多模态主题模型，能够：
1. 学习颜色和文本的联合分布
2. 生成有意义的主题表示
3. 支持跨模态生成（文生色、色生文）
4. 预测人工评分
5. 为后续的Topic-RAG系统提供基础

## 📋 完整训练流程

### 第一步：环境准备

```bash
# 1. 安装依赖
python setup_environment.py

# 2. 检查环境
python check_data.py
```

### 第二步：数据预处理

训练脚本会自动完成以下预处理步骤：

1. **读取数据**: 从Excel文件读取颜色和文本数据
2. **颜色处理**: 
   - 提取15个RGB值（5个颜色×3个通道）
   - 数据已在[0,1]区间内，仅进行边界裁剪确保数值稳定性
3. **文本处理**:
   - 使用TF-IDF向量化
   - 构建词汇表
   - 生成文本特征向量
4. **评分处理**:
   - 使用 `Targets` 列作为人工评分
   - 归一化评分到[0,1]区间（原始范围：1.43-4.35）
   - 如果没有评分列，使用随机评分
5. **数据集划分**: 训练集90%，测试集10%

### 第三步：模型训练

#### 模型架构

基于moETM的多模态主题模型：

```
输入层:
├── 颜色塔: 15维 → 128维 → 64维 → K维(μ, logσ²)
└── 文本塔: V_text维 → 128维 → 64维 → K维(μ, logσ²)

融合层:
├── 高斯乘积: 融合两个模态的分布
├── 重参数化: 采样潜在向量δ
└── Softmax: 生成主题比例θ

解码层:
├── 颜色解码: θ × α × ρ_color + λ_color
└── 文本解码: θ × α × ρ_text + λ_text

预测层:
└── 评分预测: θ → 1维评分
```

#### 损失函数

总损失 = 颜色重构损失 + 文本重构损失 + kl_weight × KL散度损失 + score_weight × 评分预测损失

1. **颜色重构损失** (MSE): 重构颜色与原始颜色的均方误差
2. **文本重构损失** (交叉熵): 重构文本与原始文本的交叉熵
3. **KL散度损失**: VAE正则化项，确保潜在空间的正则性
4. **评分预测损失** (MSE): 预测评分与真实评分的均方误差

#### 训练参数

| 参数 | 值 | 说明 |
|------|----|----|
| 主题数量 | 50 | 潜在主题的数量K |
| 嵌入维度 | 32 | 嵌入空间维度L |
| 隐藏维度 | 128 | 编码器隐藏层维度 |
| 批次大小 | 64 | 训练批次大小 |
| 学习率 | 0.001 | Adam优化器学习率 |
| KL权重 | 0.01 | KL散度损失权重 |
| 评分权重 | 1.0 | 评分预测损失权重 |
| 训练轮数 | 100 | 最大训练轮数 |
| 早停 | 10 | 早停轮数 |

### 第四步：评估指标

训练过程中会计算以下评估指标：

#### 聚类质量指标
- **ARI**: Adjusted Rand Index
- **NMI**: Normalized Mutual Information  
- **ASW**: Average Silhouette Width

#### 主题质量指标
- **Topic_Coherence_Mod1/Mod2**: 主题一致性（颜色/文本模态）
- **Topic_Sparsity_Mod1/Mod2**: 主题稀疏性（颜色/文本模态）
- **Topic_Specificity_Mod1/Mod2**: 主题特异性（颜色/文本模态）
- **Topic_Discreteness**: 主题离散性
- **Cross_Modal_Alignment**: 跨模态对齐

#### 批次效应指标（简化版本）
- **B_kBET**: 0.0（需要批次信息）
- **B_ASW**: 0.0（需要批次信息）
- **B_GC**: 0.0（需要批次信息）
- **B_ebm**: 0.0（需要批次信息）

### 第五步：输出文件

#### 模型文件 (`models/`)
- `best_model.pth`: 最佳模型权重
- `final_model.pth`: 最终模型权重
- `best_color_encoder.pth`: 颜色编码器（分别保存方式）
- `best_text_encoder.pth`: 文本编码器（分别保存方式）
- `best_decoder.pth`: 解码器（分别保存方式）
- `best_theta.pt`: 主题比例矩阵 (N × K)
- `final_theta.pt`: 最终主题比例矩阵
- `tfidf_vectorizer.pkl`: TF-IDF向量化器
- `vocab.json`: 词汇表
- `model_architecture.json`: 模型架构信息

#### 矩阵文件 (`models/matrices/`)
- `alpha.pt`: 主题嵌入矩阵 (K × L)
- `rho_color.pt`: 颜色特征嵌入矩阵 (L × 15)
- `rho_text.pt`: 文本特征嵌入矩阵 (L × V_text)
- `lambda_color.pt`: 颜色偏置矩阵
- `lambda_text.pt`: 文本偏置矩阵
- `final_alpha.pt`: 最终主题嵌入矩阵
- `final_rho_color.pt`: 最终颜色特征嵌入矩阵
- `final_rho_text.pt`: 最终文本特征嵌入矩阵
- `final_lambda_color.pt`: 最终颜色偏置矩阵
- `final_lambda_text.pt`: 最终文本偏置矩阵

#### 输出文件 (`outputs/`)
- `loss_curves.png`: 损失曲线图
- `params.json`: 训练参数
- `training_history.json`: 训练历史
- `metrics_epoch_*.json`: 各轮次的评估指标

## 🚀 快速开始

### 方法一：一键启动（推荐）

```bash
python quick_start.py
```

### 方法二：分步执行

```bash
# 1. 检查数据
python check_data.py

# 2. 开始训练
python run_training.py

# 3. 或者直接运行训练脚本
python src/train_simple.py --data_path palettes_descriptions.xlsx
```

## 📊 关键矩阵说明

训练完成后，您将获得以下关键矩阵：

### 1. Alpha矩阵 (α)
- **维度**: K × L
- **作用**: 主题嵌入矩阵，连接主题空间和特征空间
- **用途**: 主题解释、跨模态生成

### 2. Rho_color矩阵 (ρ_color)
- **维度**: L × 15
- **作用**: 颜色特征嵌入矩阵
- **用途**: 颜色重构、颜色生成

### 3. Rho_text矩阵 (ρ_text)
- **维度**: L × V_text
- **作用**: 文本特征嵌入矩阵
- **用途**: 文本重构、文本生成

### 4. Theta矩阵 (θ)
- **维度**: N × K
- **作用**: 每个样本的主题比例分布
- **用途**: 特征表示、相似性计算、Topic-RAG检索

## 🎯 后续应用

训练完成后，您可以：

1. **跨模态生成**:
   - 文生色：输入文本描述，生成对应的颜色方案
   - 色生文：输入颜色方案，生成对应的文本描述

2. **Topic-RAG系统**:
   - 使用theta矩阵进行相似性检索
   - 结合RAG技术实现智能颜色推荐

3. **主题分析**:
   - 分析学习到的主题含义
   - 可视化主题-颜色和主题-文本的关联

4. **评分预测**:
   - 预测新颜色方案的审美评分
   - 辅助设计决策

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   python src/train_simple.py --batch_size 32 --hidden_dim 64
   ```

2. **训练不收敛**
   ```bash
   python src/train_simple.py --lr 0.0001 --kl_weight 0.1
   ```

3. **数据格式错误**
   ```bash
   python check_data.py
   ```

### 性能优化建议

- 使用GPU加速训练
- 调整批次大小平衡内存和速度
- 使用早停机制避免过拟合
- 根据数据规模调整主题数量

## 📚 参考资料

- moETM论文：多组学嵌入式主题模型
- VAE理论：变分自编码器
- 主题模型：LDA、ETM等
- 多模态学习：跨模态表示学习 