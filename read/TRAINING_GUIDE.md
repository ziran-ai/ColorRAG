# 多模态主题模型训练指南

## 概述

本指南将帮助您训练一个多模态主题模型，该模型能够学习颜色和文本之间的联合分布，并生成有意义的主题表示。

## 文件结构

```
CT-Model/
├── src/
│   ├── topic_model.py                    # 模型定义
│   ├── train_simple.py                   # 简单训练脚本
│   ├── train_with_separate_encoders.py   # 分别保存编码器训练脚本
│   ├── train_complete.py                 # 完整训练脚本
│   └── eval_utils.py                     # 评估工具
├── models/                               # 模型保存目录
│   └── matrices/                         # 矩阵文件目录
├── outputs/                              # 输出结果目录
├── run_training.py                      # 传统训练启动脚本
├── run_training_separate.py             # 分别保存训练启动脚本
├── palettes_descriptions.xlsx            # 数据文件
└── TRAINING_GUIDE.md                    # 本指南
```

## 数据格式要求

您的数据文件 `palettes_descriptions.xlsx` 应包含以下列：

### 必需列：
- **颜色列**: `Color_1_R`, `Color_1_G`, `Color_1_B`, `Color_2_R`, `Color_2_G`, `Color_2_B`, ..., `Color_5_R`, `Color_5_G`, `Color_5_B`
  - 这些列包含5个颜色的RGB值（已在0-1范围内，无需额外归一化）
- **文本列**: `description` 或 `Description`
  - 包含每个颜色方案的文本描述

### 可选列：
- **评分列**: `Targets`, `targets`, `score`, `Score`
  - 包含人工评分（用于监督学习，会自动归一化到0-1范围）

## 训练步骤

### 1. 环境准备

确保安装了所有必要的依赖：

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib joblib tqdm
```

### 2. 数据检查

首先检查您的数据文件格式是否正确：

```python
import pandas as pd

# 读取数据
df = pd.read_excel('palettes_descriptions.xlsx')
print("数据形状:", df.shape)
print("列名:", list(df.columns))

# 检查颜色列
color_cols = [f'Color_{i}_{c}' for i in range(1, 6) for c in ['R', 'G', 'B']]
missing_cols = [col for col in color_cols if col not in df.columns]
if missing_cols:
    print("缺少颜色列:", missing_cols)
else:
    print("颜色列完整")

# 检查文本列
text_cols = ['description', 'Description', 'text', 'Text']
text_col = None
for col in text_cols:
    if col in df.columns:
        text_col = col
        break

if text_col:
    print(f"找到文本列: {text_col}")
else:
    print("未找到文本列")
```

### 3. 开始训练

#### 方法一：分别保存方式（推荐，获得编码器和解码器）

```bash
# 使用分别保存方式训练
python run_training_separate.py

# 或者直接运行
python src/train_with_separate_encoders.py --save_separate
```

#### 方法二：传统方式（完整模型）

```bash
# 使用传统方式训练
python run_training.py

# 或者直接运行训练脚本
python src/train_simple.py \
    --data_path palettes_descriptions.xlsx \
    --num_topics 50 \
    --embedding_dim 32 \
    --hidden_dim 128 \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.001 \
    --kl_weight 0.01 \
    --score_weight 1.0 \
    --early_stopping 10
```

#### 训练方式对比

| 方式 | 优点 | 适用场景 |
|------|------|----------|
| **分别保存方式** | 支持单独使用编码器/解码器，便于跨模态生成 | Topic-RAG系统、跨模态生成 |
| **传统方式** | 简单易用，完整端到端推理 | 快速原型、简单部署 |

### 4. 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `palettes_descriptions.xlsx` | 数据文件路径 |
| `--num_topics` | 50 | 主题数量K |
| `--embedding_dim` | 32 | 嵌入空间维度L |
| `--hidden_dim` | 128 | 编码器隐藏层维度 |
| `--batch_size` | 64 | 批次大小 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--kl_weight` | 0.01 | KL散度损失权重 |
| `--score_weight` | 1.0 | 评分预测损失权重 |
| `--early_stopping` | 10 | 早停轮数 |

## 训练过程

### 损失函数

模型使用四个损失函数的加权和：

1. **颜色重构损失** (MSE): 重构颜色与原始颜色的均方误差
2. **文本重构损失** (交叉熵): 重构文本与原始文本的交叉熵
3. **KL散度损失**: VAE正则化项，确保潜在空间的正则性
4. **评分预测损失** (MSE): 预测评分与真实评分的均方误差

总损失 = 颜色损失 + 文本损失 + kl_weight × KL损失 + score_weight × 评分损失

### 评估指标

训练过程中会计算以下评估指标：

- **ARI**: Adjusted Rand Index（聚类质量）
- **NMI**: Normalized Mutual Information（聚类质量）
- **ASW**: Average Silhouette Width（聚类质量）
- **Topic_Coherence_Mod1/Mod2**: 主题一致性（颜色/文本模态）
- **Topic_Sparsity_Mod1/Mod2**: 主题稀疏性（颜色/文本模态）
- **Topic_Specificity_Mod1/Mod2**: 主题特异性（颜色/文本模态）
- **Topic_Discreteness**: 主题离散性
- **Cross_Modal_Alignment**: 跨模态对齐

## 输出文件

### 分别保存方式的文件结构

```
models/
├── best_color_encoder.pth      # 颜色编码器
├── best_text_encoder.pth       # 文本编码器
├── best_decoder.pth            # 解码器（包含嵌入矩阵）
├── best_theta.pt               # 主题比例矩阵 (N × K)
├── final_theta.pt              # 最终主题比例矩阵
├── tfidf_vectorizer.pkl        # TF-IDF向量化器
├── vocab.json                  # 词汇表
├── model_architecture.json     # 模型架构信息
└── matrices/                   # 矩阵文件夹
    ├── alpha.pt                # 主题嵌入矩阵 (K × L)
    ├── rho_color.pt            # 颜色特征嵌入矩阵 (L × 15)
    ├── rho_text.pt             # 文本特征嵌入矩阵 (L × V_text)
    ├── lambda_color.pt         # 颜色偏置矩阵
    ├── lambda_text.pt          # 文本偏置矩阵
    ├── final_alpha.pt          # 最终主题嵌入矩阵
    ├── final_rho_color.pt      # 最终颜色特征嵌入矩阵
    ├── final_rho_text.pt       # 最终文本特征嵌入矩阵
    ├── final_lambda_color.pt   # 最终颜色偏置矩阵
    └── final_lambda_text.pt    # 最终文本偏置矩阵
```

### 传统方式的文件结构

```
models/
├── best_model.pth              # 最佳模型权重
├── final_model.pth             # 最终模型权重
├── best_theta.pt               # 主题比例矩阵
├── final_theta.pt              # 最终主题比例矩阵
├── tfidf_vectorizer.pkl        # TF-IDF向量化器
├── vocab.json                  # 词汇表
├── model_architecture.json     # 模型架构信息
└── matrices/                   # 矩阵文件夹
    ├── alpha.pt                # 主题嵌入矩阵
    ├── rho_color.pt            # 颜色特征嵌入矩阵
    ├── rho_text.pt             # 文本特征嵌入矩阵
    ├── lambda_color.pt         # 颜色偏置矩阵
    ├── lambda_text.pt          # 文本偏置矩阵
    └── final_*.pt              # 最终版本矩阵
```

### 输出文件 (`outputs/`)

- `loss_curves.png`: 损失曲线图
- `params.json`: 训练参数
- `training_history.json`: 训练历史
- `metrics_epoch_*.json`: 各轮次的评估指标

## 模型使用

### 分别保存方式的使用

```python
import torch
import joblib
from src.topic_model import MultiOmicsETM

# 创建模型实例
model = MultiOmicsETM(...)  # 使用相同的参数

# 加载颜色编码器
color_encoder_state = torch.load('models/best_color_encoder.pth')
model.color_encoder.load_state_dict(color_encoder_state['color_encoder'])
model.color_mean.load_state_dict(color_encoder_state['color_mean'])
model.color_logvar.load_state_dict(color_encoder_state['color_logvar'])

# 加载文本编码器
text_encoder_state = torch.load('models/best_text_encoder.pth')
model.text_encoder.load_state_dict(text_encoder_state['text_encoder'])
model.text_mean.load_state_dict(text_encoder_state['text_mean'])
model.text_logvar.load_state_dict(text_encoder_state['text_logvar'])

# 加载解码器
decoder_state = torch.load('models/best_decoder.pth')
model.alpha = decoder_state['alpha']
model.rho_color = decoder_state['rho_color']
model.rho_text = decoder_state['rho_text']
model.lambda_color = decoder_state['lambda_color']
model.lambda_text = decoder_state['lambda_text']
model.score_predictor.load_state_dict(decoder_state['score_predictor'])

# 加载向量化器
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# 加载矩阵（可选，用于直接访问）
alpha = torch.load('models/matrices/alpha.pt')
rho_color = torch.load('models/matrices/rho_color.pt')
rho_text = torch.load('models/matrices/rho_text.pt')
theta = torch.load('models/best_theta.pt')
```

### 传统方式的使用

```python
import torch
import joblib
from src.topic_model import MultiOmicsETM

# 加载完整模型
model = MultiOmicsETM(...)  # 使用相同的参数
model.load_state_dict(torch.load('models/best_model.pth'))

# 加载向量化器
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# 加载矩阵
alpha = torch.load('models/matrices/alpha.pt')
rho_color = torch.load('models/matrices/rho_color.pt')
rho_text = torch.load('models/matrices/rho_text.pt')
theta = torch.load('models/best_theta.pt')
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小 `batch_size`
   - 减小 `hidden_dim` 或 `embedding_dim`

2. **训练不收敛**
   - 调整学习率 `lr`
   - 调整损失权重 `kl_weight` 和 `score_weight`
   - 增加训练轮数 `epochs`

3. **数据格式错误**
   - 检查颜色列名是否正确
   - 确保RGB值在0-255范围内
   - 检查文本列是否存在

4. **评估指标异常**
   - 检查数据质量
   - 调整主题数量 `num_topics`

### 性能优化

- 使用GPU加速训练（设置 `--device cuda`）
- 调整批次大小以平衡内存使用和训练速度
- 使用早停机制避免过拟合

## 下一步

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

### 推荐使用方式

对于Topic-RAG系统，推荐使用**分别保存方式**，因为：
- 支持单独使用编码器或解码器
- 便于跨模态生成
- 更灵活的模型使用方式
- 便于集成到复杂的系统中

详细的使用方法请参考项目中的其他文档。 