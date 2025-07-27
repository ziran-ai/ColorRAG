# 模型保存方式对比


### 方式二：分别保存编码器和解码器

**特点**：
- 分别保存颜色编码器、文本编码器、解码器
- 支持单独使用各个组件
- 便于跨模态生成

**文件结构**：
```
models/
├── best_color_encoder.pth  # 颜色编码器
├── best_text_encoder.pth   # 文本编码器
├── best_decoder.pth        # 解码器（包含嵌入矩阵）
├── best_theta.pt           # 主题比例矩阵
├── tfidf_vectorizer.pkl    # TF-IDF向量化器
├── vocab.json              # 词汇表
└── model_architecture.json # 模型架构信息
```

**加载方式**：
```python
# 创建模型实例
model = MultiOmicsETM(...)

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
```

**优点**：
- 支持单独使用编码器或解码器
- 便于跨模态生成（文生色、色生文）
- 更灵活的模型使用方式
- 便于模型分析和调试

**缺点**：
- 加载过程稍复杂
- 需要手动组装模型

## 使用场景对比

### 传统方式适用场景
- 完整的端到端推理
- 同时有颜色和文本输入
- 简单的模型部署
- 快速原型开发

### 分别保存方式适用场景
- 跨模态生成（只有文本或只有颜色）
- 模型分析和调试
- 灵活的推理应用
- Topic-RAG系统

## 跨模态生成对比

### 传统方式（需要零向量）
```python
# 文本到颜色生成
text_vector = vectorizer.transform([text]).toarray()
text_tensor = torch.FloatTensor(text_vector)
color_tensor = torch.zeros(1, model.color_dim)  # 零向量

# 完整编码
mu_color, logvar_color = model.encode_color(color_tensor)
mu_text, logvar_text = model.encode_text(text_tensor)
mu, logvar = model.product_of_gaussians(mu_color, logvar_color, mu_text, logvar_text)
delta = model.reparameterize(mu, logvar)
theta = model.get_theta(delta)
```

### 分别保存方式（直接使用编码器）
```python
# 文本到颜色生成
text_vector = vectorizer.transform([text]).toarray()

# 直接使用文本编码器
mu_text, logvar_text = model.encode_text(torch.FloatTensor(text_vector))
mu_color = torch.zeros_like(mu_text)
logvar_color = torch.ones_like(logvar_text) * 10  # 高方差

# 高斯乘积
mu, logvar = model.product_of_gaussians(mu_color, logvar_color, mu_text, logvar_text)
delta = model.reparameterize(mu, logvar)
theta = model.get_theta(delta)
```

## 训练脚本选择

### 使用传统方式
```bash
python run_training.py
# 或
python src/train_simple.py
```

### 使用分别保存方式
```bash
python run_training_separate.py
# 或
python src/train_with_separate_encoders.py --save_separate
```

## 加载脚本选择

### 自动检测加载方式
```python
python load_separate_models.py
```

这个脚本会自动检测模型保存方式并相应加载。

## 推荐使用方式

### 对于您的Topic-RAG系统
**推荐使用分别保存方式**，因为：

1. **跨模态生成**：您需要从文本生成颜色，分别保存的编码器更适合
2. **灵活推理**：可以单独使用文本编码器或颜色编码器
3. **模块化设计**：便于集成到复杂的系统中

### 训练命令
```bash
# 使用分别保存方式训练
python run_training_separate.py

# 或者直接运行
python src/train_with_separate_encoders.py --save_separate
```

### 使用训练好的模型
```bash
# 加载并测试模型
python load_separate_models.py
```

这样您就能获得三个独立的模型文件，更适合您的Topic-RAG系统需求！ 