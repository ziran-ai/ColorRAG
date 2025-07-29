# Topic-RAG系统使用指南

## 概述

Topic-RAG系统是一个基于多模态主题模型的检索增强生成系统，能够根据用户的文本需求和图片输入，推荐合适的颜色方案。

## 系统架构

系统包含两个核心模块：

### 模块一：图片理解 + 文本融合 + Topic Model推理
1. **图片理解**：使用LLM分析图片内容、风格和关键元素
2. **文本融合**：将图片理解结果与用户文本需求结合
3. **Topic Model推理**：使用训练好的多模态主题模型生成跨模态表示

### 模块二：相似度检索 + 重排序 + 最终生成
1. **初步检索**：基于文本相似度找到Top-K个候选方案
2. **重排序**：结合颜色相似度对候选方案进行重新排序
3. **最终生成**：使用LLM生成优美的方案描述

## 安装依赖

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib joblib tqdm pillow google-generativeai
```

## 环境配置

### 1. 设置DeepSeek API密钥（可选）

如果您想使用LLM功能，需要设置DeepSeek API密钥：

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

或者在代码中设置：

```python
import os
os.environ['DEEPSEEK_API_KEY'] = 'your_api_key_here'
```

或者在初始化系统时直接传入：

```python
api_key = "sk-3c4ba59c8b094106995821395c7bc60e"  # 您的API密钥
system = TopicRAGSystem(device='cpu', api_key=api_key)
```

### 2. 检查模型文件

确保以下文件存在：
- `models/best_color_encoder.pth`
- `models/best_text_encoder.pth`
- `models/best_decoder.pth`
- `models/tfidf_vectorizer.pkl`
- `models/vocab.json`
- `data/palettes_descriptions.xlsx`

## 快速开始

### 1. 基本使用

```python
from topic_rag_system import TopicRAGSystem

# 初始化系统（使用DeepSeek API）
api_key = "sk-3c4ba59c8b094106995821395c7bc60e"  # 您的API密钥
system = TopicRAGSystem(device='cpu', api_key=api_key)  # 或 'cuda' 如果有GPU

# 运行完整流程
user_text = "我想要一个现代简约风格的配色方案，适合办公环境"
image_path = "path/to/your/image.jpg"

result = system.run_full_pipeline(user_text, image_path, top_k=10)
print(result['final_description'])
```

### 2. 分模块使用

```python
# 模块一：从用户输入到Topic Model推理
recon_text_prob, recon_color_prob = system.module_one_process(user_text, image_path)

# 模块二：检索、重排序、生成
result = system.module_two_process(recon_text_prob, recon_color_prob, top_k=10)
```

### 3. 运行测试

```bash
python test_topic_rag.py
```

### 4. 运行示例

```bash
python example_usage.py
```

## 详细使用说明

### 系统初始化

```python
system = TopicRAGSystem(
    model_dir='models',  # 模型文件目录
    device='cpu',        # 计算设备
    api_key='your_api_key'  # DeepSeek API密钥（可选）
)
```

初始化过程包括：
- 加载训练好的模型和编码器
- 构建检索数据库
- 初始化LLM（如果可用）

### 输入格式

#### 文本输入
- 支持中文和英文
- 建议包含风格、场景、氛围等关键词
- 示例：
  - "现代简约风格，适合办公环境"
  - "温暖舒适，适合家居客厅"
  - "充满活力，适合创意工作室"

#### 图片输入
- 支持常见图片格式（JPG, PNG等）
- 建议上传与需求相关的参考图片
- 如果没有图片，系统会使用默认图片

### 输出格式

系统返回一个字典，包含：

```python
{
    'best_plan': pd.Series,           # 最佳方案数据
    'final_description': str,          # 最终生成的描述
    'candidate_scores': List[Dict],    # 候选方案得分
    'best_score': float               # 最佳方案得分
}
```

### 参数说明

#### `run_full_pipeline` 参数
- `user_text`: 用户文本需求
- `image_path`: 图片路径
- `top_k`: 初步检索的候选数量（默认10）

#### `module_two_process` 参数
- `recon_text_prob`: 重构的文本概率向量
- `recon_color_prob`: 重构的颜色概率向量
- `top_k`: 候选数量

## 高级功能

### 1. 自定义权重

您可以修改文本和颜色相似度的权重：

```python
# 在 topic_rag_system.py 中修改
final_score = 0.6 * text_score + 0.4 * color_score  # 当前权重
# 可以调整为：
final_score = 0.7 * text_score + 0.3 * color_score  # 更重视文本
```

### 2. 无LLM模式

如果不想使用LLM，系统会自动降级到简单模式：

```python
# 系统会自动检测LLM可用性
if not system.llm_available:
    print("LLM不可用，使用简单模式")
```

### 3. 批量处理

```python
# 批量处理多个需求
user_inputs = [
    ("现代办公风格", "office.jpg"),
    ("温馨家居风格", "home.jpg"),
    ("创意工作室风格", "studio.jpg")
]

results = []
for text, image in user_inputs:
    result = system.run_full_pipeline(text, image)
    results.append(result)
```

## 性能优化

### 1. GPU加速

如果有GPU，使用CUDA加速：

```python
system = TopicRAGSystem(device='cuda')
```

### 2. 批量检索

对于大量查询，可以预先构建检索数据库：

```python
# 系统会自动构建和缓存检索数据库
system._build_retrieval_database()
```

### 3. 内存优化

对于大型数据集，可以调整检索参数：

```python
# 减少候选数量以节省内存
result = system.run_full_pipeline(user_text, image_path, top_k=5)
```

## 故障排除

### 常见问题

1. **模型加载失败**
   ```
   错误：FileNotFoundError: models/best_color_encoder.pth
   解决：检查模型文件是否存在，确保训练已完成
   ```

2. **DeepSeek API初始化失败**
   ```
   错误：DeepSeek API初始化失败
   解决：检查API密钥是否正确设置，或使用无LLM模式
   ```

3. **内存不足**
   ```
   错误：CUDA out of memory
   解决：使用CPU模式或减少batch_size
   ```

4. **数据格式错误**
   ```
   错误：KeyError: 'Color_1_R'
   解决：检查数据文件格式，确保包含所有必需的颜色列
   ```

5. **API密钥无效**
   ```
   错误：DeepSeek API调用失败
   解决：检查API密钥是否有效，确保有足够的配额
   ```

### 调试模式

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

system = TopicRAGSystem(device='cpu')
```

## 扩展功能

### 1. 添加新的颜色方案

将新的颜色方案添加到数据文件中：

```python
# 在 data/palettes_descriptions.xlsx 中添加新行
new_plan = {
    'Color_1_R': 0.5, 'Color_1_G': 0.3, 'Color_1_B': 0.7,
    'Color_2_R': 0.8, 'Color_2_G': 0.6, 'Color_2_B': 0.4,
    # ... 其他颜色
    'description': '新的颜色方案描述'
}
```

### 2. 自定义相似度计算

```python
# 在 topic_rag_system.py 中修改相似度计算
def custom_similarity(vec1, vec2):
    # 自定义相似度函数
    return cosine_similarity(vec1, vec2)
```

### 3. 集成其他LLM

```python
# 修改 _init_llm 方法以支持其他LLM
def _init_llm(self):
    # 支持OpenAI、Claude等
    pass
```

## 性能指标

### 检索质量
- 文本相似度范围：0-1
- 颜色相似度范围：0-1
- 综合得分：加权平均

### 响应时间
- 系统初始化：~5-10秒
- 单次查询：~2-5秒
- 批量查询：~1-3秒/查询

## 最佳实践

1. **输入描述要具体**：包含风格、场景、氛围等关键词
2. **图片质量要好**：清晰、相关的参考图片
3. **合理设置top_k**：根据需求平衡质量和速度
4. **定期更新数据**：添加新的颜色方案以提高推荐质量

## 技术支持

如果遇到问题，请检查：
1. 模型文件是否完整
2. 数据格式是否正确
3. 依赖包是否安装
4. API密钥是否有效

## 更新日志

- v1.0.0: 初始版本，支持基本的Topic-RAG功能
- 支持图片理解、文本融合、跨模态检索
- 支持LLM生成和简单模式
- 支持GPU加速和批量处理 