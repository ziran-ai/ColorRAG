# 🔧 AETM 技术指南

## 📋 核心文件详解

### 🌐 主应用文件

#### `rag/langchain_web_app.py` - 主Web应用
**作用**: 项目的核心Web界面，集成所有功能模块
**功能**:
- Streamlit界面框架
- 用户输入处理 (文本+图片)
- RAG系统调用
- 结果展示和可视化
- 多方案配色展示

**关键类**:
- `LangChainRAGWebApp`: 主应用类
- `load_system()`: 加载RAG系统
- `run_rag_generation()`: 执行RAG生成流程
- `parse_color_schemes()`: 解析多个配色方案

### 🤖 RAG系统核心

#### `rag/langchain/simple_langchain_rag.py` - LangChain RAG实现
**作用**: 基于LangChain框架的检索增强生成系统
**功能**:
- 知识库向量化和检索
- 多模态输入处理
- 提示词工程
- LLM调用和结果生成

**关键类**:
- `SimpleLangChainRAG`: 主RAG系统类
- `build_knowledge_base()`: 构建向量知识库
- `run_rag_pipeline()`: 完整RAG流程
- `retrieve_relevant_documents()`: 文档检索

#### `rag/langchain/build_knowledge_base.py` - 知识库构建
**作用**: 将Excel数据转换为向量化知识库
**功能**:
- 读取配色数据
- TF-IDF向量化
- 创建LangChain文档
- 保存向量数据库

### 🛠️ 工具模块

#### `rag/utils/ali_qwen_vl.py` - 阿里视觉理解
**作用**: 调用阿里通义千问视觉模型分析图片
**功能**:
- 图片上传到图床
- 调用通义千问VL API
- 图片内容理解和描述

#### `rag/utils/doubao_vl.py` - 豆包视觉理解
**作用**: 调用豆包视觉模型分析图片 (备用方案)
**功能**:
- 图片处理和上传
- 豆包API调用
- 视觉内容分析

#### `rag/utils/deepseek_translate.py` - 翻译服务
**作用**: 使用DeepSeek API进行中英文翻译
**功能**:
- 中文到英文翻译
- 英文到中文翻译
- 保持原意和风格

### 🏗️ 传统RAG系统 (备用)

#### `rag/tradition/topic_rag_system.py` - 基于主题模型的RAG
**作用**: 基于自训练主题模型的RAG系统
**功能**:
- 加载训练好的主题模型
- 基于主题的文档检索
- 传统RAG流程实现

#### `rag/tradition/run_from_file.py` - 文件输入运行器
**作用**: 从文件读取输入并运行RAG系统
**功能**:
- 读取JSON/TXT输入文件
- 批量处理多个请求
- 结果保存和导出

### 📊 数据文件

#### `data/palettes_descriptions.xlsx` - 核心配色数据库
**内容**: 10,702条专业配色记录
**字段**:
- `name`: 配色方案名称
- `description`: 详细描述
- `colors`: RGB颜色值列表
- `style`: 设计风格标签
- `application`: 应用场景

#### `data/dataset_color.xlsx` - 原始颜色数据
**内容**: 原始的颜色数据集
**用途**: 模型训练的基础数据

### 🤖 模型文件

#### `models/best_*.pth` - 训练好的模型
- `best_color_encoder.pth`: 颜色编码器
- `best_text_encoder.pth`: 文本编码器  
- `best_decoder.pth`: 解码器
- `best_theta.pt`: 主题模型参数

#### `models/tfidf_vectorizer.pkl` - TF-IDF向量化器
**作用**: 文本向量化工具
**用途**: 将文本转换为数值向量

#### `models/vocab.json` - 词汇表
**内容**: 模型训练使用的词汇映射
**格式**: {"word": index} 字典

### 🏗️ 源代码模块

#### `src/topic_model.py` - 主题模型实现
**作用**: 核心的VAE主题模型架构
**包含**:
- `ColorTopicVAE`: 主模型类
- `ColorEncoder`: 颜色编码器
- `TextEncoder`: 文本编码器
- `Decoder`: 解码器

#### `src/train_complete.py` - 完整训练流程
**作用**: 端到端的模型训练脚本
**功能**:
- 数据加载和预处理
- 模型训练循环
- 评估和保存

## 🔄 系统工作流程

### 1. 用户输入处理
```
用户文本描述 → 文本预处理 → 中英文翻译 → 标准化格式
用户上传图片 → 图片分析 → 色彩提取 → 风格识别
```

### 2. 知识检索流程
```
输入向量化 → 相似度计算 → Top-K检索 → 相关文档返回
```

### 3. 方案生成流程
```
检索结果 + 用户需求 → 提示词构建 → LLM调用 → 多方案生成
```

### 4. 结果展示流程
```
生成文本 → 方案解析 → 颜色提取 → 可视化展示
```

## 🛠️ 开发和部署

### 开发环境设置
```bash
# 1. 创建虚拟环境
python -m venv aetm_env
source aetm_env/bin/activate

# 2. 安装依赖
pip install -r rag/requirements.txt

# 3. 配置API密钥
export DEEPSEEK_API_KEY="your_key"
export ALI_API_KEY="your_key"
```

### 运行应用
```bash
cd rag
streamlit run langchain_web_app.py --server.port 8501
```

### 生产部署
```bash
# 使用Docker部署
docker build -t aetm-app .
docker run -p 8501:8501 aetm-app

# 或使用云服务
# 支持AWS, Azure, GCP等平台
```

## 🔧 配置说明

### API配置
- **DeepSeek**: 用于翻译和文本生成
- **阿里通义千问**: 用于图片理解
- **豆包**: 备用图片理解服务

### 模型配置
- **向量维度**: 512
- **主题数量**: 50
- **检索Top-K**: 5

### 界面配置
- **端口**: 8501
- **主题**: Streamlit默认主题
- **语言**: 中文界面

## 📈 性能优化

### 1. 缓存策略
- 知识库向量缓存
- API调用结果缓存
- 图片处理结果缓存

### 2. 并发处理
- 异步API调用
- 多线程图片处理
- 批量向量计算

### 3. 内存管理
- 模型懒加载
- 大文件流式处理
- 定期垃圾回收

## 🐛 常见问题

### Q1: 系统启动失败
**原因**: 依赖包未安装或API密钥未配置
**解决**: 检查requirements.txt和环境变量

### Q2: 生成速度慢
**原因**: API调用延迟或模型加载时间长
**解决**: 使用缓存和模型预加载

### Q3: 配色方案解析失败
**原因**: LLM输出格式不标准
**解决**: 优化提示词和解析逻辑

## 🔮 扩展开发

### 添加新的视觉理解服务
1. 在`rag/utils/`下创建新的API模块
2. 实现标准的图片分析接口
3. 在主系统中集成调用

### 添加新的LLM服务
1. 修改`simple_langchain_rag.py`中的LLM配置
2. 适配新的API调用格式
3. 测试生成质量

### 扩展数据库
1. 准备新的配色数据
2. 运行`build_knowledge_base.py`重建向量库
3. 更新元数据配置

---

**🔧 技术细节决定产品质量！**
