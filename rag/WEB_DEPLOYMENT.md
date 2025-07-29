# RAG系统可视化平台部署指南

## 概述

本文档介绍如何部署RAG系统的可视化平台，包括Streamlit和Gradio两种方案。

## 🚀 快速开始

### 方案一：Streamlit（功能丰富）

#### 1. 安装依赖
```bash
cd /root/autodl-tmp/AETM/rag
pip install -r requirements.txt
```

#### 2. 启动应用
```bash
streamlit run web_app.py --server.port 8501 --server.address 0.0.0.0
```

#### 3. 访问应用
打开浏览器访问：`http://localhost:8501`

### 方案二：Gradio（轻量级）

#### 1. 安装依赖
```bash
pip install gradio
```

#### 2. 启动应用
```bash
python gradio_app.py
```

#### 3. 访问应用
打开浏览器访问：`http://localhost:7860`

## 📊 功能对比

| 功能 | Streamlit | Gradio |
|------|-----------|--------|
| 界面复杂度 | 高 | 低 |
| 自定义程度 | 高 | 中 |
| 部署难度 | 中 | 低 |
| 性能 | 中 | 高 |
| 适合场景 | 复杂应用 | 快速原型 |

## 🏗️ 系统架构

### Streamlit版本架构
```
web_app.py
├── RAGWebApp类
│   ├── 系统初始化
│   ├── 界面渲染
│   ├── 结果展示
│   └── 数据管理
├── 多标签页界面
│   ├── 设计生成
│   ├── 结果分析
│   ├── 对话历史
│   └── 系统监控
└── 交互功能
    ├── 文件上传
    ├── 参数调整
    └── 结果导出
```

### Gradio版本架构
```
gradio_app.py
├── RAGGradioApp类
│   ├── 系统加载
│   ├── 界面创建
│   └── 事件处理
├── 简化界面
│   ├── 输入区域
│   ├── 输出区域
│   └── 示例展示
└── 核心功能
    ├── 文本输入
    ├── 图片上传
    └── 结果生成
```

## 🔧 配置说明

### 环境变量配置
```bash
# 设置API密钥
export DEEPSEEK_API_KEY="your_api_key_here"

# 设置模型路径
export MODEL_DIR="../models"

# 设置数据路径
export DATA_DIR="../data"
```

### 配置文件
创建 `.streamlit/config.toml` 文件：
```toml
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

## 📱 界面功能

### Streamlit版本功能

#### 1. 主界面
- **系统状态监控**: 实时显示系统加载状态
- **参数配置**: 可调整检索候选数量、生成温度等
- **图片上传**: 支持拖拽上传参考图片

#### 2. 设计生成
- **智能输入**: 支持多行文本输入
- **实时生成**: 点击按钮即时生成方案
- **结果展示**: 美观的Markdown格式展示

#### 3. 结果分析
- **可视化图表**: 使用Plotly生成交互式图表
- **统计信息**: 详细的生成统计和性能指标
- **历史记录**: 保存和查看历史生成结果

#### 4. 对话历史
- **会话管理**: 保持对话上下文
- **历史查看**: 可展开查看详细对话内容
- **导出功能**: 支持导出对话记录

#### 5. 系统监控
- **性能指标**: 实时监控系统性能
- **资源使用**: 显示CPU、内存、GPU使用情况
- **错误日志**: 记录和显示系统错误

### Gradio版本功能

#### 1. 简化界面
- **文本输入**: 支持多行文本输入
- **图片上传**: 可选的参考图片上传
- **参数调整**: 检索候选数量滑块

#### 2. 结果展示
- **生成方案**: Markdown格式展示
- **候选方案**: 显示参考候选方案
- **统计信息**: 基本的生成统计

#### 3. 示例展示
- **使用示例**: 提供多种输入示例
- **使用技巧**: 详细的使用指导
- **系统信息**: 技术架构说明

## 🚀 部署选项

### 1. 本地部署
```bash
# 克隆项目
git clone <repository_url>
cd AETM/rag

# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run web_app.py
```

### 2. Docker部署
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 3. 云平台部署

#### Streamlit Cloud
1. 将代码推送到GitHub
2. 在Streamlit Cloud连接GitHub仓库
3. 设置环境变量
4. 自动部署

#### Hugging Face Spaces
1. 创建新的Space
2. 上传代码文件
3. 配置requirements.txt
4. 自动构建和部署

## 🔍 性能优化

### 1. 系统优化
```python
# 启用缓存
@st.cache_data
def load_system():
    return TopicRAGSystem(device='cpu', api_key=api_key)

# 异步处理
async def generate_design_async(user_input, image_path):
    # 异步生成逻辑
    pass
```

### 2. 界面优化
```python
# 懒加载
if st.button("加载系统"):
    with st.spinner("正在加载..."):
        system = load_system()

# 分页显示
if len(results) > 10:
    page = st.selectbox("选择页面", range(1, (len(results)//10)+2))
    start_idx = (page-1) * 10
    end_idx = start_idx + 10
    st.write(results[start_idx:end_idx])
```

### 3. 内存优化
```python
# 定期清理缓存
if st.button("清理缓存"):
    st.cache_data.clear()
    st.cache_resource.clear()

# 限制历史记录
MAX_HISTORY = 100
if len(history) > MAX_HISTORY:
    history = history[-MAX_HISTORY:]
```

## 🔧 故障排除

### 常见问题

1. **系统加载失败**
   ```bash
   # 检查模型文件
   ls -la ../models/
   
   # 检查API密钥
   echo $DEEPSEEK_API_KEY
   ```

2. **端口被占用**
   ```bash
   # 查找占用端口的进程
   lsof -i :8501
   
   # 杀死进程
   kill -9 <PID>
   ```

3. **依赖安装失败**
   ```bash
   # 升级pip
   pip install --upgrade pip
   
   # 清理缓存
   pip cache purge
   
   # 重新安装
   pip install -r requirements.txt --no-cache-dir
   ```

### 调试模式
```bash
# 启用详细日志
export STREAMLIT_LOG_LEVEL=debug
streamlit run web_app.py --logger.level=debug
```

## 📈 监控和维护

### 1. 日志监控
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_app.log'),
        logging.StreamHandler()
    ]
)
```

### 2. 性能监控
```python
import time

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        st.metric("执行时间", f"{end_time - start_time:.2f}s")
        return result
    return wrapper
```

### 3. 错误处理
```python
try:
    result = system.generate_design(user_input)
except Exception as e:
    st.error(f"生成失败: {e}")
    st.info("请检查输入格式或联系管理员")
```

## 🎯 最佳实践

### 1. 用户体验
- 提供清晰的输入提示
- 显示处理进度
- 给出错误提示和建议
- 支持结果保存和分享

### 2. 系统稳定性
- 实现错误重试机制
- 添加超时处理
- 定期清理临时文件
- 监控系统资源使用

### 3. 安全性
- 验证用户输入
- 限制文件上传大小
- 保护API密钥
- 记录访问日志

## 📚 扩展开发

### 1. 添加新功能
```python
# 添加新的标签页
with st.tabs(["设计生成", "新功能"]):
    with st.tabs[0]:
        # 原有功能
        pass
    with st.tabs[1]:
        # 新功能
        pass
```

### 2. 集成其他模型
```python
# 添加模型选择
model_type = st.selectbox("选择模型", ["Topic-RAG", "其他模型"])

if model_type == "Topic-RAG":
    result = topic_rag_system.generate(input)
else:
    result = other_model.generate(input)
```

### 3. 多语言支持
```python
# 语言选择
language = st.selectbox("选择语言", ["中文", "English"])

if language == "中文":
    # 中文界面
    pass
else:
    # 英文界面
    pass
```

## 🎉 总结

通过可视化平台，您的RAG系统可以：

1. **提升用户体验**: 直观的界面和交互
2. **扩大使用范围**: 非技术用户也能使用
3. **提高效率**: 批量处理和结果管理
4. **便于监控**: 实时监控系统状态
5. **易于部署**: 支持多种部署方式

选择合适的方案（Streamlit或Gradio）取决于您的具体需求和使用场景。 