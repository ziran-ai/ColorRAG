# 🚀 AETM 部署和使用指南

## 📋 系统要求

### 硬件要求
- **CPU**: 4核心以上
- **内存**: 8GB以上 (推荐16GB)
- **存储**: 10GB可用空间
- **网络**: 稳定的互联网连接 (用于API调用)

### 软件要求
- **操作系统**: Linux/macOS/Windows
- **Python**: 3.8-3.11
- **浏览器**: Chrome/Firefox/Safari (支持现代Web标准)

## 🛠️ 安装部署

### 方法一: 直接部署 (推荐)

#### 1. 环境准备
```bash
# 克隆项目
git clone <your-repo-url>
cd AETM

# 激活虚拟环境
source aetm_env/bin/activate

# 或创建新的虚拟环境
python -m venv aetm_env
source aetm_env/bin/activate  # Linux/macOS
# aetm_env\Scripts\activate  # Windows
```

#### 2. 安装依赖
```bash
# 安装RAG系统依赖
pip install -r rag/requirements.txt

# 如果需要训练功能，安装训练依赖
pip install -r requirements_training.txt
```

#### 3. 配置API密钥
```bash
# 方法1: 环境变量
export DEEPSEEK_API_KEY="sk-your-deepseek-key"
export ALI_API_KEY="your-ali-key"

# 方法2: 修改代码中的默认值
# 编辑 rag/utils/deepseek_translate.py
# 编辑 rag/utils/ali_qwen_vl.py
```

#### 4. 启动应用
```bash
cd rag
streamlit run langchain_web_app.py --server.port 8501 --server.address 0.0.0.0
```

#### 5. 访问应用
打开浏览器访问: `http://localhost:8501`

### 方法二: Docker部署

#### 1. 创建Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r rag/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "rag/langchain_web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### 2. 构建和运行
```bash
# 构建镜像
docker build -t aetm-app .

# 运行容器
docker run -p 8501:8501 \
  -e DEEPSEEK_API_KEY="your-key" \
  -e ALI_API_KEY="your-key" \
  aetm-app
```

### 方法三: 云平台部署

#### Streamlit Cloud
1. 将代码推送到GitHub
2. 在Streamlit Cloud创建应用
3. 配置环境变量
4. 自动部署

#### AWS/Azure/GCP
1. 使用云平台的容器服务
2. 配置负载均衡和自动扩展
3. 设置域名和SSL证书

## 🎯 使用指南

### 基础使用流程

#### 1. 访问界面
- 打开浏览器访问应用地址
- 等待系统加载完成 (显示"LangChain RAG系统已就绪")

#### 2. 输入设计需求
**文本描述示例**:
```
我需要为一家现代咖啡店设计配色方案。
风格要求：现代简约、温暖舒适
色调偏好：暖色调为主，不要过于鲜艳
应用场景：店面装修、品牌设计、网站
目标客户：25-40岁的都市白领
期望氛围：专业而放松，有品质感
```

**快速标签选择**:
- 风格: 现代简约
- 色调: 暖色调、低饱和度
- 场景: 餐厅咖啡、品牌设计

#### 3. 上传参考图片
- 支持格式: PNG, JPG, JPEG
- 建议尺寸: 500x500像素以上
- 图片类型: 设计作品、自然风景、艺术作品等

#### 4. 生成配色方案
- 点击"生成专业配色方案"按钮
- 等待15-30秒处理时间
- 查看生成的3个不同方案

#### 5. 查看和使用结果
- **配色方案**: 查看5色搭配和颜色值
- **方案详情**: 了解设计理念和应用建议
- **检索知识**: 查看参考的专业案例
- **下载结果**: 保存完整的方案信息

### 高级功能

#### 1. 多方案对比
- 系统生成3个不同风格的方案
- 每个方案都有独特的设计理念
- 可以对比选择最适合的方案

#### 2. 专业术语理解
- 主色: 占主导地位的颜色
- 辅色: 支撑主色的颜色
- 强调色: 用于突出重点的颜色
- 中性色: 平衡整体的颜色

#### 3. 应用建议
- 每个方案都提供具体的使用指导
- 包含不同场景的应用方法
- 给出配色比例和搭配建议

## 🔧 配置优化

### 性能优化

#### 1. 缓存配置
```python
# 在 langchain_web_app.py 中添加缓存
@st.cache_data
def load_knowledge_base():
    return SimpleLangChainRAG()
```

#### 2. 内存优化
```bash
# 设置环境变量限制内存使用
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200
```

#### 3. 并发设置
```bash
# 启动时设置并发参数
streamlit run langchain_web_app.py \
  --server.port 8501 \
  --server.maxUploadSize 200 \
  --server.enableCORS false
```

### 安全配置

#### 1. API密钥管理
```bash
# 使用环境变量文件
echo "DEEPSEEK_API_KEY=your-key" > .env
echo "ALI_API_KEY=your-key" >> .env

# 在代码中加载
from dotenv import load_dotenv
load_dotenv()
```

#### 2. 访问控制
```python
# 添加简单的访问控制
def check_password():
    def password_entered():
        if st.session_state["password"] == "your_password":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", 
                     on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", 
                     on_change=password_entered, key="password")
        st.error("Password incorrect")
        return False
    else:
        return True
```

## 🐛 故障排除

### 常见问题及解决方案

#### 1. 系统启动失败
**错误**: `ModuleNotFoundError`
**解决**:
```bash
# 检查依赖安装
pip list | grep streamlit
pip install -r rag/requirements.txt
```

#### 2. API调用失败
**错误**: `API key not found`
**解决**:
```bash
# 检查环境变量
echo $DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY="your-key"
```

#### 3. 图片上传失败
**错误**: `File size too large`
**解决**:
```bash
# 调整上传限制
streamlit run app.py --server.maxUploadSize 200
```

#### 4. 生成速度慢
**原因**: API调用延迟
**解决**:
- 检查网络连接
- 使用国内API服务
- 启用缓存机制

#### 5. 配色解析失败
**原因**: LLM输出格式不标准
**解决**:
- 检查提示词模板
- 更新解析正则表达式
- 添加容错处理

### 日志和调试

#### 1. 启用详细日志
```bash
# 设置日志级别
export STREAMLIT_LOGGER_LEVEL=debug
streamlit run langchain_web_app.py
```

#### 2. 查看系统状态
```python
# 在应用中添加状态检查
st.sidebar.write("系统状态:")
st.sidebar.write(f"Python版本: {sys.version}")
st.sidebar.write(f"Streamlit版本: {st.__version__}")
st.sidebar.write(f"内存使用: {psutil.virtual_memory().percent}%")
```

## 📊 监控和维护

### 性能监控
- 响应时间统计
- API调用成功率
- 用户使用情况
- 系统资源使用

### 定期维护
- 更新依赖包版本
- 清理临时文件
- 备份重要数据
- 检查API配额

### 数据备份
```bash
# 备份重要文件
tar -czf aetm_backup.tar.gz \
  data/ models/ rag/langchain/knowledge_base.*
```

## 🔮 扩展开发

### 添加新功能
1. 在`langchain_web_app.py`中添加新的界面元素
2. 在`simple_langchain_rag.py`中扩展RAG功能
3. 测试和部署新功能

### 集成新服务
1. 在`rag/utils/`下添加新的API模块
2. 在主应用中集成调用
3. 更新配置和文档

---

**🚀 成功部署，开始您的AI配色之旅！**
