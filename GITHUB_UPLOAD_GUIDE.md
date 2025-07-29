# 📤 AETM项目上传到GitHub指南

## 🚀 **完整上传步骤**

### 📋 **步骤一：在GitHub创建仓库**

#### 1. 登录GitHub
- 访问 https://github.com
- 登录您的GitHub账号

#### 2. 创建新仓库
- 点击右上角的 "+" 按钮
- 选择 "New repository"
- 填写仓库信息：
  - **Repository name**: `AETM` 或 `AI-Color-Design-System`
  - **Description**: `🎨 AI Enhanced Topic Modeling for Color Design - 基于深度学习和RAG技术的智能配色设计系统`
  - **Visibility**: Public (推荐) 或 Private
  - **不要**勾选 "Add a README file" (我们已经有了)
  - **不要**勾选 "Add .gitignore" (我们已经有了)
- 点击 "Create repository"

### 📁 **步骤二：准备本地仓库**

#### 1. 进入项目目录
```bash
cd /root/autodl-tmp/AETM
```

#### 2. 初始化Git仓库
```bash
git init
```

#### 3. 添加所有文件
```bash
git add .
```

#### 4. 检查要提交的文件
```bash
git status
```

#### 5. 创建首次提交
```bash
git commit -m "🎨 Initial commit: AETM - AI Enhanced Topic Modeling for Color Design

✨ Features:
- 🤖 Deep learning based topic modeling
- 🔍 LangChain RAG system with 10,702 professional color schemes
- 🖼️ Multi-modal input (text + image)
- 🌐 Streamlit web interface
- 🎯 Personalized color scheme generation

📁 Project Structure:
- Core models and training pipeline
- LangChain RAG implementation
- Web application interface
- Comprehensive documentation"
```

### 🔗 **步骤三：连接到GitHub**

#### 1. 添加远程仓库
```bash
# 替换 YOUR_USERNAME 为您的GitHub用户名
# 替换 YOUR_REPOSITORY_NAME 为您创建的仓库名
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
```

#### 2. 设置主分支
```bash
git branch -M main
```

#### 3. 推送到GitHub
```bash
git push -u origin main
```

### 🔐 **步骤四：处理认证（如果需要）**

#### 方法1：使用Personal Access Token (推荐)
1. 在GitHub设置中生成Personal Access Token
2. 使用token作为密码：
```bash
# 当提示输入密码时，输入您的Personal Access Token
git push -u origin main
```

#### 方法2：使用SSH密钥
1. 生成SSH密钥：
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. 添加SSH密钥到GitHub账户

3. 使用SSH URL：
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

## 📝 **步骤五：完善GitHub仓库**

### 1. 检查上传结果
- 访问您的GitHub仓库页面
- 确认所有文件都已上传
- 检查README.md是否正确显示

### 2. 设置仓库描述和标签
在GitHub仓库页面：
- 点击右上角的 "⚙️ Settings"
- 在 "General" 部分添加：
  - **Description**: `🎨 AI Enhanced Topic Modeling for Color Design`
  - **Website**: 如果有部署的话
  - **Topics**: `ai`, `color-design`, `rag`, `langchain`, `streamlit`, `deep-learning`, `topic-modeling`

### 3. 创建Release (可选)
- 点击 "Releases"
- 点击 "Create a new release"
- 设置版本号：`v1.0.0`
- 标题：`🎨 AETM v1.0.0 - Initial Release`
- 描述项目特点和功能

## 🔄 **后续更新流程**

### 日常更新命令
```bash
# 1. 添加修改的文件
git add .

# 2. 提交更改
git commit -m "✨ Add new feature: [描述新功能]"

# 3. 推送到GitHub
git push origin main
```

### 常用Git命令
```bash
# 查看状态
git status

# 查看提交历史
git log --oneline

# 查看远程仓库
git remote -v

# 拉取最新更改
git pull origin main

# 创建新分支
git checkout -b feature/new-feature

# 切换分支
git checkout main
```

## 📊 **文件大小检查**

### 检查大文件
```bash
# 查找大于50MB的文件
find . -type f -size +50M

# 查看文件大小
du -sh *
```

### 处理大文件
如果有文件超过100MB，考虑：
1. 使用Git LFS (Large File Storage)
2. 将大文件移到云存储
3. 在.gitignore中排除大文件

## 🛡️ **安全注意事项**

### 1. 检查敏感信息
确保以下信息不会上传：
- API密钥和访问令牌
- 数据库密码
- 个人身份信息
- 大型模型文件（如果不必要）

### 2. 环境变量示例
创建 `.env.example` 文件：
```bash
# API配置示例
DEEPSEEK_API_KEY=your_deepseek_api_key_here
ALI_API_KEY=your_ali_api_key_here
DOUBAO_API_KEY=your_doubao_api_key_here
```

## 📚 **推荐的仓库结构**

您的GitHub仓库将包含：
```
AETM/
├── 📖 README.md                    # 项目主页
├── 📋 PROJECT_README.md            # 详细项目说明
├── 🔧 TECHNICAL_GUIDE.md           # 技术文档
├── 🚀 DEPLOYMENT_GUIDE.md          # 部署指南
├── 📤 GITHUB_UPLOAD_GUIDE.md       # 本指南
├── 📄 LICENSE                      # 开源许可证
├── 🚫 .gitignore                   # Git忽略文件
├── 📦 requirements_training.txt     # 训练依赖
├── 📊 data/                        # 数据文件
├── 🤖 models/                      # 模型文件
├── 🎯 rag/                         # RAG系统
├── 🏗️ src/                         # 源代码
├── 📈 outputs/                     # 输出文件
└── 📚 read/                        # 文档
```

## 🎯 **成功标志**

上传成功后，您应该能够：
1. ✅ 在GitHub上看到完整的项目文件
2. ✅ README.md正确显示项目信息
3. ✅ 其他用户可以clone您的项目
4. ✅ 项目有清晰的描述和标签
5. ✅ 没有敏感信息泄露

## 🔗 **有用的链接**

- [GitHub官方文档](https://docs.github.com/)
- [Git基础教程](https://git-scm.com/book)
- [Markdown语法指南](https://guides.github.com/features/mastering-markdown/)
- [开源许可证选择](https://choosealicense.com/)

---

**🎉 恭喜！您的AETM项目即将在GitHub上与世界分享！**
