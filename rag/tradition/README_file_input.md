# RAG系统文件输入使用指南

## 概述

本系统支持从文件读取用户输入并批量运行RAG系统，生成个性化的设计方案。

## 支持的文件格式

### 1. 纯文本文件 (.txt)
每行一个用户需求：
```
我想要一个现代简约风格的配色方案，适合办公环境
我想要一个古典奢华的配色方案，适合高端酒店大堂环境
我想要一个色彩缤纷的配色方案，适合儿童游乐场环境
```

### 2. JSON文件 (.json)
包含用户需求的数组：
```json
[
  "我想要一个现代简约风格的配色方案，适合办公环境",
  "我想要一个古典奢华的配色方案，适合高端酒店大堂环境",
  "我想要一个色彩缤纷的配色方案，适合儿童游乐场环境"
]
```

### 3. CSV文件 (.csv)
包含用户需求列的文件：
```csv
user_input
我想要一个现代简约风格的配色方案，适合办公环境
我想要一个古典奢华的配色方案，适合高端酒店大堂环境
我想要一个色彩缤纷的配色方案，适合儿童游乐场环境
```

## 使用方法

### 基本用法

```bash
# 使用纯文本文件
python run_from_file.py --input inputs.txt

# 使用JSON文件
python run_from_file.py --input inputs.json

# 使用CSV文件
python run_from_file.py --input inputs.csv
```

### 高级用法

```bash
# 指定输出文件
python run_from_file.py --input inputs.txt --output results/my_results.json

# 指定图片路径
python run_from_file.py --input inputs.txt --image my_image.jpg

# 调整检索候选数量
python run_from_file.py --input inputs.txt --top_k 10

# 使用自定义API密钥
python run_from_file.py --input inputs.txt --api_key your_api_key_here
```

### 完整参数说明

```bash
python run_from_file.py [参数]

参数:
  --input, -i         输入文件路径 (必需)
  --output, -o        输出文件路径 (默认: outputs/rag_results.json)
  --image             图片路径 (默认: test_image.jpg)
  --top_k             检索候选数量 (默认: 5)
  --api_key           DeepSeek API密钥 (默认: sk-3c4ba59c8b094106995821395c7bc60e)
```

## 输出格式

系统会生成JSON格式的输出文件，包含以下信息：

```json
[
  {
    "input_id": 1,
    "user_input": "我想要一个现代简约风格的配色方案，适合办公环境",
    "generated_plan": "### **全新设计方案： \"都市静界\" 现代简约办公配色方案**...",
    "candidates": [
      {
        "description": "This palette, titled \"70's Revival,\"...",
        "text_score": 0.994,
        "color_score": 0.927,
        "combined_score": 0.967
      }
    ],
    "timestamp": "2024-01-01T12:00:00.000000"
  }
]
```

## 示例运行

### 1. 创建输入文件

```bash
# 创建纯文本输入文件
echo "我想要一个现代简约风格的配色方案，适合办公环境" > my_inputs.txt
echo "我想要一个古典奢华的配色方案，适合高端酒店大堂环境" >> my_inputs.txt
```

### 2. 运行系统

```bash
python run_from_file.py --input my_inputs.txt --output my_results.json
```

### 3. 查看结果

```bash
# 查看输出文件
cat my_results.json

# 或者使用jq格式化查看
jq '.' my_results.json
```

## 批量处理建议

### 1. 准备输入文件
- 确保每个用户需求描述清晰具体
- 包含风格、场景、氛围等关键词
- 避免过于简单或模糊的描述

### 2. 监控处理进度
- 系统会显示每个输入的处理进度
- 成功/失败的统计信息
- 平均生成长度等指标

### 3. 结果分析
- 检查生成方案的质量和多样性
- 分析检索到的候选方案相关性
- 根据结果调整输入描述

## 错误处理

系统会处理以下错误情况：
- 文件不存在或格式错误
- API调用失败
- 系统初始化失败
- 单个输入处理失败

错误信息会记录在输出文件中，不会影响其他输入的处理。

## 性能优化

### 1. 批量处理
- 一次性处理多个输入，减少系统初始化开销
- 建议批量大小：10-50个输入

### 2. 资源管理
- 系统会自动管理内存和GPU资源
- 长时间运行建议监控系统资源使用

### 3. 错误恢复
- 支持断点续传（重新运行会跳过已处理的输入）
- 失败重试机制

## 注意事项

1. **API限制**: 注意DeepSeek API的调用频率限制
2. **文件编码**: 确保输入文件使用UTF-8编码
3. **图片路径**: 确保指定的图片文件存在
4. **输出目录**: 系统会自动创建输出目录
5. **网络连接**: 确保网络连接稳定，API调用需要网络

## 故障排除

### 常见问题

1. **文件读取失败**
   - 检查文件路径是否正确
   - 确认文件格式是否支持
   - 验证文件编码是否为UTF-8

2. **API调用失败**
   - 检查API密钥是否有效
   - 确认网络连接正常
   - 查看API调用频率限制

3. **系统初始化失败**
   - 检查模型文件是否存在
   - 确认依赖包已安装
   - 验证GPU/CPU配置

### 调试模式

```bash
# 启用详细日志
python run_from_file.py --input inputs.txt --debug
```

## 扩展功能

### 1. 自定义输出格式
可以修改`save_results_to_file`函数支持其他输出格式（如CSV、Excel等）

### 2. 并行处理
可以添加多进程支持，提高批量处理效率

### 3. 结果过滤
可以添加结果质量评估和过滤功能

### 4. 输入验证
可以添加输入格式验证和预处理功能 