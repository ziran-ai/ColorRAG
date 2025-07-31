#!/usr/bin/env python3
"""
Streamlit应用启动脚本
"""

import os
import sys
import subprocess
import argparse

def check_dependencies():
    """检查依赖包"""
    try:
        import streamlit
        import torch
        import numpy
        import PIL
        import sklearn
        import requests
        print("✅ 所有依赖包已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        return False

def check_models():
    """检查模型文件"""
    model_files = [
        "/root/autodl-tmp/AETM/models/model_architecture.json",
        "/root/autodl-tmp/AETM/models/best_decoder.pth",
        "/root/autodl-tmp/AETM/models/best_theta.pt"
    ]
    
    missing_files = []
    for file in model_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️ 缺少模型文件: {missing_files}")
        return False
    else:
        print("✅ 模型文件检查通过")
        return True

def check_knowledge_base():
    """检查知识库"""
    kb_file = "knowledge_base.pkl"
    if os.path.exists(kb_file):
        print("✅ 知识库文件存在")
        return True
    else:
        print(f"⚠️ 知识库文件不存在: {kb_file}")
        return False

def main():
    parser = argparse.ArgumentParser(description="RAG配色方案生成系统 - Streamlit版")
    parser.add_argument("--port", type=int, default=8503, help="端口号")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--check-only", action="store_true", help="仅检查环境")
    
    args = parser.parse_args()
    
    print("🎨 RAG配色方案生成系统 - Streamlit版")
    print("=" * 60)
    
    # 环境检查
    print("📋 正在检查系统环境...")
    
    deps_ok = check_dependencies()
    models_ok = check_models()
    kb_ok = check_knowledge_base()
    
    if args.check_only:
        print("\n📊 环境检查完成")
        print(f"依赖包: {'✅' if deps_ok else '❌'}")
        print(f"模型文件: {'✅' if models_ok else '⚠️'}")
        print(f"知识库: {'✅' if kb_ok else '⚠️'}")
        return 0
    
    if not deps_ok:
        print("❌ 依赖包检查失败，请安装必要的包")
        return 1
    
    if not models_ok:
        print("❌ 模型文件缺失，请检查模型路径")
        return 1
    
    if not kb_ok:
        print("❌ 知识库文件缺失，请检查知识库文件")
        return 1
    
    # 启动Streamlit应用
    print(f"\n🚀 正在启动Streamlit应用...")
    print(f"地址: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止应用")
    print("=" * 60)
    
    try:
        cmd = [
            "streamlit", "run", "streamlit_app.py",
            "--server.port", str(args.port),
            "--server.address", args.host,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light",
            "--theme.primaryColor", "#2E86AB"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        return 1
    except FileNotFoundError:
        print("❌ 未找到streamlit命令，请确保已安装streamlit")
        print("运行: pip install streamlit")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
