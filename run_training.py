#!/usr/bin/env python3
"""
多模态主题模型训练脚本
"""

import subprocess
import sys
import os

def main():
    """主函数"""
    print("开始训练多模态主题模型...")
    
    # 训练参数
    cmd = [
        sys.executable, "src/train_simple.py",
        "--data_path", "data/palettes_descriptions.xlsx",
        "--num_topics", "50",
        "--embedding_dim", "32",
        "--hidden_dim", "128",
        "--batch_size", "64",
        "--epochs", "100",
        "--lr", "0.001",
        "--kl_weight", "0.01",
        "--score_weight", "1.0",
        "--early_stopping", "10",
        "--save_dir", "models",
        "--output_dir", "outputs",
        "--device", "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    ]
    
    print("执行命令:", " ".join(cmd))
    
    try:
        # 运行训练脚本
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("训练完成！")
        print("输出:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("训练失败！")
        print("错误:", e.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 