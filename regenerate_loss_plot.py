#!/usr/bin/env python3
"""
重新生成损失曲线图
"""

import json
import matplotlib.pyplot as plt
import os

def regenerate_loss_plot():
    """重新生成损失曲线图"""
    # 读取训练历史
    with open('outputs/training_history.json', 'r') as f:
        history = json.load(f)
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Loss Curves', fontsize=16)
    
    loss_types = ['total', 'color', 'text', 'kl', 'score']
    titles = ['Total Loss', 'Color Loss', 'Text Loss', 'KL Loss', 'Score Loss']
    
    for i, (loss_type, title) in enumerate(zip(loss_types, titles)):
        row = i // 3
        col = i % 3
        
        if loss_type in train_losses and train_losses[loss_type]:
            axes[row, col].plot(train_losses[loss_type], label='Train', color='blue', linewidth=2)
        if loss_type in val_losses and val_losses[loss_type]:
            axes[row, col].plot(val_losses[loss_type], label='Validation', color='red', linewidth=2)
        
        axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        axes[row, col].set_xlabel('Epochs', fontsize=10)
        axes[row, col].set_ylabel('Loss', fontsize=10)
        axes[row, col].legend(fontsize=9)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].tick_params(labelsize=9)
    
    # 隐藏右下角的子图
    axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('outputs/loss_curves_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("损失曲线图已重新生成: outputs/loss_curves_fixed.png")

if __name__ == "__main__":
    regenerate_loss_plot() 