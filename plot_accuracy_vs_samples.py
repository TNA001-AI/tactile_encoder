#!/usr/bin/env python3
"""
绘制模型准确率随数据样本数量变化的趋势图
横坐标：数据样本数量 (50, 100, 150)
纵坐标：测试准确率
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def collect_accuracies_by_samples():
    """收集各模型在不同样本数量下的准确率"""
    results_dir = Path('results')
    
    # 存储结构: {model_type: {sample_size: accuracy}}
    model_data = {}
    
    # 遍历所有results目录
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
            
        results_file = model_dir / 'results.json'
        if not results_file.exists():
            continue
        
        # 读取结果文件
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # 提取准确率
        if 'test_results' not in data or 'accuracy' not in data['test_results']:
            continue
        
        model_name = model_dir.name
        accuracy = data['test_results']['accuracy']
        
        # 解析模型类型和样本数量
        if '_50' in model_name:
            model_type = model_name.replace('_50', '')
            sample_size = 50
        elif '_100' in model_name:
            model_type = model_name.replace('_100', '')
            sample_size = 100
        elif '_150' in model_name:
            model_type = model_name.replace('_150', '')
            sample_size = 150
        else:
            # 无样本数量后缀的模型（如resnet, deepcnn）
            continue
        
        # 存储数据
        if model_type not in model_data:
            model_data[model_type] = {}
        model_data[model_type][sample_size] = accuracy
        
        print(f"{model_name}: {accuracy:.4f} (样本数: {sample_size})")
    
    return model_data

def plot_accuracy_vs_samples(model_data, save_path='reports/figures/comparisons/accuracy_vs_samples.png'):
    """绘制准确率随样本数量变化的折线图"""
    
    # 样本数量
    sample_sizes = [50, 100, 150]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 定义颜色和标记样式
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # 绘制每个模型的折线
    for idx, (model_type, sample_accuracies) in enumerate(sorted(model_data.items())):
        # 准备数据点
        x_data = []
        y_data = []
        
        for size in sample_sizes:
            if size in sample_accuracies:
                x_data.append(size)
                y_data.append(sample_accuracies[size])
        
        # 如果至少有两个数据点，则绘制
        if len(x_data) >= 2:
            ax.plot(x_data, y_data, 
                   marker=markers[idx % len(markers)], 
                   color=colors[idx],
                   linewidth=3.5,
                   markersize=14,
                   label=model_type.upper(),
                   alpha=0.8)
            
            # 为每个模型在不同x位置设置不同的偏移，避免重叠
            # 偏移策略：在50和150样本处交替上下，100样本处根据y值分散
            offset_strategies = {
                'attention': {50: -25, 100: 30, 150: 35},
                'cnn': {50: 20, 100: -30, 150: -25},
                'mlp': {50: -30, 100: 40, 150: 25},
                'resnet': {50: 25, 100: -40, 150: -35}
            }
            
            for x, y in zip(x_data, y_data):
                offset = offset_strategies.get(model_type.lower(), {}).get(x, 15)
                ax.annotate(f'{y:.3f}', 
                           xy=(x, y), 
                           xytext=(0, offset), 
                           textcoords='offset points',
                           ha='center',
                           fontsize=11,
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=colors[idx], 
                                   alpha=0.5,
                                   edgecolor='white',
                                   linewidth=0.5))
    
    # 设置标签和标题
    ax.set_xlabel('Training Sample Size (Samples per Class)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=18, fontweight='bold')
    ax.set_title('Model Accuracy vs Training Sample Size', 
                fontsize=20, fontweight='bold', pad=25)
    
    # 设置x轴刻度
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels([f'{s}' for s in sample_sizes], fontsize=15)
    
    # 设置y轴范围和刻度
    ax.set_ylim([0, 1.05])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    ax.tick_params(axis='y', labelsize=14)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=1.5)
    ax.set_axisbelow(True)
    
    # 添加参考线
    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=2, alpha=0.5, label='90% Accuracy')
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=14, framealpha=0.9, ncol=2)
    
    # 调整布局
    plt.tight_layout()
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n准确率vs样本数量图已保存至: {save_path}")
    
    # 显示图表
    plt.show()

def print_summary_table(model_data):
    """打印数据摘要表格"""
    print("\n" + "=" * 70)
    print("模型准确率随样本数量变化统计表")
    print("=" * 70)
    print(f"{'模型类型':<15} {'50样本':>12} {'100样本':>12} {'150样本':>12} {'增长率':>12}")
    print("-" * 70)
    
    for model_type in sorted(model_data.keys()):
        sample_acc = model_data[model_type]
        acc_50 = sample_acc.get(50, 0)
        acc_100 = sample_acc.get(100, 0)
        acc_150 = sample_acc.get(150, 0)
        
        # 计算从50到150的增长率
        if acc_50 > 0:
            growth = ((acc_150 - acc_50) / acc_50) * 100
            growth_str = f"+{growth:.1f}%"
        else:
            growth_str = "N/A"
        
        print(f"{model_type.upper():<15} {acc_50:>11.4f} {acc_100:>11.4f} {acc_150:>11.4f} {growth_str:>12}")
    
    print("=" * 70)

def main():
    print("=" * 70)
    print("收集各模型在不同样本数量下的准确率数据...")
    print("=" * 70)
    
    # 收集数据
    model_data = collect_accuracies_by_samples()
    
    if not model_data:
        print("未找到足够的模型结果数据！")
        return
    
    print(f"\n共找到 {len(model_data)} 个模型类型的数据")
    
    # 打印摘要表格
    print_summary_table(model_data)
    
    # 绘制图表
    print("\n生成准确率vs样本数量趋势图...")
    plot_accuracy_vs_samples(model_data)
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
