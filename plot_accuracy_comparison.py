#!/usr/bin/env python3
"""
绘制所有模型的测试准确率对比图
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def collect_accuracies():
    """收集所有模型的测试准确率"""
    results_dir = Path('results')
    model_accuracies = {}
    
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
        if 'test_results' in data and 'accuracy' in data['test_results']:
            model_name = model_dir.name
            accuracy = data['test_results']['accuracy']
            model_accuracies[model_name] = accuracy
            print(f"{model_name}: {accuracy:.4f}")
    
    return model_accuracies

def plot_accuracy_comparison(model_accuracies, save_path='reports/figures/comparisons/accuracy_comparison.png'):
    """绘制准确率对比条形图"""
    # 按准确率排序
    sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
    models = [item[0] for item in sorted_models]
    accuracies = [item[1] for item in sorted_models]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制条形图
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax.bar(range(len(models)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 在条形上方添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 设置标签和标题
    ax.set_xlabel('模型', fontsize=12, fontweight='bold')
    ax.set_ylabel('测试准确率 (Test Accuracy)', fontsize=12, fontweight='bold')
    ax.set_title('各模型测试准确率对比\nModel Test Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    
    # 设置x轴刻度
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    # 设置y轴范围
    ax.set_ylim([0, 1.05])
    
    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # 添加参考线 (如90%准确率线)
    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='90% 准确率')
    ax.legend(loc='lower right')
    
    # 调整布局
    plt.tight_layout()
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n准确率对比图已保存至: {save_path}")
    
    # 显示图表
    plt.show()

def plot_detailed_comparison(model_accuracies, save_path='reports/figures/comparisons/accuracy_detailed.png'):
    """绘制带有训练集数量信息的详细对比"""
    # 按数据集分组
    dataset_groups = {'50': [], '100': [], '150': [], 'other': []}
    
    for model, acc in model_accuracies.items():
        if '_50' in model:
            dataset_groups['50'].append((model, acc))
        elif '_100' in model:
            dataset_groups['100'].append((model, acc))
        elif '_150' in model:
            dataset_groups['150'].append((model, acc))
        else:
            dataset_groups['other'].append((model, acc))
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('各模型在不同数据集规模下的准确率对比', fontsize=16, fontweight='bold', y=0.995)
    
    dataset_titles = {
        '50': '50样本数据集',
        '100': '100样本数据集',
        '150': '150样本数据集',
        'other': '其他数据集'
    }
    
    for idx, (dataset_size, ax) in enumerate(zip(['50', '100', '150', 'other'], axes.flatten())):
        group = dataset_groups[dataset_size]
        
        if not group:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(dataset_titles[dataset_size])
            continue
        
        # 排序
        group = sorted(group, key=lambda x: x[1], reverse=True)
        models = [item[0].replace(f'_{dataset_size}', '') for item in group]
        accuracies = [item[1] for item in group]
        
        # 绘制条形图
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = ax.bar(range(len(models)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 设置标签
        ax.set_title(dataset_titles[dataset_size], fontsize=12, fontweight='bold')
        ax.set_ylabel('准确率', fontsize=10)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # 添加90%参考线
        ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1, alpha=0.4)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"详细对比图已保存至: {save_path}")
    
    plt.show()

def main():
    print("=" * 60)
    print("收集所有模型的测试准确率...")
    print("=" * 60)
    
    # 收集准确率数据
    model_accuracies = collect_accuracies()
    
    if not model_accuracies:
        print("未找到任何模型结果！")
        return
    
    print(f"\n共找到 {len(model_accuracies)} 个模型的结果")
    print("=" * 60)
    
    # 绘制对比图
    print("\n生成准确率对比图...")
    plot_accuracy_comparison(model_accuracies)
    
    print("\n生成详细对比图（按数据集分组）...")
    plot_detailed_comparison(model_accuracies)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
