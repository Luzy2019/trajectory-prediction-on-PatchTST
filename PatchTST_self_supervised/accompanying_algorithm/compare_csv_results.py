#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV数据对比脚本
对比两个CSV文件中的MSE和average数据
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_csv_data(file_path):
    """加载CSV文件数据"""
    try:
        df = pd.read_csv(file_path, header=0)
        print(f"成功加载文件: {file_path}")
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def plot_comparison(data1, data2, label1, label2, save_path=None):
    """绘制对比图"""
    # 创建子图 - 左右布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 定义颜色，确保两个子图中相同标签使用相同颜色
    color1 = 'blue'  # label1的颜色
    color2 = 'red'   # label2的颜色
    
    # 获取数据长度
    min_len = min(len(data1), len(data2))
    # 第一个子图：MSE对比
    ax1.plot(range(min_len), data1[:min_len], label=label1, linewidth=2, alpha=0.8, color=color1)
    ax1.plot(range(min_len), data2[:min_len], label=label2, linewidth=2, alpha=0.8, color=color2)
    ax1.set_title('CAV-H 多轮预测MSE值对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('步数', fontsize=12)
    ax1.set_ylabel('MSE值', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 第二个子图：Average对比
    # 检查是否有average列数据
    if len(data1) > 0 and len(data2) > 0:
        # 对于average数据，绘制水平线，使用相同的颜色
        ax2.axhline(y=data1[0], label=label1, linewidth=3, alpha=0.8, color=color1)
        ax2.axhline(y=data2[0], label=label2, linewidth=3, alpha=0.8, color=color2)
        
        # 在水平线上添加数值标签
        ax2.text(0.1, data1[0], f'{data1[0]:.6f}', fontsize=10, color=color1, 
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        ax2.text(0.1, data2[0], f'{data2[0]:.6f}', fontsize=10, color=color2, 
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax2.set_title('CAV-H 多轮预测Average值对比', fontsize=14, fontweight='bold')
        ax2.set_xlabel('', fontsize=12)
        ax2.set_ylabel('Average值', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        # 设置x轴范围，让水平线更明显
        ax2.set_xlim(-0.5, 0.5)
        # 隐藏x轴刻度
        ax2.set_xticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()

def plot_comparison_with_average(mse1, mse2, avg1, avg2, label1, label2, save_path=None):
    """绘制MSE和Average对比图"""
    # 创建子图 - 左右布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 定义颜色，确保两个子图中相同标签使用相同颜色
    color1 = 'blue'  # label1的颜色
    color2 = 'red'   # label2的颜色
    
    # 获取MSE数据长度
    min_len = min(len(mse1), len(mse2))
    
    # 第一个子图：MSE对比
    ax1.plot(range(min_len), mse1[:min_len], label=label1, linewidth=2, alpha=0.8, color=color1)
    ax1.plot(range(min_len), mse2[:min_len], label=label2, linewidth=2, alpha=0.8, color=color2)
    ax1.set_title('CAV-H 多轮预测MSE值对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('步数', fontsize=12)
    ax1.set_ylabel('MSE值', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 第二个子图：Average对比
    # 检查是否有average列数据
    if avg1 is not None and avg2 is not None and len(avg1) > 0 and len(avg2) > 0:
        # 对于average数据，绘制水平线，使用相同的颜色
        ax2.axhline(y=avg1[0], label=label1, linewidth=3, alpha=0.8, color=color1)
        ax2.axhline(y=avg2[0], label=label2, linewidth=3, alpha=0.8, color=color2)
        
        # 在水平线上添加数值标签
        ax2.text(0.1, avg1[0], f'{avg1[0]:.6f}', fontsize=10, color=color1, 
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        ax2.text(0.1, avg2[0], f'{avg2[0]:.6f}', fontsize=10, color=color2, 
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax2.set_title('CAV-H 多轮预测Average值对比', fontsize=14, fontweight='bold')
        ax2.set_xlabel('', fontsize=12)
        ax2.set_ylabel('Average值', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        # 设置x轴范围，让水平线更明显
        ax2.set_xlim(-0.5, 0.5)
        # 隐藏x轴刻度
        ax2.set_xticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    # 文件路径
    file1_path = "../saved_results/CAV-H/20552/CAV-H_patchtst_finetuned_cw100_tw100_patch20_stride20_epochs-finetune20_model1_acc_last_3dim_.csv"
    file2_path = "../saved_results/CAV-H/20552/CAV-H_patchtst_finetuned_cw100_tw100_patch20_stride20_epochs-finetune20_model1_acc_last_3dim.csv"
    
    # 检查文件是否存在
    if not os.path.exists(file1_path):
        print(f"文件不存在: {file1_path}")
        return
    if not os.path.exists(file2_path):
        print(f"文件不存在: {file2_path}")
        return
    
    # 加载数据
    print("正在加载数据...")
    df1 = load_csv_data(file1_path)
    df2 = load_csv_data(file2_path)
    
    if df1 is None or df2 is None:
        print("数据加载失败，程序退出")
        return
    
    # 提取MSE列数据（第一列）
    mse1 = df1.iloc[:, 0].values
    mse2 = df2.iloc[:, 0].values
    
    # 提取average列数据（第二列，如果存在且不为空）
    avg1 = None
    avg2 = None
    
    if df1.shape[1] > 1:
        avg1 = df1.iloc[:, 1].values
        # 过滤掉空值
        avg1 = avg1[~pd.isna(avg1)]
    
    if df2.shape[1] > 1:
        avg2 = df2.iloc[:, 1].values
        # 过滤掉空值
        avg2 = avg2[~pd.isna(avg2)]
    
    # 设置标签
    label1 = "Fine-tuned"
    label2 = "Partial-Fine-tuned"
    
    print(f"\n数据统计:")
    print(f"文件1 MSE数据长度: {len(mse1)}")
    print(f"文件2 MSE数据长度: {len(mse2)}")
    if avg1 is not None:
        print(f"文件1 Average数据长度: {len(avg1)}")
    if avg2 is not None:
        print(f"文件2 Average数据长度: {len(avg2)}")
    
    # 绘制合并的对比图（MSE + Average）
    print("\n正在绘制合并对比图...")
    plot_comparison_with_average(mse1, mse2, avg1, avg2, label1, label2, "combined_comparison.png")
    
    # 计算统计信息
    print(f"\n统计信息:")
    print(f"MSE1 - 最小值: {np.min(mse1):.6f}, 最大值: {np.max(mse1):.6f}, 平均值: {np.mean(mse1):.6f}")
    print(f"MSE2 - 最小值: {np.min(mse2):.6f}, 最大值: {np.max(mse2):.6f}, 平均值: {np.mean(mse2):.6f}")
    
    if avg1 is not None and len(avg1) > 0:
        print(f"Average1 - 最小值: {np.min(avg1):.6f}, 最大值: {np.max(avg1):.6f}, 平均值: {np.mean(avg1):.6f}")
    if avg2 is not None and len(avg2) > 0:
        print(f"Average2 - 最小值: {np.min(avg2):.6f}, 最大值: {np.max(avg2):.6f}, 平均值: {np.mean(avg2):.6f}")

if __name__ == "__main__":
    main()
