#!/usr/bin/env python3
"""
测试脚本：验证数据增强效果
"""

import numpy as np
import matplotlib.pyplot as plt
from trajectory_augmentation import TrajectoryAugmentation

def test_augmentation_methods():
    """测试各种增强方法的效果"""
    
    # 创建测试数据：一个简单的正弦波
    t = np.linspace(0, 4*np.pi, 100)
    test_data = np.column_stack([
        np.sin(t),  # 第一个特征：正弦波
        np.cos(t),  # 第二个特征：余弦波
        t/10        # 第三个特征：线性增长
    ])
    
    # 初始化增强器
    augmenter = TrajectoryAugmentation()
    
    # 测试各种增强方法
    methods = ['time_warp', 'magnitude_warp', 'jittering']
    method_params = {
        'time_warp': {'sigma': 0.1, 'knot': 4},
        'magnitude_warp': {'sigma': 0.1, 'knot': 4},
        'jittering': {'sigma': 0.01}
    }
    
    # 生成增强结果
    individual_results = {}
    for method in methods:
        augmented_data = augmenter.methods[method](test_data.copy(), **method_params[method])
        individual_results[method] = augmented_data
    
    # 生成组合增强结果
    combined_result = augmenter.combined_augmentation(test_data.copy(), methods, **method_params)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('数据增强效果测试', fontsize=16, fontweight='bold')
    
    # 显示第一个特征
    feature_idx = 0
    time_steps = np.arange(len(test_data))
    
    # 原始数据
    axes[0, 0].plot(time_steps, test_data[:, feature_idx], 'b-', label='原始数据', linewidth=2)
    axes[0, 0].set_title('原始数据')
    axes[0, 0].set_xlabel('时间步')
    axes[0, 0].set_ylabel('数值')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 单独增强结果
    for i, method in enumerate(methods):
        row = (i + 1) // 2
        col = (i + 1) % 2
        if row < 2 and col < 2:
            axes[row, col].plot(time_steps, test_data[:, feature_idx], 'b-', label='原始数据', linewidth=2)
            axes[row, col].plot(time_steps, individual_results[method][:, feature_idx], 'r-', label='增强数据', linewidth=2)
            axes[row, col].set_title(f'{method}')
            axes[row, col].set_xlabel('时间步')
            axes[row, col].set_ylabel('数值')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()
    
    # 组合增强结果
    axes[1, 1].plot(time_steps, test_data[:, feature_idx], 'b-', label='原始数据', linewidth=2)
    axes[1, 1].plot(time_steps, combined_result[:, feature_idx], 'r-', label='组合增强数据', linewidth=2)
    axes[1, 1].set_title('组合增强')
    axes[1, 1].set_xlabel('时间步')
    axes[1, 1].set_ylabel('数值')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('test_augmentation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("测试完成！结果已保存为 test_augmentation_results.png")
    
    # 打印统计信息
    print("\n数据统计信息：")
    print(f"原始数据形状: {test_data.shape}")
    print(f"原始数据范围: [{test_data.min():.3f}, {test_data.max():.3f}]")
    
    for method in methods:
        data = individual_results[method]
        print(f"{method} 数据范围: [{data.min():.3f}, {data.max():.3f}]")
    
    print(f"组合增强数据范围: [{combined_result.min():.3f}, {combined_result.max():.3f}]")

if __name__ == "__main__":
    test_augmentation_methods()
