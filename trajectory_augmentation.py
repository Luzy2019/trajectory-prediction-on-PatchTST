import os
import glob
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import argparse
from config import input_dir, base_output_dir, methods, method_params

# 配置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class TrajectoryAugmentation:
    def __init__(self):
        """初始化增强器"""
        self.methods = {
            'time_shift': self.time_shift,
            'noise_injection': self.noise_injection,
            'random_scaling': self.random_scaling,
            'jittering': self.jittering,
            'time_warp': self.time_warp,
            'magnitude_warp': self.magnitude_warp,
            'window_slice': self.window_slice,
            'smoothing': self.smoothing,
            'permutation': self.permutation
        }

    def process_directory(self, input_dir, output_dir, method_configs, n_augmentations=1, visualize=False):
        # 创建输出目录（如果不存在）
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 获取所有轨迹文件
        trajectory_files = glob.glob(os.path.join(input_dir, "*.txt"))

        if not trajectory_files:
            print(f"在 {input_dir} 中未找到任何TXT文件")
            return

        print(f"找到 {len(trajectory_files)} 个轨迹文件，开始处理...")

        # 处理每个文件
        for file_path in tqdm(trajectory_files, desc="处理文件"):
            try:
                # 加载轨迹数据
                trajectory = self.load_trajectory(file_path)

                # 为可视化准备原始数据
                original_data = trajectory.copy()

                # 执行数据增强，生成n_augmentations个增强版本
                for aug_idx in range(n_augmentations):
                    # 为每个增强版本随机选择一个方法
                    method_name, params = method_configs[np.random.randint(0, len(method_configs))]

                    # 应用所选增强方法
                    if method_name in self.methods:
                        augmented_data = self.methods[method_name](trajectory.copy(), **params)

                        # if trajectory.shape[1] > 6:
                        #     augmented_data[:, 6:] = trajectory[:, 6:]
                        
                        # 构建输出文件名
                        file_name = os.path.basename(file_path)
                        base_name = os.path.splitext(file_name)[0]
                        output_file = os.path.join(output_dir, f"{base_name}_aug_{method_name}_{aug_idx + 1}.txt")

                        # 保存增强后的轨迹
                        self.save_trajectory(augmented_data, output_file)

                        # 可视化（如果需要）
                        if visualize and aug_idx == 0:  # 仅为第一个增强版本生成可视化
                            self.visualize_augmentation(original_data, augmented_data, method_name,
                                                        output_dir, base_name)
                    else:
                        print(f"警告: 未知的增强方法 '{method_name}'")

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")

    def load_trajectory(self, file_path):
        try:
            # 加载数据，假设是CSV格式，以逗号分隔
            data = np.loadtxt(file_path, delimiter=',')
            return data
        except Exception as e:
            raise Exception(f"加载轨迹文件时出错: {str(e)}")

    def save_trajectory(self, data, file_path):
        try:
            # 保存为CSV格式，以逗号分隔
            np.savetxt(file_path, data, delimiter=',', fmt='%.6f')
        except Exception as e:
            raise Exception(f"保存轨迹文件时出错: {str(e)}")

    def visualize_augmentation(self, original_data, augmented_data, method_name, output_dir, base_name):
        n_features = min(3, original_data.shape[1])  # 最多显示前3个特征

        fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
        if n_features == 1:
            axes = [axes]  # 确保axes是可迭代的

        time_steps = np.arange(len(original_data))

        for i in range(n_features):
            ax = axes[i]
            ax.plot(time_steps, original_data[:, i], 'b-', label='原始轨迹')
            ax.plot(time_steps, augmented_data[:, i], 'r-', label='增强轨迹')
            ax.set_title(f'特征 {i + 1} - 增强方法: {method_name}')
            ax.set_xlabel('时间步')
            ax.set_ylabel('值')
            ax.grid(True)
            ax.legend()

        plt.tight_layout()

        # 保存图表
        viz_path = os.path.join(output_dir, f"{base_name}_{method_name}_visualization.png")
        plt.savefig(viz_path)
        plt.close()

    def time_shift(self, data, shift_range=(-10, 10)):
        shift = np.random.randint(shift_range[0], shift_range[1] + 1)

        if shift == 0:
            return data  # 无需移动

        # 使用numpy的roll函数进行移动
        shifted_data = np.roll(data, shift, axis=0)

        # 处理边界情况
        if shift > 0:
            # 向后移动，前面的部分用第一个值填充
            shifted_data[:shift] = data[0]
        elif shift < 0:
            # 向前移动，后面的部分用最后一个值填充
            shifted_data[shift:] = data[-1]

        return shifted_data

    def noise_injection(self, data, scale=0.01):
        noise = np.random.normal(loc=0.0, scale=scale, size=data.shape)
        noisy_data = data + noise
        return noisy_data

    def random_scaling(self, data, features_to_scale=None, min_scale=0.9, max_scale=1.1):
        scaled_data = data.copy()

        if features_to_scale is None:
            features_to_scale = range(data.shape[1])

        # 为每个特征生成不同的缩放因子
        for feature_idx in features_to_scale:
            scale = np.random.uniform(min_scale, max_scale)
            scaled_data[:, feature_idx] = data[:, feature_idx] * scale

        return scaled_data

    def jittering(self, data, sigma=0.03):
        n_timesteps, n_features = data.shape
        augmented_data = data.copy()

        # 生成连续的随机扰动
        perturbation = np.zeros((n_timesteps, n_features))
        perturbation[0] = np.random.normal(0, sigma, n_features)

        for t in range(1, n_timesteps):
            # 每个时间点的扰动都与前一个时间点有关联
            perturbation[t] = 0.9 * perturbation[t - 1] + 0.1 * np.random.normal(0, sigma, n_features)

        augmented_data += perturbation

        return augmented_data

    def time_warp(self, data, sigma=0.2, knot=4):
        n_timesteps, n_features = data.shape

        # 创建时间扭曲映射
        orig_steps = np.arange(n_timesteps)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2))
        warp_steps = np.linspace(0, n_timesteps - 1, num=knot + 2)
        warper = interp1d(warp_steps, warp_steps * random_warps, kind='linear')

        # 应用扭曲
        new_steps = warper(orig_steps)
        new_steps = np.clip(new_steps, 0, n_timesteps - 1)

        # 对每个特征进行插值
        warped_data = np.zeros_like(data)
        for feature in range(n_features):
            time_series = data[:, feature]
            warper_feature = interp1d(orig_steps, time_series, kind='linear', bounds_error=False,
                                      fill_value="extrapolate")
            warped_data[:, feature] = warper_feature(new_steps)

        return warped_data

    def magnitude_warp(self, data, sigma=0.2, knot=4):
        n_timesteps, n_features = data.shape

        # 创建幅度扭曲函数
        steps = np.arange(n_timesteps)
        knot_points = np.linspace(0, n_timesteps - 1, num=knot + 2)

        # 每个特征使用不同的扭曲函数
        warped_data = data.copy()

        for feature in range(n_features):
            magnitudes = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2))
            magnitude_function = interp1d(knot_points, magnitudes, kind='linear')

            # 应用扭曲
            warp_factors = magnitude_function(steps)
            warped_data[:, feature] = data[:, feature] * warp_factors

        return warped_data

    def window_slice(self, data, reduce_ratio=0.9):
        n_timesteps, n_features = data.shape
        window_size = int(n_timesteps * reduce_ratio)

        # 为轨迹选择随机的起始位置
        start_idx = np.random.randint(0, n_timesteps - window_size + 1)
        window = data[start_idx:start_idx + window_size]

        # 将窗口数据重新采样到原始长度
        augmented_data = np.zeros_like(data)
        for feature in range(n_features):
            feature_series = window[:, feature]
            x_original = np.linspace(0, 1, window_size)
            x_new = np.linspace(0, 1, n_timesteps)
            interpolator = interp1d(x_original, feature_series, kind='linear')
            augmented_data[:, feature] = interpolator(x_new)

        return augmented_data

    def smoothing(self, data, window_length=11, polyorder=3):
        if window_length % 2 == 0:
            window_length += 1  # 确保窗口长度为奇数

        n_timesteps, n_features = data.shape
        smoothed_data = np.zeros_like(data)

        for feature in range(n_features):
            feature_series = data[:, feature]
            try:
                smoothed_data[:, feature] = savgol_filter(feature_series, window_length, polyorder)
            except:
                # 如果滤波失败，保留原始数据
                smoothed_data[:, feature] = feature_series

        return smoothed_data

    def permutation(self, data, max_segments=5):
        n_timesteps, n_features = data.shape

        # 随机选择分段数量
        num_segments = np.random.randint(2, max_segments + 1)
        segment_points = np.linspace(0, n_timesteps, num_segments + 1).astype(int)

        # 创建分段
        segments = []
        for j in range(len(segment_points) - 1):
            start, end = segment_points[j], segment_points[j + 1]
            segments.append(data[start:end])

        # 随机打乱分段顺序
        np.random.shuffle(segments)

        # 重新组合分段
        augmented_data = np.vstack(segments)

        return augmented_data

    def combined_augmentation(self, data, methods=['time_warp', 'magnitude_warp', 'jittering'], **kwargs):
        """
        组合多种数据增强方法
        
        Args:
            data: 输入数据
            methods: 要应用的增强方法列表
            **kwargs: 各方法的参数
        
        Returns:
            增强后的数据
        """
        augmented_data = data.copy()
        
        # 依次应用每种增强方法
        for method in methods:
            if method in self.methods:
                # 获取该方法的参数
                method_params = kwargs.get(method, {})
                augmented_data = self.methods[method](augmented_data, **method_params)
            else:
                print(f"警告: 未知的增强方法 '{method}'")
        
        return augmented_data

    def process_individual_augmentations(self, input_dir, output_dir, methods=['time_warp', 'magnitude_warp', 'jittering'], 
                                       method_params=None, visualize=True):
        """
        对每个文件分别应用不同的增强方法
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            methods: 要应用的增强方法列表
            method_params: 各方法的参数字典
            visualize: 是否生成可视化
        """
        if method_params is None:
            method_params = {
                'time_warp': {'sigma': 0.1, 'knot': 4},
                'magnitude_warp': {'sigma': 0.1, 'knot': 4},
                'jittering': {'sigma': 0.01}
            }
        
        # 获取所有轨迹文件
        trajectory_files = glob.glob(os.path.join(input_dir, "*.txt"))
        
        if not trajectory_files:
            print(f"在 {input_dir} 中未找到任何TXT文件")
            return
        
        print(f"找到 {len(trajectory_files)} 个轨迹文件，开始单独增强处理...")
        
        # 处理每个文件
        for file_path in tqdm(trajectory_files, desc="处理文件"):
            try:
                # 加载轨迹数据
                trajectory = self.load_trajectory(file_path)
                file_name = os.path.basename(file_path)
                base_name = os.path.splitext(file_name)[0]
                
                # 对每种方法分别应用
                for method in methods:
                    if method in self.methods:
                        # 应用增强方法
                        augmented_data = self.methods[method](trajectory.copy(), **method_params.get(method, {}))
                        
                        # 创建输出目录
                        _output_dir = os.path.join(output_dir, method)
                        if not os.path.exists(_output_dir):
                            os.makedirs(_output_dir)

                        # 保存增强后的数据
                        output_file = os.path.join(_output_dir, f"{base_name}_aug_{method}_1.txt")
                        self.save_trajectory(augmented_data, output_file)
                        
                        # 生成可视化
                        if visualize:
                            self.visualize_individual_augmentation(trajectory, augmented_data, method, output_dir, base_name)
                    else:
                        print(f"警告: 未知的增强方法 '{method}'")
                        
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")

    def process_combined_augmentation(self, input_dir, output_dir, methods=['time_warp', 'magnitude_warp', 'jittering'],
                                    method_params=None, visualize=True):
        """
        对每个文件应用组合增强方法
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            methods: 要组合的增强方法列表
            method_params: 各方法的参数字典
            visualize: 是否生成可视化
        """
        if method_params is None:
            method_params = {
                'time_warp': {'sigma': 0.1, 'knot': 4},
                'magnitude_warp': {'sigma': 0.1, 'knot': 4},
                'jittering': {'sigma': 0.01}
            }
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 获取所有轨迹文件
        trajectory_files = glob.glob(os.path.join(input_dir, "*.txt"))
        
        if not trajectory_files:
            print(f"在 {input_dir} 中未找到任何TXT文件")
            return
        
        print(f"找到 {len(trajectory_files)} 个轨迹文件，开始组合增强处理...")
        
        # 处理每个文件
        for file_path in tqdm(trajectory_files, desc="处理文件"):
            try:
                # 加载轨迹数据
                trajectory = self.load_trajectory(file_path)
                file_name = os.path.basename(file_path)
                base_name = os.path.splitext(file_name)[0]
                
                # 应用组合增强
                combined_augmented_data = self.combined_augmentation(trajectory.copy(), methods, **method_params)
                
                # 保存增强后的数据
                output_file = os.path.join(output_dir, f"{base_name}_aug_combined_1.txt")
                self.save_trajectory(combined_augmented_data, output_file)
                
                # 生成可视化
                if visualize:
                    self.visualize_combined_augmentation(trajectory, combined_augmented_data, methods, output_dir, base_name)
                    
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")

    def visualize_individual_augmentation(self, original_data, augmented_data, method_name, output_dir, base_name):
        """可视化单个增强方法的效果"""
        n_features = min(9, original_data.shape[1])  # 最多显示前3个特征
        
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features))
        if n_features == 1:
            axes = [axes]
        
        time_steps = np.arange(len(original_data))
        
        for i in range(n_features):
            ax = axes[i]
            ax.plot(time_steps, original_data[:, i], 'b-', label='原始数据', linewidth=2)
            ax.plot(time_steps, augmented_data[:, i], 'r-', label='增强数据', linewidth=2)
            ax.set_title(f'特征 {i + 1} - 增强方法: {method_name}')
            ax.set_xlabel('时间步')
            ax.set_ylabel('数值')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        # 保存图表
        viz_path = os.path.join(output_dir, method_name, f"{base_name}_{method_name}_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_combined_augmentation(self, original_data, augmented_data, methods, output_dir, base_name):
        """可视化组合增强方法的效果"""
        n_features = min(3, original_data.shape[1])  # 最多显示前3个特征
        
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features))
        if n_features == 1:
            axes = [axes]
        
        time_steps = np.arange(len(original_data))
        method_str = '+'.join(methods)
        
        for i in range(n_features):
            ax = axes[i]
            ax.plot(time_steps, original_data[:, i], 'b-', label='原始数据', linewidth=2)
            ax.plot(time_steps, augmented_data[:, i], 'r-', label='组合增强数据', linewidth=2)
            ax.set_title(f'特征 {i + 1} - 组合增强方法: {method_str}')
            ax.set_xlabel('时间步')
            ax.set_ylabel('数值')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        # 保存图表
        viz_path = os.path.join(output_dir, f"{base_name}_combined_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description="战斗机轨迹数据增强工具")

    parser.add_argument("--input_dir", type=str, required=False,
                        help="包含轨迹TXT文件的输入目录")
    parser.add_argument("--output_dir", type=str, required=False,
                        help="保存增强数据的输出目录")
    parser.add_argument("--methods", type=str, nargs="+", default=["time_warp", "magnitude_warp", "jittering"],
                        help="要使用的增强方法，可选多个")
    parser.add_argument("--n_augmentations", type=int, default=1,
                        help="每个输入文件生成的增强文件数量")
    parser.add_argument("--visualize", action="store_true",
                        help="是否生成可视化结果")
    parser.add_argument("--mode", type=str, choices=["individual", "combined"], default="original",
                        help="增强模式: individual(单独增强), combined(组合增强)")

    args = parser.parse_args()

    # 初始化增强器
    augmenter = TrajectoryAugmentation()

    _input_dir = args.input_dir or input_dir
    _output_dir = args.output_dir or os.path.join(f"{base_output_dir}-{args.mode}")
    _methods =  args.methods or methods

    _method_params = method_params

    # 根据模式选择处理方式
    if args.mode == "individual":
        # 单独增强模式：分别应用time_warp, magnitude_warp, jittering
        print("执行单独数据增强模式...")
        augmenter.process_individual_augmentations(
            _input_dir,
            _output_dir,
            _methods,
            _method_params,
            args.visualize
        )
        
    elif args.mode == "combined":
        # 组合增强模式：同时应用三种方法
        print("执行组合数据增强模式...")
        augmenter.process_combined_augmentation(
            _input_dir,
            _output_dir,
            _methods,
            _method_params,
            args.visualize
        )

    else:
        raise Exception('输入参数有误，请查看参数')

    print(f"数据增强完成！增强后的轨迹保存在 {args.output_dir}")


if __name__ == "__main__":
    main()