__all__ = ["input_dir", "base_output_dir", "methods", "method_params"]

# 输入文件目录
input_dir="ORIGIN"

# 生成文件目录
base_output_dir="ORIGIN-MAKE"

# 使用数据增强方式
methods = ["time_warp", "magnitude_warp", "jittering"]

# 数据增强方法参数
method_params = {
    # 1. 时间扭曲 (time_warp)
    # - sigma (float, 默认: 0.1)：控制时间扭曲的强度，值越大时间轴扭曲越明显
    # - knot (int, 默认: 4)：插值节点数量，影响时间扭曲的平滑度，节点越多扭曲越平滑
    "time_warp": {"sigma": 0.1, "knot": 4},

    # 2. 幅度扭曲 (magnitude_warp)
    # - sigma (float, 默认: 0.1)：控制幅度扭曲的强度，值越大数值变化越明显
    # - knot (int, 默认: 4)：插值节点数量，影响幅度扭曲的平滑度
    "magnitude_warp": {"sigma": 0.1, "knot": 4},

    # 3. 抖动增强 (jittering)
    # - sigma (float, 默认: 0.01)：控制添加的高斯噪声标准差，值越大抖动越明显
    "jittering": {"sigma": 0.01},

    # 4. 时间偏移 (time_shift)
    # - shift_range (tuple, 默认: (-10, 10))：时间偏移的范围，单位为时间步长
    "time_shift": {"shift_range": (-10, 10)},

    # 5. 噪声注入 (noise_injection)
    # - scale (float, 默认: 0.005)：噪声的缩放因子，控制注入噪声的强度
    "noise_injection": {"scale": 0.005},

    # 6. 随机缩放 (random_scaling)
    # - features_to_scale (list, 默认: None)：指定需要缩放的特征列，None表示缩放所有特征
    # - min_scale (float, 默认: 0.95)：最小缩放因子
    # - max_scale (float, 默认: 1.05)：最大缩放因子
    "random_scaling":{"features_to_scale": None, "min_scale": 0.95, "max_scale": 1.05},

    # 7. 窗口切片 (window_slice)
    # - reduce_ratio (float, 默认: 0.95)：数据保留比例，0.95表示保留95%的数据
    "window_slice": {"reduce_ratio": 0.95},

    # 8. 平滑处理 (smoothing)
    # - window_length (int, 默认: 11)：滑动窗口长度，必须为奇数
    # - polyorder (int, 默认: 3)：多项式拟合的阶数，必须小于window_length
    "smoothing": {"window_length": 11, "polyorder": 3},

    # 9. 排列增强 (permutation)
    #   - max_segments (int, 默认: 3)：最大分段数量，控制数据重排的复杂度
    "permutation": {"max_segments": 3},
}