## 数据增强处理

### 创建conda环境

```sh
conda create --name data-gan python=3.6.4

conda activate data-gan
```

### 安装依赖

```sh
pip install -r requirement.txt
```

### 配置文件

```sh
# make.sh
INPUT_DIR="ORIGIN" # 原数据文件目录
BASE_OUTPUT_DIR="ORIGIN-MAKE" # 生成数据文件目录
METHODS=("time_shift" "jittering" "time_warp" "magnitude_warp" "random_scaling" "noise_injection" "window_slice" "smoothing" "permutation") # 数据增强方法
```

### 执行脚本
```sh
# 执行数据增强脚本
sh make.sh
```

