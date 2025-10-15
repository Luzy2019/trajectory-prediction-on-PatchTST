# trajectory-on-PatchTST

## 一、执行脚本

- pre-train script 预训练脚本

```sh
python patchtst_pretrain.py --dset source_domain
```

- linear-probe (base-line)

```sh
python patchtst_finetune.py --dset <target_domain_name> --is_linear_probe 1

# examples
python patchtst_finetune.py --dset source_domain --is_linear_probe 1
```

- fine-tuning 微调

```sh
python patchtst_finetune.py --dset <target_domain_name> --is_finetune 1 --dataset_augmented 1

# examples
python patchtst_finetune.py --dset CAV-H --is_finetune 1 --dataset_augmented 1
python patchtst_finetune.py --dset s0.3548_m907 --is_finetune 1 --dataset_augmented 1
python patchtst_finetune.py --dset HTV2 --is_finetune 1 --dataset_augmented 1

# partial parameters finetune
python patchtst_finetune.py --dset <target_domain_name> --is_finetune 1 --dataset_augmented 1 --partial_freeze <1|2|3>

# examples
python patchtst_finetune.py --dset CAV-H --is_finetune 1 --dataset_augmented 1 --partial_freeze 3
python patchtst_finetune.py --dset s0.3548_m907 --is_finetune 1 --dataset_augmented 1 --partial_freeze 3
python patchtst_finetune.py --dset HTV2 --is_finetune 1 --dataset_augmented 1 --partial_freeze 3
```

- test 测试

```sh
python patchtst_finetune.py --is_test [1|2] --dset <target_domain_name> 

# examples
# 测试 base-line
python patchtst_finetune.py --is_test 1 --dset s0.3548_m907
python patchtst_finetune.py --is_test 1 --dset HTV2
python patchtst_finetune.py --is_test 1 --dset CAV-H

# 测试 fine-tune
python patchtst_finetune.py --is_test 2 --dset CAV-H
python patchtst_finetune.py --is_test 2 --dset s0.3548_m907
python patchtst_finetune.py --is_test 2 --dset HTV2

# 测试 陪试模型
python patchtst_finetune.py --is_test 3 --dset CAV-H
python patchtst_finetune.py --is_test 3 --dset s0.3548_m907
python patchtst_finetune.py --is_test 3 --dset HTV2
```

## 二、结果保存

> 以 `CAV-H为例`
> 
> - `base_dir`: saved_results\\CAV-H\\20552\\CAV-H_patchtst_finetuned_cw100_tw100_patch20_stride20_epochs-finetune20_model1_acc_last_3dim.csv.csv

1. acc_last_3dim.csv

每条记录后3维平均mse，所有记录后3维平均mse
```csv
mse,average
0.043750,0.346588
0.044433,
0.045683,
0.050517,
0.054380,
...
```

3. w_distance_9dim.csv

整体的w-distance以及9个feature的各自w-distance

## 三、模型保存

> 以 `CAV-H为例`
> 
> - `base_dir`: saved_models\CAV-H\xxx.pth
> - 文件路径中包含fine-tuned为整体微调的结果以及模型


> 源域预训练的模型保存在
>`saved_models\source_domain\patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth`


# 项目操作步骤
查看`README_operation.md`
