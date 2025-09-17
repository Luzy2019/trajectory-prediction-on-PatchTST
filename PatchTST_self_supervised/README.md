# trajectory-on-PatchTST

## 一、执行脚本

- pre-train script 预训练脚本

```sh
python patchtst_pretrain.py --dset source_domain
```

- linear-probe script 仅head参数微调

```sh
python patchtst_finetune.py --dset <target_domain_name> --is_finetune 1 --pretrained_model <module_path>

# examples
python patchtst_finetune.py --dset CAV-H --is_linear_probe 1 --pretrained_mode saved_models/source_domain/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth

python patchtst_finetune.py --dset s0.3548_m907 --is_linear_probe 1 --pretrained_mode saved_models/source_domain/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth

python patchtst_finetune.py --dset HTV2 --is_linear_probe 1 --pretrained_mode saved_models/source_domain/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth

python patchtst_finetune.py --dset source_domain --is_linear_probe 1 --pretrained_mode saved_models/source_domain/masked_patchtst/based_model/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
```

- fine-tuning script 全部参数微调

```sh
python patchtst_finetune.py --dset <target_domain_name> --is_finetune 1 --pretrained_model <module_path>

# examples
python patchtst_finetune.py --dset CAV-H --is_finetune 1 --pretrained_mode saved_models/source_domain/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
python patchtst_finetune.py --dset s0.3548_m907 --is_finetune 1 --pretrained_mode saved_models/source_domain/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
python patchtst_finetune.py --dset HTV2 --is_finetune 1 --pretrained_mode saved_models/source_domain/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
python patchtst_finetune.py --dset processed_data_55 --is_finetune 1 --pretrained_mode saved_models/source_domain/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth
```

- only test script 仅仅用于测试

```sh
python patchtst_finetune.py --dset <target_domain_name> --pretrained_model <module_path> --patch_len <patch_len> --stride <stride> --dataset_size <dataset_size>

# examples
# 测试 fine-tune
python patchtst_finetune.py --dset s0.3548_m907 --patch_len 20 --stride 20 --dataset_size 5138
python patchtst_finetune.py --dset HTV2 --patch_len 20 --stride 20 --dataset_size 5138
python patchtst_finetune.py --dset CAV-H --patch_len 20 --stride 20 --dataset_size 5138

# 测试linear-probe
python patchtst_finetune.py --dset CAV-H --pretrained_model saved_models/source_domain/source_domain_patchtst_linear-probe_cw100_tw100_patch100_stride100_epochs-finetune20_model1
python patchtst_finetune.py --dset s0.3548_m907 --pretrained_model saved_models/source_domain/source_domain_patchtst_linear-probe_cw100_tw100_patch100_stride100_epochs-finetune20_model1
python patchtst_finetune.py --dset HTV2 --pretrained_model saved_models/source_domain/source_domain_patchtst_linear-probe_cw100_tw100_patch100_stride100_epochs-finetune20_model1
```

## 二、结果保存

> 以 `processed_data_55为例`
> 
> - `base_dir`: saved_results\\CAV-H\\[dataset_size]\\xxx.csv

1. acc_9dim.csv

微调后，9个feature的各自mse,mae结果，用于计算后三维的mse,mae

2. acc.csv

微调后，整体mse,mae结果

3. w_distance_9dim.csv

微调后，整体的w-distance以及9个feature的各自w-distance

## 三、模型保存

> 以 `CAV-H为例`
> 
> - `base_dir`: saved_models\CAV-H\xxx.pth
> - 文件路径中包含fine-tuned为整体微调的结果以及模型
> - 文件路径中包含linear-probe为仅微调head的结果以及模型


> 源域预训练的模型保存在
>`saved_models\source_domain\patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth`


# 项目操作步骤
查看`operation.md`
