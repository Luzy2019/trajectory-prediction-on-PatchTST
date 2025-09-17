## 操作步骤

### 1. 安装依赖
```sh
pip install -r requirement.txt
```

### 2. 操作流程

1. 清空历史记录
```sh
rm -r ./saved_results/
```

2. 进入工作目录
```sh
cd ./PatchTST_self_supervised/
```

3. 测试baseline模型在`CAV-H`,`s0.3548_m907`,`HTV2`数据集上的结果
```sh
python patchtst_finetune.py --dset CAV-H --pretrained_model saved_models/source_domain/source_domain_patchtst_linear-probe_cw100_tw100_patch100_stride100_epochs-finetune20_model1

python patchtst_finetune.py --dset s0.3548_m907 --pretrained_model saved_models/source_domain/source_domain_patchtst_linear-probe_cw100_tw100_patch100_stride100_epochs-finetune20_model1

python patchtst_finetune.py --dset HTV2 --pretrained_model saved_models/source_domain/source_domain_patchtst_linear-probe_cw100_tw100_patch100_stride100_epochs-finetune20_model1
```

4. 查看记录
`saved_results/CAV-H/5138/CAV-H_patchtst_finetuned_cw100_tw100_patch100_stride100_epochs-finetune20_model1_acc_9dim.csv`

5. 测试fine-tune模型在`CAV-H`,`s0.3548_m907`,`HTV2`数据集上的结果
```sh
python patchtst_finetune.py --dset CAV-H --patch_len 20 --stride 20
python patchtst_finetune.py --dset s0.3548_m907 --patch_len 20 --stride 20
python patchtst_finetune.py --dset HTV2 --patch_len 20 --stride 20
```

6. 查看记录
`saved_results/CAV-H/5138/CAV-H_patchtst_finetuned_cw100_tw100_patch20_stride20_epochs-finetune20_model1_acc_9dim.csv`