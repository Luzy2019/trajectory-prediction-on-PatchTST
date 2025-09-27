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
python patchtst_finetune.py --is_test 1 --dset CAV-H
python patchtst_finetune.py --is_test 1 --dset s0.3548_m907
python patchtst_finetune.py --is_test 1 --dset HTV2
```

4. 查看记录
`saved_results/CAV-H/20552/CAV-H_patchtst_finetuned_cw100_tw100_patch100_stride100_epochs-finetune20_model1_acc_last_3dim.csv`

5. 测试fine-tune模型在`CAV-H`,`s0.3548_m907`,`HTV2`数据集上的结果
```sh
python patchtst_finetune.py --is_test 2 --dset CAV-H
python patchtst_finetune.py --is_test 2 --dset s0.3548_m907
python patchtst_finetune.py --is_test 2 --dset HTV2
```

6. 查看记录
`saved_results/CAV-H/20552/CAV-H_patchtst_finetuned_cw100_tw100_patch20_stride20_epochs-finetune20_model1_acc_last_3dim.csv`