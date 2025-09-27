```
①　激活conda环境：执行命令“conda activate pytorch”
②　进入代码文件主目录：执行命令“cd PatchTST_self_supervised/”
③　修改数据集为真实数据集source_domain：修改测试指令中的参数--dset为“--dset source_domain”
④　设置测试模式为离线测试：测试指令中的增加参数“--is_linear_probe 1”
⑤　训练离线对手模型，执行命令：
python patchtst_finetune.py --dset source_domain --is_linear_probe 1
⑥　测试调整前在目标域s0.3548_m907数据集上进行测试，计算MSE：执行命令：
python patchtst_finetune.py --is_test 1 --dset s0.3548_m907 
⑦　查看模型调整前在线数据预测结果：双击打开保存的s0.3548_m907_patchtst_finetuned_cw100_tw100_patch100_stride100_epochs-finetune20_model1_acc_last_3dim.csv
⑧　设置测试模式为在线调整：测试指令中的增加参数“--is_finetune 1”
⑨　使用数据增强后的数据集，对预训练模型进行微调：执行命令
python patchtst_finetune.py --is_finetune 1 --dset s0.3548_m907 --data --dataset_augmented 1
⑩　查看模型收敛轮次结果
s0.3548_m907_patchtst_finetuned_cw100_tw100_patch20_stride20_epochs-finetune50_model1_losses.csv
s0.3548_m907_patchtst_finetuned_cw100_tw100_patch20_stride20_epochs-finetune50_model1_losses.png
⑪　将增强后的模型在仅目标域s0.3548_m907数据集上进行测试，计算MSE：执行命令
python patchtst_finetune.py --is_test 2 --dset s0.3548_m907
⑫　查看模型调整后在线数据预测结果：双击打开保存的s0.3548_m907_patchtst_finetuned_cw100_tw100_patch20_stride20_epochs-finetune20_model1_acc_last_3dim.csv
计算模型在线调整前后预测准确率提升度，计算公式为（调整前MSE均值-调整后MSE均值）/调整前MSE均值*100%
```