import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import matplotlib.pyplot as plt

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.core import *
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *
from config import *

import argparse
parser = argparse.ArgumentParser()

# Pretraining and Finetuning
parser.add_argument('--is_finetune', type=int, default=0, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')
parser.add_argument('--is_test', type=int, default=0, help='use test data')
parser.add_argument('--dset_finetune', type=str, default='source_domain', help='data_set name')
parser.add_argument('--dataset_augmented', type=int, default=0, help='use dataset augmentation or not')
parser.add_argument('--partial_freeze', type=int, default=0, help='use dataset augmentation or not')

args = parser.parse_args()
if args.is_finetune: args = merge_namespaces(FINETUNE_CONFIG, args)
elif args.is_linear_probe: args = merge_namespaces(BASELINE_CONFIG, args)
elif args.is_test == 1: args = merge_namespaces(TEST_BL_CONFIG, args)
elif args.is_test == 2 or args.is_test == 3: args = merge_namespaces(TEST_FT_CONFIG, args) # 1 base-line 2 fine-tune
else: raise ValueError('Invalid argument')

print('args:', args)
args.model_path = 'accompanying_algorithm/saved_models/' + args.dset_finetune + '/' if (args.is_test == 3 or args.partial_freeze > 0) else 'saved_models/' + args.dset_finetune + '/'
if not os.path.exists(args.model_path): os.makedirs(args.model_path)
args.result_path = 'accompanying_algorithm/saved_results/' + args.dset_finetune + '/' + args.dataset_size + '/' if (args.is_test == 3 or args.partial_freeze > 0) else 'saved_results/' + args.dset_finetune + '/' + args.dataset_size + '/'
if not os.path.exists(args.result_path): os.makedirs(args.result_path)

# args.save_finetuned_model = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.finetuned_model_id)
suffix_name = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_model' + str(args.finetuned_model_id)

# 进行fine-tune 微调
if args.is_finetune: args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name
# 进行linear-probe 线性探测
elif args.is_linear_probe: args.save_finetuned_model = args.dset_finetune+'_patchtst_linear-probe'+suffix_name
# 其他
else: args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name

# get available GPU devide
set_device()

def get_model(c_in, args, head_type, weight_path=None):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)
    
    # get model
    model = PatchTST(c_in=c_in,
                target_dim=args.target_points,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,                        
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act='relu',
                head_type=head_type,
                res_attention=False
                )    
    if weight_path: model = transfer_weights(weight_path, model)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model



def find_lr(head_type):
    # get dataloader
    dls = get_dls(args)    
    model = get_model(dls.vars, args, head_type)
    # transfer weight
    # weight_path = args.model_path + args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
        
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=args.lr, 
                        cbs=cbs,
                        )                        
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr

def finetune_func(lr=args.lr):
    print('end-to-end finetuning')
    # get dataloader
    dls = get_dls(args)
    # get model 
    model = get_model(dls.vars, args, head_type='prediction')
    # transfer weight
    # weight_path = args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')   
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.model_path)
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse]
                        )                            
    # fit the data to the model
    #learn.fit_one_cycle(n_epochs=args.n_epochs_finetune, lr_max=lr)
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=10, partial_freeze=args.partial_freeze)
    save_recorders(learn)

def linear_probe_func(lr=args.lr):
    print('linear probing')
    # get dataloader
    dls = get_dls(args)
    # get model 
    model = get_model(dls.vars, args, head_type='prediction')
    # transfer weight
    # weight_path = args.model_path + args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')    
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.model_path)
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse]
                        )                            
    # fit the data to the model
    learn.linear_probe(n_epochs=args.n_epochs_finetune, base_lr=lr)
    # save_recorders(learn)

def save_format_result(out):
    # 输出每条记录后三维的平均mse和所有记录后三位平均mse
    # 将numpy数组转换为数值列表
    mse_values = [float(item[0]) for item in out[3]]  # 提取每个数组的第一个值
    average_value = float(np.array([item[0] for item in out[3]]).mean())  # 计算平均值并转换为单个数值
    # print('mse_values dim', out[5])
    df_data = {
        'mse': mse_values,
        'average': [average_value] + [None] * (len(mse_values) - 1)  # 只在第一行显示平均值
    }
    pd.DataFrame(df_data).to_csv(args.result_path + args.save_finetuned_model + '_acc_last_3dim.csv', float_format='%.6f', index=False)
    # 输出9维w-distance
    pd.DataFrame(data={'avg w-distance': out[4][0], 'each dim w-distance': out[4][1]}).to_csv(args.result_path + args.save_finetuned_model + '_acc_w_distance_9dim.csv', float_format='%.6f', index=False)
    # pd.DataFrame(np.array(out[4][1]).reshape(1,-1), columns=['mse']).to_csv(args.result_path + args.save_finetuned_model + '_acc_w_distance_9dim.csv', float_format='%.6f', index=False)

def save_recorders(learn):
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.model_path + args.save_finetuned_model + '_losses.csv', float_format='%.6f', index=False)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 绘制损失曲线图
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    if valid_loss:  # 如果有验证损失数据
        plt.plot(epochs, valid_loss, 'r-', label='Valid Loss', linewidth=2)
    
    plt.title(f'Train/Valid Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plot_path = args.model_path + args.save_finetuned_model + '_losses.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    
    print(f'Loss curve plot saved to: {plot_path}')

def test_func(weight_path):
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type='prediction').to('cuda')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs)
    out  = learn.test(dls.test, weight_path=weight_path+'.pth', scores=[mse])         # out: a list of [pred, targ, score]
    # save results
    pd.DataFrame(np.array(out[2]).reshape(1,-1), columns=['mse']).to_csv(args.result_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
    # save target results
    save_format_result(out)
    return out


if __name__ == '__main__':
        
    if args.is_finetune:
        args.dset = args.dset_finetune
        # Finetune
        suggested_lr = find_lr(head_type='prediction')        
        finetune_func(suggested_lr)        
        print('finetune completed')
        # Test
        out = test_func(args.model_path+args.save_finetuned_model)         
        print('----------- Complete! -----------')

    elif args.is_linear_probe:
        args.dset = args.dset_finetune
        # Finetune
        suggested_lr = find_lr(head_type='prediction')        
        linear_probe_func(suggested_lr)        
        print('finetune completed')
        # Test
        out = test_func(args.model_path+args.save_finetuned_model)        
        print('----------- Complete! -----------')

    else:
        args.dset = args.dset_finetune
        weight_path = args.pretrained_model if args.pretrained_model else args.model_path+args.dset_finetune+'_patchtst_finetuned'+suffix_name
        print('weight_path',weight_path)
        # Test
        out = test_func(weight_path)        
        print('----------- Complete! -----------')


