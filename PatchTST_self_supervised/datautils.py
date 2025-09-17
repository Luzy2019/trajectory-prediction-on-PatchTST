

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['source_domain', 's0.3548_m907', 'CAV-H', 'HTV2']

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

    ROOT_PATH = f'./dataset/{params.dataset_size}/dg/' if params.dataset_augmented else f'./dataset/{params.dataset_size}/'
    DATA_PATH = params.dset + '.csv'
    
    size = [params.context_points, 0, params.target_points]
    dls = DataLoaders(
        datasetCls=Dataset_Custom,      # dataset class: 数据集类
        dataset_kwargs={                # dataset key word args
        'root_path': ROOT_PATH,
        'data_path': DATA_PATH,
        'features': params.features,    # M MS S
        'scale': True, # True           # scale 处理
        'size': size,                   # [100, 0, 100]
        'use_time_features': params.use_time_features # False
        },
        batch_size=params.batch_size,   # 32
        workers=params.num_workers,     # number of workers
    )

    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls