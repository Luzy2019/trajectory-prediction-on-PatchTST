

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

ROOT_PATH = './dataset/'
DSETS = ['source_domain', 's0.3548_m907', 'CAV-H', 'HTV2']

def get_dls(params):
    
    # assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

    # root_path = './dataset/' + params.dataset_size + '/both/'
    # root_path = './dataset/' + params.dataset_size + '/only-dg/'
    root_path = './dataset/' + params.dataset_size + '/'
        
    if params.dset == 'source_domain':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,      # dataset class: 数据集类
                dataset_kwargs={                # dataset key word args
                'root_path': './dataset/others/',
                'data_path': 'source_domain.csv',
                'features': params.features,    # M MS S
                'scale': True, # True           # scale 处理
                'size': size,                   # [100, 0, 100]
                'use_time_features': params.use_time_features # False
                },
                batch_size=params.batch_size,   # 32
                workers=params.num_workers,     # number of workers
                )

    elif params.dset == 'processed_data_55':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'processed_data_55.csv',
                'features': params.features,
                'scale': True, # True
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    
    elif params.dset == 's0.3548_m907':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 's0.3548_m907.csv',
                'features': params.features,
                'scale': True, # True
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'CAV-H':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'CAV-H.csv',
                'features': params.features,
                'scale': True, # True
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'HTV2':
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'HTV2.csv',
                'features': params.features,
                'scale': True, # True
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
        
    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    # dls.vars = 9, dls.len = 3, dls.c = 2
    return dls