from types import SimpleNamespace

__all__ = ['BASELINE_CONFIG', 'FINETUNE_CONFIG', 'TEST_FT_CONFIG', 'TEST_BL_CONFIG', 'merge_namespaces']

DATASET_NAME = 'CAV-H'
DATASET_SIZE = '5138'
DATASET_TEST_SIZE = '20552'

PETRAINED_MODEL_PATH = 'saved_models/source_domain/patchtst_pretrained_cw100_patch10_stride10_epochs-pretrain10_mask0.1_model1.pth'
BASELINE_MODEL_PATH = 'saved_models/source_domain/source_domain_patchtst_linear-probe_cw100_tw100_patch100_stride100_epochs-finetune20_model1'
FINETUNED_MODEL_PATH = f'saved_models/{DATASET_NAME}/{DATASET_NAME}_patchtst_finetuned_cw100_tw100_patch20_stride20_epochs-finetune20_model1'

FINETUNE_CONFIG = SimpleNamespace(
    # Dataset and dataloader
    dset_finetune = DATASET_NAME,
    context_points = 100,
    target_points = 100,
    batch_size = 64,
    num_workers = 0,
    scaler = 'standard',
    features = 'M',
    dataset_size = DATASET_SIZE,
    dataset_augmented = 1,
    # Patch
    patch_len = 20,
    stride = 20,
    # RevIN
    revin = 0,
    # use time feature
    use_time_features = 1,
    # Model args
    n_layers = 3,
    n_heads = 16,
    d_model = 64,
    d_ff = 256,
    dropout = 0.2,
    head_dropout = 0.2,
    partial_freeze = 0, # 1: 只微调最外层 2：微调后2层，3：微调全部3层
    # Optimization args
    n_epochs_finetune = 20,
    lr = 1e-4,
    # Pretrained model name
    pretrained_model = PETRAINED_MODEL_PATH,
    # model id to keep track of the number of models saved
    finetuned_model_id = 1,
    model_type = 'based_model'
)

BASELINE_CONFIG = SimpleNamespace(
    # Dataset and dataloader
    dset_finetune = DATASET_NAME,
    context_points = 100,
    target_points = 100,
    batch_size = 64,
    num_workers = 0,
    scaler = 'standard',
    features = 'M',
    dataset_size = DATASET_SIZE,
    dataset_augmented = 0,
    # Patch
    patch_len = 100,
    stride = 100,
    # RevIN
    revin = 0,
    # use time feature
    use_time_features = 1,
    # Model args
    n_layers = 3,
    n_heads = 16,
    d_model = 64,
    d_ff = 256,
    dropout = 0.2,
    head_dropout = 0.2,
    partial_freeze = 0,
    # Optimization args
    n_epochs_finetune = 20,
    lr = 1e-4,
    # Pretrained model name
    pretrained_model = PETRAINED_MODEL_PATH,
    # model id to keep track of the number of models saved
    finetuned_model_id = 1,
    model_type = 'based_model'
)

TEST_BL_CONFIG = SimpleNamespace(
    # Dataset and dataloader
    dset_finetune = DATASET_NAME,
    context_points = 100,
    target_points = 100,
    batch_size = 64,
    num_workers = 0,
    scaler = 'standard',
    features = 'M',
    dataset_size = DATASET_TEST_SIZE,
    dataset_augmented = 0,
    # Patch
    patch_len = 100,
    stride = 100,
    # RevIN
    revin = 0,
    # use time feature
    use_time_features = 1,
    # Model args
    n_layers = 3,
    n_heads = 16,
    d_model = 64,
    d_ff = 256,
    dropout = 0.2,
    head_dropout = 0.2,
    partial_freeze = 0,
    # Optimization args
    n_epochs_finetune = 20,
    lr = 1e-4,
    # Pretrained model name
    pretrained_model = BASELINE_MODEL_PATH,
    # model id to keep track of the number of models saved
    finetuned_model_id = 1,
    model_type = 'based_model'
)

TEST_FT_CONFIG = SimpleNamespace(
    # Dataset and dataloader
    dset_finetune = DATASET_NAME,
    context_points = 100,
    target_points = 100,
    batch_size = 64,
    num_workers = 0,
    scaler = 'standard',
    features = 'M',
    dataset_size = DATASET_TEST_SIZE,
    dataset_augmented = 0,
    # Patch
    patch_len = 20,
    stride = 20,
    # RevIN
    revin = 0,
    # use time feature
    use_time_features = 1,
    # Model args
    n_layers = 3,
    n_heads = 16,
    d_model = 64,
    d_ff = 256,
    dropout = 0.2,
    head_dropout = 0.2,
    partial_freeze = 0,
    # Optimization args
    n_epochs_finetune = 20,
    lr = 1e-4,
    # Pretrained model name
    pretrained_model = None,
    # model id to keep track of the number of models saved
    finetuned_model_id = 1,
    model_type = 'based_model'
)

def merge_namespaces(ns1, ns2):
    """合并两个SimpleNamespace对象"""
    merged = SimpleNamespace()
    merged.__dict__.update(ns1.__dict__)
    merged.__dict__.update(ns2.__dict__)
    return merged