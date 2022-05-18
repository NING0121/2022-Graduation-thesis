import sys
sys.path.append("/home/featurize/work/2022-Graduation-thesis")
sys.path.append("H:\项目管理\毕业设计\project")
sys.path.append("/data/NING/2022-Graduation-thesis")
import torch
import numpy as np
from tqdm import tqdm
import os
from data_loader import VariantWordDataset
from torch.utils.data import DataLoader
from torch import nn
from config import *
from Model import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

# 配置类初始化
config = arg_parse()
                        
# 数据集构建
train_set = VariantWordDataset("train", config, isAligned=config.isAligned, supply_ratio=config.supply_ratio)
valid_set = VariantWordDataset("test", config, isAligned=config.isAligned)
print(f"Train size: {len(train_set)}")


# dataloader 初始化
# 数据传输cpu数目
n_cpu = os.cpu_count()
train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=train_set.generate_batch, num_workers=n_cpu)
valid_dataloader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=valid_set.generate_batch, num_workers=n_cpu)

# Select the Model
if config.model == 'RNNSearch':
    model = RNNSearchModel(config)
elif config.model == 'ConvS2S':
    model = ConvS2SModel(config)
elif config.model == 'Transformer':
    model = TransformerModel(config)


# wandb logger配置
wandb_logger = WandbLogger(project="variantWordDetection",
                    name = model.log_name,
                    save_dir = 'Logs',
                    # log_model=True,
                    offline=True
                    )

# checkpoint保存
checkpoint_callback = ModelCheckpoint(
    monitor="valid_BLEU_SCORE",
    dirpath=f"Weights/Weights_{model.check_name}",
    filename=f"{model.check_name}"+"-{epoch:02d}-{valid_f1:.2f}",
    save_top_k=3,
    mode="max",
)


# trainer 定义
trainer = pl.Trainer(
    max_epochs=config.epochs, 
    gpus=1,
    logger = wandb_logger,
    callbacks=[checkpoint_callback]
    )


# 训练
trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=valid_dataloader
)