import sys
sys.path.append("/home/featurize/work/2022-Graduation-thesis")
sys.path.append("H:\项目管理\毕业设计\project")
import torch
import numpy as np
from tqdm import tqdm
import os
from Utils import VariantWordDataset
from torch.utils.data import DataLoader
from torch import nn
from Utils.config import Config
from Model import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

if __name__ == "__main__":

    # 配置类初始化
    config = Config()
                         
    # 数据集构建
    train_set = VariantWordDataset("train", config.source_dic_path, config.target_dic_path, isAligned=True)
    valid_set = VariantWordDataset("test", config.source_dic_path, config.target_dic_path, isAligned=True)
    print(f"Train size: {len(train_set)}")


    # dataloader 初始化
    # 数据传输cpu数目
    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=train_set.generate_batch, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=valid_set.generate_batch, num_workers=n_cpu)

    # model = ConvS2SModel(config)
    model = TransformerModel(config)
    # model = RNNSearchModel(config)



    # wandb logger配置
    wandb_logger = WandbLogger(project="variantWordDetection",
                        name = model.log_name,
                        save_dir = '../Logs',
                        log_model="all",
                        # offline=True
                        )
    
    # checkpoint保存
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_f1",
        dirpath="../Weights",
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