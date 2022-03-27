from argparse import Namespace
from distutils.command.config import config
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
from torchvision.models import resnet50
from downstream_modules import DownstreamDataloader, DownstreamLinearModule
from config_training import *

kwargs = {
    "num_classes": 1000,
    "cifar": False,
    "max_epochs": 60,
    "precision": 16,
    "lars": False,
    "lr": 0.02,
    "auto_lr_find": True,
    "exclude_bias_n_norm": False,
    "gpus": 4,    # Number of GPUs
    "weight_decay": 0,
    "scheduler": SCHEDULER,
    "lr_decay_steps": [45],
    "replica_batch_size": 256,
    "batch_size": 1024,
    "num_workers": 4,
    "metric": 'accuracy',
    "task": 'finetune',
    "backbone_weights": WEIGHTS
}

model = DownstreamLinearModule(**kwargs)

dataloader = DownstreamDataloader(DATASET, download=False, task=kwargs['task'], batch_size=kwargs["replica_batch_size"], num_workers=kwargs['num_workers'])

wandb_logger = WandbLogger(
    # name of the experiment
    name=f'{METHOD}_semi-supervised_{DATASET}_lr={kwargs["lr"]}_wd={kwargs["weight_decay"]}',
    project="MNCRL_downstream_tasks",  # name of the wandb project
    entity='mlbrl',
    group=DATASET,
    job_type='semi_supervised',
    offline=False,
)
wandb_logger.watch(model, log="all", log_freq=50)

trainer = Trainer(accelerator='gpu', auto_select_gpus=False, gpus=kwargs["gpus"],
                   logger=wandb_logger, max_epochs=kwargs["max_epochs"], auto_lr_find=kwargs["auto_lr_find"], strategy="ddp")

if __name__ == '__main__':
    print("start training")
    print(f"Weights : {WEIGHTS}")
    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)
    print("end training")

# def train():
#     print("start training")
#     print(f"Weights : {WEIGHTS}")
#     # print(train_loader)
#     # trainer.fit(model, train_loader, val_loader)
#     # if VAL_PATH != TEST_PATH:
#         # trainer.validate(model, test_loader)
#     print("end training")