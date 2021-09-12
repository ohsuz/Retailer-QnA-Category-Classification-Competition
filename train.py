import logging
import wandb
from tqdm import tqdm, trange
import argparse

import torch
import torch.optim as optim
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
import torch.optim as optim
from torch.optim import AdamW
import random
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from modules.metrics import get_metric_fn

from modules.dataset import CustomDataset
from modules.trainer import Trainer
from modules.utils import seed_everything, load_yaml, save_yaml, get_logger, make_directory
from modules.earlystoppers import LossEarlyStopper
from modules.recorders import PerformanceRecorder

from torch.utils.data import DataLoader
from modules.model import IntentClassifier
import torch.nn as nn
import os
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disabling parallelism to avoid deadlocks
logging.disable(logging.WARNING)
# DEBUG
DEBUG = False

# DIR
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')

# CONFIG
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True)
args = parser.parse_args()
CONFIG_NAME = args.config_name
print(CONFIG_NAME)
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config', f'{CONFIG_NAME}.yml')
config = load_yaml(TRAIN_CONFIG_PATH)

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# WANDB
RUN_NAME = config['WANDB']['run_name']

# TRAIN
DATA_TYPE = config['TRAIN']['data_type'] # original or all 
MODEL_NAME = config['TRAIN']['model_name']
EPOCHS = config['TRAIN']['num_epochs']
BATCH_SIZE = config['TRAIN']['batch_size']
LEARNING_RATE = config['TRAIN']['learning_rate']
EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']
OPTIMIZER = config['TRAIN']['optimizer']
SCHEDULER = config['TRAIN']['scheduler']
MOMENTUM = config['TRAIN']['momentum']
WEIGHT_DECAY = config['TRAIN']['weight_decay']

# TRAIN SERIAL
KST = timezone(timedelta(hours=9))
TRAIN_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
TRAIN_SERIAL = f'{TRAIN_TIMESTAMP}' if DEBUG is not True else 'DEBUG'

# PERFORMANCE RECORD
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'records', 'train', TRAIN_SERIAL)
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']

if __name__ == '__main__':
    wandb.login()

    # Set random seed
    seed_everything(RANDOM_SEED)

    # Set device
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))

    # Set train result directory
    make_directory(PERFORMANCE_RECORD_DIR)

    # Set system logger
    system_logger = get_logger(name='train',
                               file_path=os.path.join(PERFORMANCE_RECORD_DIR, 'train_log.log'))

    # Load dataset & dataloader
    train_dataset = CustomDataset(model_name=MODEL_NAME, data_dir=DATA_DIR, data_type=DATA_TYPE, mode='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if DATA_TYPE != 'all':
        validation_dataset = CustomDataset(model_name=MODEL_NAME, data_dir=DATA_DIR, data_type=DATA_TYPE, mode='val')
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load Model
    model = IntentClassifier(model_name=MODEL_NAME).to(device)
    system_logger.info('===== Review Model Architecture =====')
    system_logger.info(f'{model} \n')

    # Set optimizer, scheduler, metric function
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5, max_lr=0.0001, epochs=EPOCHS, steps_per_epoch=len(train_dataloader))

    # Set metrics
    metric_fn = accuracy_score

    # Set trainer
    trainer = Trainer(model, device, metric_fn ,optimizer, scheduler, logger=system_logger)

    # Set earlystopper
    early_stopper = LossEarlyStopper(patience=EARLY_STOPPING_PATIENCE, verbose=True, logger=system_logger)

    # Set performance recorder
    key_column_value_list = [
        TRAIN_SERIAL,
        TRAIN_TIMESTAMP,
        EARLY_STOPPING_PATIENCE,
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        RANDOM_SEED]

    performance_recorder = PerformanceRecorder(column_name_list=PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               record_dir=PERFORMANCE_RECORD_DIR,
                                               key_column_value_list=key_column_value_list,
                                               logger=system_logger,
                                               model=model,
                                               optimizer=optimizer,
                                               scheduler=scheduler)

    # Save config yaml file
    save_yaml(os.path.join(PERFORMANCE_RECORD_DIR, 'train_config.yaml'), config)
    
    # Train
    criterion = -1
    wandb.init(project='AICONNECT', config=config, entity="ohsuz", name=RUN_NAME)

    for epoch_index in trange(EPOCHS):
        trainer.train_epoch(train_dataloader, epoch_index=epoch_index)
        if DATA_TYPE != 'all':
            trainer.validate_epoch(validation_dataloader, epoch_index=epoch_index)
        
        if DATA_TYPE != 'all':
            wandb.log({"epoch": epoch_index,
                       "train_loss": trainer.train_mean_loss, "train_score": trainer.train_score,
                       "val_loss": trainer.val_mean_loss, "val_score": trainer.validation_score})

            # early_stopping check
            early_stopper.check_early_stopping(loss=trainer.val_mean_loss)

            if early_stopper.stop:
                print('Early stopped')
                break
                
            if trainer.validation_score > criterion:
                criterion = trainer.validation_score
                check_point = {
                    'epoch': epoch_index,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(check_point, os.path.join(MODEL_DIR, f'{RUN_NAME}.pt'))
                
        else:
            wandb.log({"epoch": epoch_index,
                       "train_loss": trainer.train_mean_loss, "train_score": trainer.train_score})
            check_point = {
                'epoch': epoch_index,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(check_point, os.path.join(MODEL_DIR, f'{RUN_NAME}.pt'))