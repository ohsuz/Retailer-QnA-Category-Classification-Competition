import os
import random
import argparse
import tqdm
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from modules.dataset import CustomDataset, TestDataset
from modules.trainer import Trainer
from modules.utils import load_yaml
from modules.model import IntentClassifier
from modules.utils import seed_everything
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn

KST = timezone(timedelta(hours=9))
TEST_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")

# DIR
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
LOGIT_DIR = os.path.join(PROJECT_DIR, 'logits')
SUBMISSION_DIR = os.path.join(PROJECT_DIR, 'submissions')

# CONFIG
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True)
args = parser.parse_args()
CONFIG_NAME = args.config_name
PREDICT_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config', f'{CONFIG_NAME}.yml')
config = load_yaml(PREDICT_CONFIG_PATH)

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# PREDICT
MODEL_NAME = config['PREDICT']['model_name']
TRAINED_MODEL_NAME = config['PREDICT']['trained_model_name']
BATCH_SIZE = config['PREDICT']['batch_size']
SUBMISSION_NAME = config['PREDICT']['submission_name']

if __name__ == '__main__':
    
    # Set random seed
    seed_everything(RANDOM_SEED)
    
    # Set device
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))

    # Load dataset & dataloader
    test_dataset = TestDataset(model_name=MODEL_NAME, data_dir=DATA_DIR, mode='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load Model
    model = IntentClassifier(model_name=MODEL_NAME).to(device)
    TRAINED_MODEL_PATH = os.path.join(MODEL_DIR, f'{TRAINED_MODEL_NAME}.pt')
    print(TRAINED_MODEL_PATH)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH)['model'])

    # Set metrics & Loss function
    metric_fn = accuracy_score
    loss_fn = nn.CrossEntropyLoss()

    # Set trainer
    trainer = Trainer(model, device, loss_fn, metric_fn)

    # Predict
    pred = []
    pred, logits = trainer.test_epoch(test_dataloader, epoch_index=0)
    pred = test_dataset.label_decoder(pred)
    print('decode completed--')

    # Save prediction
    pred_df = pd.DataFrame()
    pred_df['conv_num'] = test_dataset.conv_num
    pred_df['intent'] = pred
    
    np.save(os.path.join(LOGIT_DIR, f'{SUBMISSION_NAME}.npy'), logits.detach().cpu().numpy())
    pred_df.to_csv(os.path.join(SUBMISSION_DIR, f'{SUBMISSION_NAME}.csv'),index=False)
