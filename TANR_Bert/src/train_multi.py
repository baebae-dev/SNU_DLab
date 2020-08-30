from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import BaseDataset
import torch
import torch.nn.functional as F
import time
import numpy as np
from config import model_name
from tqdm import tqdm
import os
from pathlib import Path 
from evaluate_multi import evaluate
import importlib 
import datetime
# import torch.nn as nn

# model, config load
try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except (AttributeError, ModuleNotFoundError):
    print(f"{model_name} not included!")
    exit() 

# device 
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2' 
# print(f'Using CPUS {os.environ["CUDA_VISIBLE_DEVICES"]}')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('using gpu nums', torch.cuda.device_count())

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        """
        if you use other metrics where a higher value is better, e.g. accuracy,
        call this with its corresponding negative value
        """
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }

    if not all_checkpoints:
        return None

    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])
 
def train():
    writer = SummaryWriter(
        log_dir=
        f"./runs/{model_name}/{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}"
    )

    if not os.path.exists('checkpoint'):
        print('makeidr checkpoint')
        os.makedirs('checkpoint')

    try:  
        pretrained_word_embedding = torch.from_numpy(
            np.load('../data/train/pretrained_word_embedding.npy')).float()
    except FileNotFoundError:
        pretrained_word_embedding = None

    # model 지정 
    model = Model(config, pretrained_word_embedding, writer).to(device)
    # model = nn.DataParallel(model, output_device=0) 
    # model = model.to(device)
    print(model) 

    # data load
    dataset = BaseDataset('../data/train/behaviors_parsed.tsv', 
                          '../data/train/news_parsed.tsv',
                          config.dataset_attributes)
    print(f"Load training dataset with size {len(dataset)}.")

    dataloader = iter(
        DataLoader(dataset,
                   batch_size=config.batch_size,
                   shuffle=True,
                   num_workers=config.num_workers,
                   drop_last=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    step = 0
    early_stopping = EarlyStopping() 

    checkpoint_dir = os.path.join('../checkpoint', model_name, 'batch_size'+str(config.batch_size)+'_num'+str(config.num_clicked_news_a_user)) 
    print(f'checkpoint_dir : {checkpoint_dir}')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)


    checkpoint_path = latest_checkpoint(checkpoint_dir)
    print(f'checkpoint_path : {checkpoint_path}')
    if checkpoint_path is not None:
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        early_stopping(checkpoint['early_stop_value'])
        model.train()

    with tqdm(total=config.num_batches, desc="Training") as pbar:
        for i in range(1, config.num_batches + 1):
            try:
                minibatch = next(dataloader)
            except StopIteration:
                exhaustion_count += 1
                tqdm.write(
                    f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
                )
                dataloader = iter(
                    DataLoader(dataset,
                               batch_size=config.batch_size,
                               shuffle=True,
                               num_workers=config.num_workers,
                               drop_last=True))
                minibatch = next(dataloader) 

            step += 1 
            # loss 
            # TopicClassificationLoss 훈련 동시 
            y_pred, topic_classification_loss = model(
                minibatch["candidate_news"], minibatch["clicked_news"])
        
            loss = torch.stack([x[0] for x in -F.log_softmax(y_pred, dim=1)
                                ]).mean()
            
            if i % 10 == 0:
                writer.add_scalar('Train/BaseLoss', loss.item(), step)
                writer.add_scalar('Train/TopicClassificationLoss',
                                    topic_classification_loss.item(), step)
                writer.add_scalar(
                    'Train/TopicBaseRatio',
                    topic_classification_loss.item() / loss.item(), step)
            
            loss += config.topic_classification_loss_weight * topic_classification_loss
            loss_full.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                writer.add_scalar('Train/Loss', loss.item(), step)

            if i % config.num_batches_show_loss == 0:
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}"
                )

            if i % config.num_batches_validate == 0:
                val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
                    model, '../data/val')
                writer.add_scalar('Validation/AUC', val_auc, step)
                writer.add_scalar('Validation/MRR', val_mrr, step)
                writer.add_scalar('Validation/nDCG@5', val_ndcg5, step)
                writer.add_scalar('Validation/nDCG@10', val_ndcg10, step)
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
                )

                early_stop, get_better = early_stopping(-val_auc)
                if early_stop:
                    tqdm.write('Early stop.')
                    break 

                elif get_better:
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'step': step,
                            'early_stop_value': -val_auc
                        }, f"../checkpoint/{model_name}/batch_size{config.batch_size}_num{config.num_clicked_news_a_user}/ckpt{config.batch_size}-{step}.pth")
                    print(f" torch save at ../checkpoint/{model_name}/{config.batch_size}/batch_size{config.batch_size}_num{config.num_clicked_news_a_user}/ckpt{config.batch_size}-{step}.pth")

            pbar.update(1)


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Training model {model_name}')
    train()
