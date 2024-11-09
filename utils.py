import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import json
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_
from model import CPDIT
from data import EpisodeDataset
from torch.amp import GradScaler, autocast

def train_model():

    config = json.load(open('config.json'))
        
    train_dataset = EpisodeDataset(config)  # check this out in data.py
    sampler = WeightedRandomSampler(train_dataset.p_sample, num_samples = config['batch_size'], replacement=True) # check this out
    train_iter = DataLoader(train_dataset, sampler=sampler, num_workers=4, batch_size=config['batch_size'], shuffle=False)
    
    model = CPDIT(config=config).to(config['device'])
    model.train()
    
    train_num = len(train_iter)
    length = config['epochs'] * train_num
    total_steps = config['epochs'] * train_num // config['batch_size']
    config['total_steps'] = total_steps
    
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = cosine_scheduler(optimizer, config['total_steps'] * config['warmup_ratio'], config['total_steps'])
    scaler = GradScaler()
    
    for epoch in tqdm(range(config['epochs'])):
        
        pbar = tqdm(train_iter)  # maximum total number
        for step, batch in enumerate(train_iter):
                        
            # get the batch
            images = batch['images'].to(config['device'])
            actions = batch['actions'].to(config['device'])
            prompt = batch['prompt']
            prompt_mask = batch['prompt_mask']
            timesteps = batch['timesteps'].to(config['device'])
            Gt = batch['Gt'].to(config['device'])
            mask = batch['mask'].to(config['device'])

            with autocast(device_type=config['device'], dtype=torch.float16):
                loss, preds = model(images, prompt, prompt_mask, actions, Gt[:,:-1], timesteps, mask, actions)
                loss = loss / config['gradient_accumulation_steps']
            
            scaler.scale(loss).backward()

            if step % config['gradient_accumulation_steps'] == 0:
                            
                if config['max_grad_norm'] != 0.0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                    
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                optimizer.zero_grad(set_to_none=True)
                pbar.set_description(desc= f'Loss={loss.item()} steps={step} lr={scheduler.get_last_lr()}')                
        
            pbar.update(1)
        
        print(f'Saving model at epoch {epoch}')
        torch.save(model.state_dict(), config['save_path'])
        print(f'Epoch {epoch} completed')

def cosine_scheduler(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)