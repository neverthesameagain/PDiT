import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import json
import math
import gymnasium as gym
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_
from model import CPDIT
from data import EpisodeDataset
from torch.amp import GradScaler, autocast
from data import preprocess_texts

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

def evaluate(model, env_name, config, vocab):

    env = gym.make(env_name, render_mode="human")
    model.eval()
    model.to(device=config['device'])

    image_dim = (7, 7, 3)
    act_dim = 6

    state, _ = env.reset()
    image = state['image']
    prompt = state['mission']
    prompt, _ = preprocess_texts([prompt], vocab)
    
    images = torch.from_numpy(image).reshape(1, *image_dim).to(device=config['device'], dtype=torch.uint8)
    prompt = torch.from_numpy(prompt).to(device=config['device'], dtype=torch.long)
    actions = torch.zeros((0, act_dim), device=config['device'], dtype=torch.float16)
    rewards = torch.zeros(0, device=config['device'], dtype=torch.float16)
    returns = torch.zeros(0, device=config['device'], dtype=torch.float16)
    timesteps = torch.tensor(0, device=config['device'], dtype=torch.long).reshape(1, 1)

    while True:
        
        env.render()

        actions = torch.cat([actions, torch.zeros((1, act_dim), device=config['device'])], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=config['device'])])
        returns = torch.cat([returns, torch.zeros(1, device=config['device'])])

        action = get_action(model, images, prompt, actions, returns, timesteps, config)
        
        actions[-1] = F.one_hot(torch.tensor(action), act_dim)
        action = torch.tensor(action).detach().cpu().numpy()

        state, reward, terminated, truncated, _ = env.step(action)
        t = timesteps[-1, -1].item()
        if reward > 0: reward = max(0, 1 - 0.9 * ((t + 1) / config['max_ep_len']))

        cur_image = torch.from_numpy(state['image']).to(device=config['device']).reshape(1, *image_dim)
        images = torch.cat([images, cur_image], dim=0)
        rewards[-1] = reward
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=config['device'], dtype=torch.long) * (t + 1)], dim=1)
        
        if terminated or truncated: break

def get_action(model, images, prompt, actions, Gt, timesteps, config):

    max_length = 16
    act_dim = 6

    images = images.reshape(1, -1, *images.size()[-3:])
    prompt = prompt.reshape(1, -1, prompt.size(-1))
    actions = actions.reshape(1, -1, act_dim)
    Gt = Gt.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    images = images[:,-max_length:]
    prompt = prompt[:,-max_length:]
    actions = actions[:,-max_length:]
    Gt = Gt[:,-max_length:]
    timesteps = timesteps[:,-max_length:]

    mask = torch.cat([torch.zeros(max_length-images.shape[1]), torch.ones(images.shape[1])])
    mask = mask.to(dtype=torch.float16, device=images.device).reshape(1, -1)

    images = torch.cat(
        [torch.zeros((images.shape[0], max_length-images.shape[1], *images.size()[-3:]), device=images.device), images],
        dim=1).to(dtype=torch.float16)
    prompt = torch.cat(
        [torch.zeros((prompt.shape[0], max_length-prompt.shape[1], prompt.size(-1)), device=prompt.device), prompt],
        dim=1).to(dtype=torch.long)
    actions = torch.cat(
        [torch.zeros((actions.shape[0], max_length-actions.shape[1], act_dim), device=actions.device), actions],
        dim=1).to(dtype=torch.float16)
    Gt = torch.cat(
        [torch.zeros((Gt.shape[0], max_length-Gt.shape[1], 1), device=Gt.device), Gt],
        dim=1).to(dtype=torch.float16)
    timesteps = torch.cat(
        [torch.zeros((timesteps.shape[0], max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
        dim=1).to(dtype=torch.long)
    
    images = images.repeat(2, 1, 1, 1, 1)
    prompt = prompt.repeat(2, 1, 1)
    actions = actions.repeat(2, 1, 1)
    Gt = Gt.repeat(2, 1, 1)
    timesteps = timesteps.repeat(2, 1)
    mask = mask.repeat(2, 1)
    
    with autocast(device_type=config['device'], dtype=torch.float16):
        _, action_preds = model(images, prompt, torch.ones_like(prompt, dtype=torch.float16), actions, Gt, timesteps, mask=mask)

    return action_preds[15].argmax().item()