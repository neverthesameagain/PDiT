import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from tqdm import tqdm
import numpy as np
import time
import random
import logging
from torch.distributions import Categorical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DecisionTransformer, self).__init__()
        self.perception_layers = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.decision_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, x):
        x = self.perception_layers(x)
        x = x.view(x.size(0), -1)  # Flatten for decision stage
        x = self.decision_layers(x)
        return x

class PPO(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PPO, self).__init__()
        self.perception_layers = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, 1)
        )

    def forward(self, x):
        x = self.perception_layers(x)
        x = x.view(x.size(0), -1) 
        return self.actor(x), self.critic(x)

class Trainer:
    def __init__(self, model, optimizer, env, gamma=0.99, reward_scale=0.01, checkpoint_dir="checkpoints"):
        self.model = model
        self.optimizer = optimizer
        self.env = env
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.diagnostics = dict()
        self.train_count = 0
        self.start_time = time.time()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info("Trainer initialized.")

    def train(self, num_episodes=1000, print_logs=False, start_episode=0):
        train_losses = []
        logs = dict()
        logger.info(f"Starting training for {num_episodes} episodes.")

        for episode in range(start_episode, num_episodes):
            obs, _ = self.env.reset()
            episode_loss = 0
            done = False
            rewards = []
            log_probs = []

            logger.info(f"Episode {episode + 1} started.")

            prev_action = None
            step_count = 0

            while not done:
                if isinstance(obs, dict) and 'image' in obs:
                    obs_tensor = torch.tensor(obs['image'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0  
                elif isinstance(obs, np.ndarray):
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0  
                else:
                    logger.error("Error: Observation does not contain 'image' key or is not a valid array.")
                    break

                action_logits = self.model(obs_tensor)
                action_probabilities = torch.softmax(action_logits, dim=-1)
                action_distribution = torch.distributions.Categorical(action_probabilities)
                action = action_distribution.sample()

                if prev_action is not None and action == prev_action:
                    step_count += 1
                else:
                    step_count = 0

                if step_count > 10:
                    action = torch.tensor((action.item() + 1) % self.env.action_space.n)  
                    step_count = 0

                log_prob = action_distribution.log_prob(action)
                log_probs.append(log_prob)

                obs, reward, done, truncated, info = self.env.step(action.item())
                rewards.append(reward * self.reward_scale)  
                prev_action = action

            if len(log_probs) == 0:
                logger.error("Error: No actions were taken in this episode.")
                continue

            # Calculate discounted rewards
            discounted_rewards = self.compute_discounted_rewards(rewards)
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

            # Normalize discounted rewards
            if len(discounted_rewards) > 1:
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

            # Calculate loss
            loss = -torch.sum(torch.stack(log_probs) * discounted_rewards)
            episode_loss += loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()

            train_losses.append(episode_loss)
            self.train_count += 1

            logger.info(f"Episode {episode + 1} completed. Loss: {episode_loss}")

            # Save checkpoint every 50 episodes
            if (episode + 1) % 50 == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"decision_transformer_checkpoint_{episode + 1}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"Checkpoint saved at episode {episode + 1}: {checkpoint_path}")

        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        if print_logs:
            logger.info('=' * 80)
            logger.info(f'Training completed for {num_episodes} episodes')
            for k, v in logs.items():
                logger.info(f'{k}: {v}')

        return logs

    def compute_discounted_rewards(self, rewards):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + (self.gamma * cumulative_reward)
            discounted_rewards.insert(0, cumulative_reward)
        return discounted_rewards

def main():
    logger.info("Starting main function.")
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    env = gym.make("BabyAI-GoToLocal-v0", render_mode="rgb_array")
    env.action_space.seed(seed)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    input_dim = 3
    hidden_dim = 64
    output_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]

    # Decision Transformer model
    decision_transformer_model = DecisionTransformer(input_dim, hidden_dim, output_dim)
    logger.info("Decision Transformer model initialized.")
    decision_transformer_optimizer = optim.Adam(decision_transformer_model.parameters(), lr=1e-5)  # Reduced learning rate for stability

    # Load from checkpoint if available
    checkpoint_dir = "checkpoints"
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("decision_transformer_checkpoint_")]
        if checkpoints:
            latest_checkpoint = '/Users/aryanmathur/Desktop/Sem5/checkpoints/decision_transformer_checkpoint_750.pth'
    start_episode = 0
    if latest_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        decision_transformer_model.load_state_dict(torch.load(checkpoint_path))
        start_episode = int(latest_checkpoint.split('_')[-1].split('.')[0])
        logger.info(f"Resuming from checkpoint: {checkpoint_path} at episode {start_episode}")

    trainer = Trainer(decision_transformer_model, decision_transformer_optimizer, env, checkpoint_dir=checkpoint_dir)
    trainer.train(num_episodes=1000, print_logs=True, start_episode=800)
    torch.save(decision_transformer_model.state_dict(), "decision_transformer_gotolocal_final.pth")
    logger.info("Decision Transformer model saved.")

if __name__ == "__main__":
    main()
