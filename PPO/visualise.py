
from stable_baselines3 import PPO
import cv2
import numpy as np

try:
    import gymnasium as gym 
except ImportError:
    import gym

env = gym.make('BabyAI-GoToLocal-v0', render_mode="human")  # 'human' mode to open a window for visualization

model = PPO.load("/Users/aryanmathur/Desktop/Sem5/ppo_gotolocal_final.zip")

obs, info = env.reset() 
done = False

if isinstance(obs, dict):
    obs = obs['image']

# image  size (3, 56, 56)
# obs (7, 7, 3)
obs_resized = cv2.resize(obs, (56, 56)) 
obs_resized = np.moveaxis(obs_resized, -1, 0) 

print("Starting BabyAI-GoToLocal...")

while not done:
    env.render() 
    
    action, _states = model.predict(obs_resized) 
    obs, reward, done, truncated, info = env.step(action) 

    if isinstance(obs, dict):
        obs_resized = cv2.resize(obs['image'], (56, 56))  
        obs_resized = np.moveaxis(obs_resized, -1, 0) 
    done = done or truncated

print("Game Over")
env.close()
