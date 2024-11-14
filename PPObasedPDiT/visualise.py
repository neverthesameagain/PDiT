import gymnasium as gym
import torch
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import torch.nn as nn

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
        x = x.view(x.size(0), -1)
        x = self.decision_layers(x)
        return x

env = gym.make('BabyAI-GoToLocal-v0', render_mode="human")
env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)

input_dim = 3  
hidden_dim = 64
output_dim = env.action_space.n 
decision_transformer_model = DecisionTransformer(input_dim, hidden_dim, output_dim)
model_path = "/Users/aryanmathur/Desktop/Sem5/decision_transformer_gotolocal_.pth"
decision_transformer_model.load_state_dict(torch.load(model_path))
decision_transformer_model.eval()  

obs, _ = env.reset()
done = False

print("Starting BabyAI-GoToLocal visualization...")

while not done:
    env.render() 
    if isinstance(obs, dict) and 'image' in obs:
        obs_tensor = torch.tensor(obs['image'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize pixel values
    elif isinstance(obs, np.ndarray):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    else:
        print("Error: Observation does not contain 'image' key or is not a valid array.")
        break

    with torch.no_grad():
        action_logits = decision_transformer_model(obs_tensor)
        action = torch.argmax(action_logits, dim=1).item()  
    obs, reward, done, truncated, info = env.step(action)
    done = done or truncated

print("Game Over")
env.close()
