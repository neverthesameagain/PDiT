import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env = gym.make("BabyAI-GoToLocal-v0", render_mode="rgb_array")
    env.action_space.seed(seed)
    env = gym.make("BabyAI-GoToLocal-v0", render_mode="rgb_array")
    env = RGBImgPartialObsWrapper(env)  
    env = ImgObsWrapper(env)  

    model = PPO(
        "CnnPolicy", env, verbose=1, tensorboard_log="./ppo_gotolocal_tensorboard/",
        n_steps=256,  
        batch_size=64,  
        gae_lambda=0.95, 
        gamma=0.99,  
        n_epochs=10,  
        ent_coef=0.01,  
        learning_rate=3e-4,  
        vf_coef=0.5 
    )

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/', name_prefix='ppo_gotolocal')
    eval_env = gym.make("BabyAI-GoToLocal-v0", render_mode="rgb_array")
    eval_env.action_space.seed(seed)  
    eval_env = RGBImgPartialObsWrapper(eval_env)
    eval_env = ImgObsWrapper(eval_env)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./best_model/', log_path='./logs/', eval_freq=5000, deterministic=True, render=False)

    model.learn(total_timesteps=200000, callback=[checkpoint_callback, eval_callback])

    model.save("ppo_gotolocal_final")

    test_env = gym.make("BabyAI-GoToLocal-v0", render_mode="human")
    test_env = RGBImgPartialObsWrapper(test_env)
    test_env = ImgObsWrapper(test_env)

    obs, _ = test_env.reset()
    mission = obs['mission'] if isinstance(obs, dict) and 'mission' in obs else test_env.unwrapped.mission
    print(f"Mission: {mission}")  
    done = False
    goal_position = None
    if 'go to' in mission:
        mission_parts = mission.split('go to ')
        if len(mission_parts) > 1:
            goal_color_object = mission_parts[1].strip()
            print(f"Identified Goal: {goal_color_object}")  

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        test_env.render()

    test_env.close()

if __name__ == "__main__":
    main()
