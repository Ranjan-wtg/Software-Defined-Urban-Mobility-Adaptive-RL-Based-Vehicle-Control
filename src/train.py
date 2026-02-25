import os
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from src.environment import AdaptiveEVEnv

def train():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 1. Environment Creation
    # Using 4 parallel environments for faster training
    n_envs = 4
    env = make_vec_env(lambda: AdaptiveEVEnv('config.yaml'), n_envs=n_envs)
    
    # 2. PPO Model Setup
    training_params = config['training']
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=training_params['learning_rate'],
        n_steps=training_params['n_steps'],
        batch_size=training_params['batch_size'],
        n_epochs=training_params['n_epochs'],
        gamma=training_params['gamma'],
        gae_lambda=training_params['gae_lambda'],
        clip_range=training_params['clip_range'],
        ent_coef=training_params['ent_coef'],
        policy_kwargs={"net_arch": training_params['net_arch']},
        verbose=1,
        tensorboard_log="./logs/ppo_ev_tensorboard/"
    )
    
    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=25000, 
        save_path='./models/', 
        name_prefix='rl_model'
    )
    
    # Evaluation environment
    eval_env = AdaptiveEVEnv('config.yaml')
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./models/best_model/',
        log_path='./logs/eval/', 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    # 4. Training
    print(f"Starting training for {training_params['total_timesteps']} steps...")
    model.learn(
        total_timesteps=training_params['total_timesteps'],
        callback=[checkpoint_callback, eval_callback]
    )
    
    # 5. Save final model
    model.save("models/ev_adaptive_policy_final")
    print("Training complete! Model saved to models/ev_adaptive_policy_final")

if __name__ == "__main__":
    train()
