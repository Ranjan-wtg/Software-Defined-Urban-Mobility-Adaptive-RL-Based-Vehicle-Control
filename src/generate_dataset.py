import pandas as pd
import numpy as np
from src.environment import AdaptiveEVEnv
from src.baselines import BaselineController
from tqdm import tqdm
import os

def generate_dataset(num_episodes=200):
    env = AdaptiveEVEnv()
    controller = BaselineController(mode='commute')
    
    all_data = []
    
    print(f"Generating {num_episodes} episodes of baseline data...")
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        scenario = info['scenario']
        
        done = False
        truncated = False
        t = 0
        
        while not (done or truncated):
            action = controller.get_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Record state
            all_data.append({
                'episode': episode,
                'step': t,
                'speed': obs[0] * 25.0,
                'accel': obs[1] * 5.0,
                'soc': obs[2],
                'traffic': obs[3],
                'gradient': obs[4] * 5.0,
                'payload': obs[5] * 100.0,
                'trip_type': int(obs[6] * 2.0),
                'dist_remaining': obs[7] * 10000.0,
                'jerk': obs[8] * 10.0,
                'comfort': obs[9],
                'reward': reward,
                'energy_wh': env.vehicle.energy_wh,
                'dist_m': env.vehicle.dist_m
            })
            
            obs = next_obs
            t += 1
            
    df = pd.DataFrame(all_data)
    os.makedirs('data/generated', exist_ok=True)
    df.to_parquet('data/generated/baseline_dataset.parquet')
    print(f"Dataset generated with {len(df)} rows. Saved to data/generated/baseline_dataset.parquet")
    return df

if __name__ == "__main__":
    generate_dataset(num_episodes=200)
