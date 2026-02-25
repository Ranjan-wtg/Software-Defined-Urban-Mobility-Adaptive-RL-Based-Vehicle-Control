import os
import yaml
import numpy as np
import pandas as pd
import json
from stable_baselines3 import PPO
from src.environment import AdaptiveEVEnv
from src.baselines import BaselineController, RuleBasedController
from tqdm import tqdm

def evaluate_models(model_path, num_episodes=50, output_path='results/evaluation_comparison.csv'):
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Load trained model
    print(f"Loading model from {model_path}...")
    rl_agent = None
    if os.path.exists(model_path + ".zip") or os.path.exists(model_path):
        rl_agent = PPO.load(model_path)
    
    # 2. Environments
    env = AdaptiveEVEnv('config.yaml')
    
    # 3. Controllers
    controllers = {
        'RL_Adaptive': rl_agent,
        'Eco': BaselineController(mode='eco'),
        'Commute': BaselineController(mode='commute'),
        'Sport': BaselineController(mode='sport'),
        'RuleBased': RuleBasedController()
    }
    
    results = []
    telemetry_data = []
    
    # 4. Evaluation Loop
    print(f"Evaluating across {num_episodes} episodes...")
    for episode in tqdm(range(num_episodes)):
        # Important: use the same scenario for all controllers in this episode
        # Reset once to get the scenario
        obs, info = env.reset()
        scenario_params = env.scenario
        
        for name, agent in controllers.items():
            if agent is None and name == 'RL_Adaptive':
                continue
                
            # Re-initialize env state with the SAME scenario
            env.vehicle.reset()
            env.scenario = scenario_params
            env.steps = 0
            
            obs = env._get_obs()
            done = False
            truncated = False
            total_reward = 0
            ep_energy = 0
            ep_dist = 0
            
            while not (done or truncated):
                if name == 'RL_Adaptive':
                    action, _ = agent.predict(obs, deterministic=True)
                else:
                    action = agent.get_action(obs)
                
                next_obs, reward, done, truncated, step_info = env.step(action)
                total_reward += reward
                
                # Capture per-step telemetry
                telemetry_data.append({
                    'episode': episode,
                    'controller': name,
                    'step': env.vehicle.steps,
                    'speed_kmh': env.vehicle.v * 3.6,
                    'soc': env.vehicle.soc,
                    'jerk': abs(env.vehicle.jerk),
                    'energy_wh': env.vehicle.energy_wh,
                    'dist_m': env.vehicle.dist_m,
                    'reward': reward,
                    'action_throttle': float(action[0]) if isinstance(action, np.ndarray) else 0,
                    'action_brake': float(action[1]) if isinstance(action, np.ndarray) and len(action)>1 else 0,
                    'trip_type': scenario_params.trip_type,
                    'traffic': scenario_params.traffic_density,
                    'r_energy': step_info['reward_breakdown']['energy'],
                    'r_comfort': step_info['reward_breakdown']['comfort'],
                    'r_progress': step_info['reward_breakdown']['progress'],
                    'r_safety': step_info['reward_breakdown']['safety'],
                    'w_energy': step_info['reward_breakdown']['weights'][0],
                    'w_comfort': step_info['reward_breakdown']['weights'][1],
                    'w_progress': step_info['reward_breakdown']['weights'][2],
                    'w_safety': step_info['reward_breakdown']['weights'][3]
                })

                obs = next_obs
            
            # Record summary
            results.append({
                'episode': episode,
                'controller': name,
                'reward': total_reward,
                'energy_wh': env.vehicle.energy_wh,
                'dist_m': env.vehicle.dist_m,
                'wh_per_km': (env.vehicle.energy_wh / (env.vehicle.dist_m / 1000.0)) if env.vehicle.dist_m > 0 else 0,
                'comfort_score': env.vehicle.get_comfort_score(),
                'success': env.vehicle.dist_m >= scenario_params.trip_distance,
                'trip_type': scenario_params.trip_type,
                'traffic': scenario_params.traffic_density,
                'payload': scenario_params.payload_kg
            })
            
    # 5. Save Results
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Summary Statistics
    summary = df.groupby('controller').agg({
        'wh_per_km': 'mean',
        'comfort_score': 'mean',
        'reward': 'mean',
        'success': 'mean'
    }).reset_index()
    
    print("\nEvaluation Summary:")
    print(summary)
    
    summary.to_json(output_path.replace('.csv', '_summary.json'), orient='records')
    
    # Save detailed telemetry for visualization
    telemetry_df = pd.DataFrame(telemetry_data)
    telemetry_df.to_parquet(output_path.replace('.csv', '_telemetry.parquet'), index=False)
    print(f"Saved telemetry to {output_path.replace('.csv', '_telemetry.parquet')}")
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/ev_adaptive_policy_final')
    parser.add_argument('--num-episodes', type=int, default=50)
    parser.add_argument('--output', type=str, default='results/evaluation_comparison.csv')
    args = parser.parse_args()
    
    evaluate_models(args.model, args.num_episodes, args.output)
