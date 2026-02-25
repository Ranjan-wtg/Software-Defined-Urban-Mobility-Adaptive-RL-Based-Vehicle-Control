import os
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.environment import AdaptiveEVEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def run_ablation(model_path, num_episodes=20):
    print("Starting Ablation Study: Adaptive vs Fixed Modes...")
    
    if not os.path.exists(model_path + ".zip"):
        print("Model not found.")
        return

    model = PPO.load(model_path)
    env = AdaptiveEVEnv('config.yaml')
    
    # Load weights from config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    from src.baselines import BaselineController, RuleBasedController
    
    controllers = {
        'RL_Adaptive': model,
        'Eco_Baseline': BaselineController(mode='eco'),
        'Sport_Baseline': BaselineController(mode='sport'),
        'RuleBased': RuleBasedController()
    }
    
    results = []
    
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        scenario_params = env.scenario
        
        for name, agent in controllers.items():
            env.vehicle.reset()
            env.scenario = scenario_params
            env.steps = 0
            
            obs = env._get_obs()
            done = False
            truncated = False
            
            while not (done or truncated):
                if name == 'RL_Adaptive':
                    action, _ = agent.predict(obs, deterministic=True)
                else:
                    action = agent.get_action(obs)
                
                next_obs, reward, done, truncated, info = env.step(action)
                obs = next_obs
            
            results.append({
                'episode': episode,
                'controller': name,
                'wh_per_km': (env.vehicle.energy_wh / (env.vehicle.dist_m / 1000.0)) if env.vehicle.dist_m > 0 else 0,
                'comfort_score': env.vehicle.get_comfort_score()
            })
            
    df = pd.DataFrame(results)
    os.makedirs("results/plots", exist_ok=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='wh_per_km', y='comfort_score', hue='controller', s=100, alpha=0.7)
    
    plt.title("Ablation Study: Pareto-Dominance of Adaptive RL")
    plt.xlabel("Energy Consumption (Wh/km) - Lower is Better")
    plt.ylabel("Ride Comfort Score - Higher is Better")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("results/plots/ablation_pareto.png")
    
    # Bar Plot for Summary
    summary = df.groupby('controller').agg({'wh_per_km': 'mean', 'comfort_score': 'mean'}).reset_index()
    print("\nAblation Summary:")
    print(summary)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    summary.plot(x='controller', y='wh_per_km', kind='bar', ax=ax1, position=1, width=0.4, color='skyblue', label='Wh/km')
    summary.plot(x='controller', y='comfort_score', kind='bar', ax=ax2, position=0, width=0.4, color='orange', label='Comfort')
    
    ax1.set_ylabel("Wh/km")
    ax2.set_ylabel("Comfort Score")
    plt.title("Adaptive Advantage: Outperforming Single-Objective Fixed Modes")
    plt.tight_layout()
    plt.savefig("results/plots/ablation_summary.png")
    
    df.to_csv("results/ablation_results.csv", index=False)
    print("Ablation plots saved to results/plots/")

if __name__ == "__main__":
    run_ablation("models/best_model/best_model", 20)
