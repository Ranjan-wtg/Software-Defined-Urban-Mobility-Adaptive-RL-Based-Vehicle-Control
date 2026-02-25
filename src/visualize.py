import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Set aesthetic style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "savefig.dpi": 300
})

def generate_radar_chart(summary_data, output_dir):
    """Generates a research-grade radar chart for multi-objective comparison."""
    from math import pi
    
    # Metrics to plot
    metrics = ['wh_per_km', 'comfort_score', 'success', 'reward']
    num_vars = len(metrics)
    
    # Normalize data for radar chart (0-1 scale)
    df = summary_data.copy()
    for m in metrics:
        if m == 'wh_per_km': # Inverse (lower is better)
            df[m] = 1 - (df[m] - df[m].min()) / (df[m].max() - df[m].min() + 1e-6)
        else:
            df[m] = (df[m] - df[m].min()) / (df[m].max() - df[m].min() + 1e-6)
    
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    for i, row in df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['controller'])
        ax.fill(angles, values, alpha=0.1)
        
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], ['Efficiency', 'Comfort', 'Safety', 'Reliability'])
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Multi-Objective Performance Radar", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_comparison.png"))
    plt.close()

def plot_telemetry_comparison(telemetry_df, output_dir):
    """Plots speed, SoC, and Jerk for a representative episode."""
    # Pick first episode of each controller
    controllers = telemetry_df['controller'].unique()
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    for ctrl in controllers:
        data = telemetry_df[(telemetry_df['controller'] == ctrl) & (telemetry_df['episode'] == 0)]
        axes[0].plot(data['step'], data['speed_kmh'], label=ctrl, linewidth=1.5)
        axes[1].plot(data['step'], data['soc'] * 100, label=ctrl, linewidth=1.5)
        axes[2].plot(data['step'], data['jerk'], label=ctrl, alpha=0.6)
        
    axes[0].set_ylabel("Speed (km/h)")
    axes[1].set_ylabel("SoC (%)")
    axes[2].set_ylabel("Jerk ($m/s^3$)")
    axes[2].set_xlabel("Time Steps")
    
    axes[0].set_title("Vehicle Dynamics Comparison (Single Trip)")
    axes[0].legend(loc='upper right', ncol=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "telemetry_comparison.png"))
    plt.close()

def plot_pareto_front(summary_data, output_dir):
    """Visualizes the Energy vs Comfort Pareto Front."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=summary_data, x='wh_per_km', y='comfort_score', 
                    hue='controller', s=200, style='controller', markers=True)
    
    # Label each point
    for i in range(summary_data.shape[0]):
        plt.text(summary_data.wh_per_km[i]+0.1, summary_data.comfort_score[i], 
                 summary_data.controller[i], fontsize=10, weight='bold')

    plt.xlabel(r"Energy Consumption (Wh/km) $\leftarrow$ Better")
    plt.ylabel(r"Ride Comfort Score $\rightarrow$ Better")
    plt.title("Multi-Objective Optimization Pareto Frontier")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_frontier.png"))
    plt.close()

def plot_reward_dna(telemetry_df, output_dir):
    """Visualizes how different reward components contribute to the total agent score."""
    data = telemetry_df[(telemetry_df['controller'] == 'RL_Adaptive') & (telemetry_df['episode'] == 0)]
    
    components = ['r_energy', 'r_comfort', 'r_progress', 'r_safety']
    labels = ['Energy Efficiency', 'Ride Comfort', 'Navigation Progress', 'Safety (Speed)']
    colors = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444']
    
    plt.figure(figsize=(12, 6))
    plt.stackplot(data['step'], 
                  [data[c] for c in components], 
                  labels=labels, colors=colors, alpha=0.8)
    
    plt.plot(data['step'], data['reward'], color='black', linewidth=1.5, label='Total Step Reward', linestyle='--')
    
    plt.title("RL Agent 'Reward DNA': Component Decomposition")
    plt.ylabel("Reward Signal Magnitude")
    plt.xlabel("Trip Time Steps")
    plt.legend(loc='lower left', ncol=2)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_dna.png"))
    plt.close()

def plot_weight_evolution(telemetry_df, output_dir):
    """Shows how the adaptive reward weights evolve throughout the trip."""
    data = telemetry_df[(telemetry_df['controller'] == 'RL_Adaptive') & (telemetry_df['episode'] == 0)]
    
    weights = ['w_energy', 'w_comfort', 'w_progress', 'w_safety']
    labels = ['Energy weight', 'Comfort weight', 'Progress weight', 'Safety weight']
    
    plt.figure(figsize=(12, 5))
    for i, w in enumerate(weights):
        plt.plot(data['step'], data[w], label=labels[i], linewidth=2)
        
    plt.title("Adaptive Priority Evolution: Context-Aware Weighting")
    plt.ylabel("Assigned Priority Weight")
    plt.xlabel("Trip Time Steps")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weight_evolution.png"))
    plt.close()

def plot_efficiency_distribution(summary_df, output_dir):
    """Violin plot comparing energy efficiency distribution across multiple episodes."""
    # Ensure raw results are used for distribution
    raw_results_file = "results/judge_comparison.csv"
    if not os.path.exists(raw_results_file): return
    
    df = pd.read_csv(raw_results_file)
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='controller', y='wh_per_km', inner="points", palette="muted")
    
    plt.title("Energy Efficiency Variance Across Fleet Scenarios")
    plt.ylabel("Energy Consumption (Wh/km)")
    plt.xlabel("Control Strategy")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficiency_distribution.png"))
    plt.close()

def plot_context_sensitivity(telemetry_df, output_dir):
    """XAI: Heatmap showing correlation between context features and priority weights."""
    features = ['soc', 'traffic', 'trip_type']
    weights = ['w_energy', 'w_comfort', 'w_progress', 'w_safety']
    
    # Calculate correlations
    corr_matrix = telemetry_df[features + weights].corr().loc[features, weights]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    
    plt.title("XAI: Context-Priority Sensitivity Map")
    plt.ylabel("Environmental Context")
    plt.xlabel("Reward Priority Weights")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "xai_context_sensitivity.png"))
    plt.close()

def plot_decision_space(telemetry_df, output_dir):
    """XAI: Mapping the agent's policy decision space (Action vs State)."""
    data = telemetry_df[telemetry_df['controller'] == 'RL_Adaptive'].copy()
    
    # Effective action (Throttle - Brake)
    data['net_action'] = data['action_throttle'] - data['action_brake']
    
    plt.figure(figsize=(11, 7))
    scatter = plt.scatter(data['speed_kmh'], data['traffic'], 
                        c=data['net_action'], cmap='RdYlGn', alpha=0.6, s=50)
    
    plt.colorbar(scatter, label='Net Action (Green=Accel, Red=Braking)')
    plt.xlabel("Vehicle Speed (km/h)")
    plt.ylabel("Traffic Density (0-3)")
    plt.title("XAI: Policy Decision Space (Speed vs Traffic)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "xai_decision_space.png"))
    plt.close()

if __name__ == "__main__":
    os.makedirs("results/plots", exist_ok=True)
    
    # Load data
    summary_file = "results/judge_comparison_summary.json"
    telemetry_file = "results/judge_comparison_telemetry.parquet"
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary_data = pd.DataFrame(json.load(f))
        generate_radar_chart(summary_data, "results/plots")
        plot_pareto_front(summary_data, "results/plots")
        plot_efficiency_distribution(summary_data, "results/plots")
    
    if os.path.exists(telemetry_file):
        telemetry_df = pd.read_parquet(telemetry_file)
        plot_telemetry_comparison(telemetry_df, "results/plots")
        plot_reward_dna(telemetry_df, "results/plots")
        plot_weight_evolution(telemetry_df, "results/plots")
        plot_context_sensitivity(telemetry_df, "results/plots")
        plot_decision_space(telemetry_df, "results/plots")
        
    print("Publication quality plots generated in results/plots/")
