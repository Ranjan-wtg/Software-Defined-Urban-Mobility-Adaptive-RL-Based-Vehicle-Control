import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os

from src.vehicle import EVehicleModel
from src.traffic import UrbanScenarioGenerator
from src.reward import AdaptiveReward

class AdaptiveEVEnv(gym.Env):
    """
    Gymnasium environment for Software-Defined Urban Mobility.
    """
    def __init__(self, config_path='config.yaml'):
        super(AdaptiveEVEnv, self).__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.vehicle = EVehicleModel(self.config)
        self.traffic = UrbanScenarioGenerator(self.config)
        self.reward_fn = AdaptiveReward(self.config)
        
        # Action Space: [throttle_smoothing, regen_intensity, accel_limit_factor]
        # Range: [0, 1] for all
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # Observation Space (10-dim):
        # [speed, accel, soc, traffic, gradient, payload, trip_type, dist_rem, jerk, comfort]
        self.observation_space = spaces.Box(low=-1, high=2, shape=(10,), dtype=np.float32)
        
        self.dt = self.config['environment']['dt']
        
    def _get_obs(self):
        # Normalize/Scale values for the agent
        v_norm = self.vehicle.v / 25.0       # scale by 90 km/h
        a_norm = self.vehicle.a / 5.0        # scale by max accel
        soc = self.vehicle.soc
        traffic = self.scenario.traffic_density / 3.0
        
        # Find current gradient
        seg_idx = min(int(self.vehicle.dist_m / 500.0), len(self.scenario.gradients) - 1)
        gradient = self.scenario.gradients[seg_idx] / 5.0
        
        payload = self.scenario.payload_kg / 100.0
        trip = self.scenario.trip_type / 2.0
        dist_rem = max(0.0, (self.scenario.trip_distance - self.vehicle.dist_m) / 10000.0)
        jerk = self.vehicle.jerk / 10.0
        comfort = self.vehicle.get_comfort_score()
        
        return np.array([
            v_norm, a_norm, soc, traffic, gradient, 
            payload, trip, dist_rem, jerk, comfort
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.vehicle.reset()
        self.scenario = self.traffic.generate()
        self.steps = 0
        
        info = {'scenario': self.scenario.__dict__}
        return self._get_obs(), info

    def step(self, action):
        # Action Decoding
        # action[0]: throttle_smoothing (actually just throttle in this simplified version)
        # action[1]: regen_intensity
        # action[2]: accel_limit_factor (scales max throttle power)
        
        throttle_cmd = float(action[0])
        regen_cmd = float(action[1])
        accel_limit = float(action[2])
        
        # Apply accel limit to throttle
        effective_throttle = throttle_cmd * accel_limit
        
        # For simplicity in this sprint, we derive braking from the scenario's "natural" stop-go
        # The RL agent controls "how" it follows the speed profile
        
        # 1. Determine target speed based on scenario
        target_v = self.scenario.avg_speed_target
        
        # Check for traffic lights (Dynamic Red/Green logic)
        for stop_pos in self.scenario.stop_positions:
            # Simple 30s Red / 30s Green cycle
            is_red = (self.steps % 120) < 60  # 0.5s dt -> 60 steps = 30s
            if is_red and abs(self.vehicle.dist_m - stop_pos) < 15.0:
                target_v = 0.0 
                break
        
        # Simple heuristic for throttle/brake based on target_v
        if self.vehicle.v < target_v - 0.5:
            throttle = effective_throttle
            brake = 0.0
        elif self.vehicle.v > target_v + 0.5:
            throttle = 0.0
            brake = 0.5 # constant braking for simplicity
        else:
            throttle = 0.1
            brake = 0.0
            
        prev_state = {
            'v': self.vehicle.v,
            'a': self.vehicle.a,
            'jerk': self.vehicle.jerk,
            'soc': self.vehicle.soc,
            'energy_wh': self.vehicle.energy_wh,
            'dist_m': self.vehicle.dist_m
        }
        
        # Find current gradient
        seg_idx = min(int(self.vehicle.dist_m / 500.0), len(self.scenario.gradients) - 1)
        grad = self.scenario.gradients[seg_idx]
        
        # Physics Step
        current_state = self.vehicle.step(
            throttle=throttle, 
            brake=brake, 
            regen=regen_cmd, 
            gradient_deg=grad, 
            dt=self.dt, 
            payload_kg=self.scenario.payload_kg
        )
        
        # Reward
        reward, reward_info = self.reward_fn.compute(
            prev_state, action, current_state, self.scenario, self.dt
        )
        
        # Termination
        self.steps += 1
        terminated = False
        truncated = False
        
        if self.vehicle.dist_m >= self.scenario.trip_distance:
            terminated = True
        elif self.vehicle.soc <= 0.0:
            terminated = True
        elif self.steps >= self.config['environment']['max_episode_steps']:
            truncated = True
            
        obs = self._get_obs()
        info = {
            'reward_breakdown': reward_info,
            'is_success': self.vehicle.dist_m >= self.scenario.trip_distance
        }
        
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    
    print("Validating Environment...")
    env = AdaptiveEVEnv()
    check_env(env)
    print("Environment validation successful!")
    
    obs, info = env.reset()
    print(f"Initial Obs: {obs}")
    for _ in range(5):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        print(f"Step: reward={reward:.4f}, v={obs[0]*25:.2f} m/s, soc={obs[2]:.4f}")
