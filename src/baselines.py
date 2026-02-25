import numpy as np

class BaselineController:
    def __init__(self, mode='eco'):
        self.mode = mode
        
    def get_action(self, obs):
        """
        Returns [throttle_smoothing, regen_intensity, accel_limit_factor]
        """
        if self.mode == 'eco':
            # High smoothing, high regen, low accel limit
            return np.array([0.5, 0.9, 0.4], dtype=np.float32)
        elif self.mode == 'commute':
            # Med smoothing, med regen, med accel limit
            return np.array([0.7, 0.5, 0.7], dtype=np.float32)
        elif self.mode == 'sport':
            # Low smoothing, low regen, high accel limit
            return np.array([1.0, 0.1, 1.0], dtype=np.float32)
        else: # default/rule-based
            return np.array([0.7, 0.5, 0.7], dtype=np.float32)

class RuleBasedController:
    def __init__(self):
        pass
        
    def get_action(self, obs):
        # obs: [speed, accel, soc, traffic, gradient, payload, trip, dist_rem, jerk, comfort]
        soc = obs[2]
        traffic = obs[3]
        
        # Simple rules:
        # If low battery, use eco-like settings
        if soc < 0.2:
            return np.array([0.5, 0.9, 0.4], dtype=np.float32)
        # If high traffic, limit acceleration
        elif traffic > 0.6:
            return np.array([0.6, 0.6, 0.5], dtype=np.float32)
        else:
            return np.array([0.8, 0.4, 0.8], dtype=np.float32)
