import numpy as np

class AdaptiveReward:
    def __init__(self, config):
        self.weights = config['reward_weights']
        self.env_params = config['environment']
        
    def get_weights(self, context):
        """
        Determines reward weightings based on trip context.
        context: {soc, trip_type, traffic_density}
        """
        weights = self.weights['default'].copy()
        
        # 1. Low Battery Priority
        if context['soc'] < 0.2:
            weights = self.weights['low_battery']
            
        # 2. Trip Type Priority
        elif context['trip_type'] == 1: # delivery
            weights = self.weights['delivery']
        elif context['trip_type'] == 0: # personal commute
            weights = self.weights['commute']
            
        # 3. Traffic Density Influence
        if context['traffic_density'] == 3: # congested
            # Blend with congested weights if not already dominated by low battery
            if context['soc'] >= 0.2:
                w_congested = np.array(self.weights['congested'])
                weights = (np.array(weights) * 0.5 + w_congested * 0.5).tolist()
                
        return weights

    def get_explanation(self, context):
        """
        Provides a human-readable explanation for the current weight choice.
        """
        if context['soc'] < 0.2:
            return f"Condition: Low Battery ({context['soc']*100:.0f}%). Priority: Energy Saving to ensure trip arrival."
        
        reasons = []
        if context['trip_type'] == 1:
            reasons.append("Trip Type: Delivery")
        elif context['trip_type'] == 0:
            reasons.append("Trip Type: Personal Commute")
            
        if context['traffic_density'] == 3:
            reasons.append("Traffic: Congested")
            
        if not reasons:
            return "Condition: Normal. Priority: Balanced efficiency and comfort."
        
        priority = "Comfort/Stability" if context['trip_type'] == 1 else "Passenger Comfort"
        if context['traffic_density'] == 3:
            priority += " & Energy Conservation"
            
        return f"Condition: {', '.join(reasons)}. Priority: {priority}."

    def compute(self, prev_state, action, current_state, scenario, dt):
        """
        Computes the multi-objective reward.
        Returns: total_reward, component_dictionary
        """
        # 1. Energy Efficiency (Wh consumed this step)
        energy_step = current_state['energy_wh'] - prev_state['energy_wh']
        # Normalize energy: typical max Wh/step is around 2.0 Wh (4kW * 0.5s / 3600 / 0.8eff)
        r_energy = -energy_step / 2.0 
        
        # 2. Comfort (Jerk and harshly braking)
        jerk = abs(current_state['jerk'])
        # ISO threshold is 0.9. Normalize by 5.0 as 'bad'
        r_comfort = - (jerk / 5.0)
        if current_state['a'] < -self.env_params['harsh_braking_threshold']:
            r_comfort -= 0.5 # Braking penalty
            
        # 3. Progress (Distance covered toward target)
        dist_step = current_state['v'] * dt
        # Normalize progress: typical speed is 11 m/s. Step is 5.5m.
        r_progress = dist_step / 10.0
        
        # 4. Safety (Speed limit adherence)
        # Find current segment speed limit
        segment_idx = min(int(current_state['dist_m'] / 500.0), len(scenario.speed_limits) - 1)
        speed_limit = scenario.speed_limits[segment_idx]
        r_safety = 0.0
        if current_state['v'] > speed_limit:
            r_safety = -(current_state['v'] - speed_limit) / speed_limit
            
        # Get Weights
        ctx = {
            'soc': current_state['soc'],
            'trip_type': scenario.trip_type,
            'traffic_density': scenario.traffic_density
        }
        w = self.get_weights(ctx)
        
        total_reward = (w[0] * r_energy) + (w[1] * r_comfort) + \
                       (w[2] * r_progress) + (w[3] * r_safety)
                       
        return total_reward, {
            'energy': r_energy,
            'comfort': r_comfort,
            'progress': r_progress,
            'safety': r_safety,
            'weights': w,
            'explanation': self.get_explanation(ctx)
        }
