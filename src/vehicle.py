import numpy as np

class EVehicleModel:
    def __init__(self, config):
        self.params = config['vehicle']
        self.env_params = config['environment']
        
        # Physical constants
        self.g = 9.81
        
        # State variables
        self.reset()
        
    def reset(self):
        self.v = 0.0          # velocity (m/s)
        self.a = 0.0          # acceleration (m/s2)
        self.jerk = 0.0       # jerk (m/s3)
        self.soc = 1.0        # state of charge (0.0 - 1.0)
        self.energy_wh = 0.0  # cumulative energy consumed (Wh)
        self.dist_m = 0.0     # cumulative distance (m)
        
        self.harsh_braking_count = 0
        self.rms_jerk_acc = 0.0
        self.steps = 0
        
    def step(self, throttle, brake, regen, gradient_deg, dt=0.5, payload_kg=0.0):
        """
        Calculates one timestep of vehicle dynamics.
        throttle: 0.0 - 1.0
        brake: 0.0 - 1.0
        regen: 0.0 - 1.0 (proportion of braking torque that is regenerative)
        """
        m = self.params['mass_base'] + payload_kg
        theta = np.deg2rad(gradient_deg)
        
        # 1. Calculate Forces
        # Rolling resistance
        F_rolling = self.params['rolling_resistance_coeff'] * m * self.g * np.cos(theta)
        
        # Aerodynamic drag
        F_aero = 0.5 * self.params['air_density'] * self.params['drag_coeff'] * \
                 self.params['frontal_area_m2'] * (self.v ** 2)
        
        # Gradient resistance
        F_gradient = m * self.g * np.sin(theta)
        
        # Motor force
        if throttle > 0:
            # P = F * v -> F = P / v
            # Use v + small epsilon to avoid division by zero
            # Power is capped by peak power
            P_motor = throttle * self.params['motor_peak_power_w']
            F_motor = P_motor / (max(self.v, 1.0))
        else:
            F_motor = 0.0
            
        # Braking force
        F_brake = brake * m * self.env_params['harsh_braking_threshold'] * 2.0 # simplified max braking
        
        # Net Force
        F_net = F_motor - F_rolling - F_aero - F_gradient - F_brake
        
        # 2. Update Kinematics
        a_new = F_net / m
        
        # Cap top speed
        v_max = self.params['top_speed_kmh'] / 3.6
        if self.v >= v_max and a_new > 0:
            a_new = 0.0
            
        v_new = max(0.0, self.v + a_new * dt)
        
        # Jerk
        jerk_new = (a_new - self.a) / dt
        
        # 3. Energy Consumption
        # Instantaneous power demand
        # P = F_motor * v / efficiency
        if throttle > 0:
            P_demand = (F_motor * self.v) / self.params['motor_efficiency']
        else:
            P_demand = 0.0
            
        # Regenerative braking
        if brake > 0:
            # Energy recovered = F_brake * v * regen_intensity * efficiency
            # Only portion of braking is regen
            P_regen = F_brake * self.v * regen * self.params['regen_efficiency']
        else:
            P_regen = 0.0
            
        energy_step_wh = (P_demand - P_regen) * (dt / 3600.0)
        self.energy_wh += energy_step_wh
        
        # SoC update
        wh_capacity = self.params['battery_capacity_wh']
        self.soc = max(0.0, self.soc - (energy_step_wh / wh_capacity))
        
        # 4. Comfort Metrics
        if a_new < -self.env_params['harsh_braking_threshold']:
            self.harsh_braking_count += 1
            
        self.rms_jerk_acc += jerk_new**2
        self.steps += 1
        
        # Update state
        self.a = a_new
        self.v = v_new
        self.jerk = jerk_new
        self.dist_m += self.v * dt
        
        return {
            'v': self.v,
            'a': self.a,
            'jerk': self.jerk,
            'soc': self.soc,
            'energy_wh': self.energy_wh,
            'dist_m': self.dist_m
        }

    def get_comfort_score(self):
        if self.steps == 0: return 1.0
        rms_jerk = np.sqrt(self.rms_jerk_acc / self.steps)
        # Normalize: 1.0 is perfect, 0.0 is very uncomfortable
        # ISO 2631-1 suggests 0.9 m/s3 as threshold
        score = np.exp(-rms_jerk / self.env_params['max_jerk_threshold'])
        # Penalty for harsh braking
        penalty = 0.1 * self.harsh_braking_count / (self.steps / 20.0) # approx 10s per penalty scale
        return max(0.0, score - penalty)

if __name__ == "__main__":
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = EVehicleModel(config)
    print("Testing Vehicle Model...")
    for i in range(20):
        state = model.step(throttle=0.8, brake=0.0, regen=0.0, gradient_deg=0, dt=0.5)
        print(f"Step {i+1}: v={state['v']:.2f} m/s, a={state['a']:.2f} m/s2, soc={state['soc']:.4f}")
    
    print("\nBraking...")
    for i in range(10):
        state = model.step(throttle=0.0, brake=0.5, regen=0.5, gradient_deg=0, dt=0.5)
        print(f"Step {i+21}: v={state['v']:.2f} m/s, a={state['a']:.2f} m/s2, soc={state['soc']:.4f}")
    
    print(f"\nFinal Comfort Score: {model.get_comfort_score():.2f}")
