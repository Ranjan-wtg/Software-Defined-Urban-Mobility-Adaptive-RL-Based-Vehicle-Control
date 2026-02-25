import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Scenario:
    trip_distance: float
    trip_type: int
    payload_kg: float
    traffic_density: int
    avg_speed_target: float
    stop_positions: List[float]
    speed_limits: List[float]
    gradients: List[float]

class UrbanScenarioGenerator:
    def __init__(self, config):
        self.params = config['scenarios']
        self.env_params = config['environment']
        
    def generate(self):
        # 1. Basic Trip Params
        trip_dist = np.random.uniform(
            self.env_params['target_distance_min'],
            self.env_params['target_distance_max']
        )
        
        trip_type = np.random.choice([0, 1, 2]) # commute, delivery, leisure
        
        # Payload based on trip type
        if trip_type == 1: # delivery
            payload = self.params['payloads']['delivery']
        else:
            payload = np.random.choice([
                self.params['payloads']['solo'],
                self.params['payloads']['pillion']
            ])
            
        traffic_density = np.random.choice([0, 1, 2, 3]) # low, med, high, congested
        
        # 2. Speed Profile
        avg_speed_base = np.random.uniform(
            self.params['avg_speed_range'][0],
            self.params['avg_speed_range'][1]
        ) / 3.6
        
        # Traffic multiplier
        traffic_multipliers = [1.0, 0.7, 0.4, 0.2]
        avg_speed_target = avg_speed_base * traffic_multipliers[traffic_density]
        
        # 3. Stops
        # Stops per km
        stop_freq = np.random.uniform(
            self.params['stop_frequency'][0],
            self.params['stop_frequency'][1]
        )
        num_stops = int((trip_dist / 1000.0) * stop_freq)
        stop_positions = np.sort(np.random.uniform(100, trip_dist - 100, num_stops)).tolist()
        
        # 4. Gradients (simplified)
        num_segments = int(trip_dist / 500.0) + 1
        gradients = np.random.normal(0, 2, num_segments).tolist()
        gradients = [max(min(g, 5), -5) for g in gradients]
        
        return Scenario(
            trip_distance=trip_dist,
            trip_type=trip_type,
            payload_kg=payload,
            traffic_density=traffic_density,
            avg_speed_target=avg_speed_target,
            stop_positions=stop_positions,
            speed_limits=[60/3.6] * num_segments, # simplified
            gradients=gradients
        )

if __name__ == "__main__":
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    generator = UrbanScenarioGenerator(config)
    scenario = generator.generate()
    print("Generated Scenario:")
    print(f"Distance: {scenario.trip_distance/1000:.2f} km")
    print(f"Trip Type: {['Commute', 'Delivery', 'Leisure'][scenario.trip_type]}")
    print(f"Payload: {scenario.payload_kg} kg")
    print(f"Traffic: {['Low', 'Medium', 'High', 'Congested'][scenario.traffic_density]}")
    print(f"Stops: {len(scenario.stop_positions)}")
    print(f"Avg Target Speed: {scenario.avg_speed_target * 3.6:.2f} km/h")
