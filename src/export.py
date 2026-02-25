import torch
import torch.nn as nn
from stable_baselines3 import PPO
import os
import onnx
import onnxruntime as ort
import numpy as np

class OnnxablePolicy(nn.Module):
    """A wrapper to make the SB3 policy ONNX-exportable."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        # SB3 policies return (action, value, log_prob)
        # We only need the deterministic action for final deployment
        action, _ = self.policy.predict_values(observation) # Placeholder-ish for raw logic
        # For PPO, we usually want the distribution mean or direct action
        # The simplest way for PPO in SB3 is using the policy object directly
        return self.policy(observation)

def export_to_onnx(model_path, export_path):
    print(f"Loading SB3 model from {model_path}...")
    model = PPO.load(model_path, device="cpu")
    
    # We create a dummy input with the observation space shape
    # AdaptiveEVEnv observation space is Box(low=-1, high=1, shape=(10,))
    dummy_input = torch.randn(1, 10)
    
    # Extract the policy
    policy = model.policy.to("cpu")
    
    # For ONNX export, we only care about the action (actor)
    # Wrap it to return just the action
    class ActorOnly(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.actor = policy.mlp_extractor.policy_net
            self.action_net = policy.action_net
            
        def forward(self, x):
            latent = self.actor(x)
            return self.action_net(latent)

    actor_model = ActorOnly(policy)
    actor_model.eval()

    print(f"Exporting to ONNX: {export_path}...")
    torch.onnx.export(
        actor_model,
        dummy_input,
        export_path,
        opset_version=12,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={'observation': {0: 'batch_size'}, 'action': {0: 'batch_size'}}
    )
    
    # Verify with onnxruntime
    print("Verifying ONNX export with ONNXRuntime...")
    session = ort.InferenceSession(export_path)
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)
    
    print(f"Success! ONNX model saved and verified. Output shape: {ort_outs[0].shape}")

if __name__ == "__main__":
    os.makedirs("models/onnx", exist_ok=True)
    model_file = "models/best_model/best_model"
    if os.path.exists(model_file + ".zip"):
        export_to_onnx(model_file, "models/onnx/tvs_adaptive_ev.onnx")
    else:
        print(f"Model file {model_file} not found. Please train the model first.")
