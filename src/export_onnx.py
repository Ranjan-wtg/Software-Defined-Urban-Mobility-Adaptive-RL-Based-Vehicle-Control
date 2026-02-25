import os
import torch as th
from stable_baselines3 import PPO
import yaml
import numpy as np
from src.environment import AdaptiveEVEnv

def export_onnx(model_path, output_path):
    print(f"Loading PPO model from {model_path}...")
    model = PPO.load(model_path, device="cpu")
    
    # 1. Policy wrapper for ONNX export
    # SB3 models include a lot of logic (optimizers, etc.) that ONNX doesn't need
    class OnnxablePolicy(th.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, observation):
            # We only need the actor's mean action for deterministic inference
            # This avoids complex distribution logic that often fails ONNX export
            return self.policy.action_net(self.policy.mlp_extractor.forward(observation)[0])

    onnxable_policy = OnnxablePolicy(model.policy)
    
    # 2. Dummy input
    dummy_input = th.randn(1, 10) # 10-dim observation
    
    # 3. Export
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Exporting to {output_path}...")
    
    th.onnx.export(
        onnxable_policy,
        dummy_input,
        output_path,
        opset_version=11,
        input_names=["input"],
        output_names=["value", "action"],
        dynamic_axes={"input": {0: "batch_size"}, "value": {0: "batch_size"}, "action": {0: "batch_size"}}
    )
    
    print("Export complete!")
    
    # 4. Verification with ONNX Runtime (if installed)
    try:
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(output_path)
        input_name = ort_sess.get_inputs()[0].name
        
        # Test inference
        test_obs = np.random.randn(1, 10).astype(np.float32)
        outputs = ort_sess.run(None, {input_name: test_obs})
        print(f"Inference test successful! Output shape: {outputs[1].shape}")
        
        # Benchmark latency
        import time
        start = time.time()
        for _ in range(100):
            _ = ort_sess.run(None, {input_name: test_obs})
        latency = (time.time() - start) * 10
        print(f"Average Inference Latency on CPU: {latency:.2f} ms")
        
    except Exception as e:
        print(f"ONNX Runtime verification skipped: {e}")

if __name__ == "__main__":
    import argparse
    import sys
    # Force UTF-8 encoding for stdout to handle emojis in logs
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/ev_adaptive_policy_final')
    parser.add_argument('--output', type=str, default='models/ev_adaptive_policy.onnx')
    args = parser.parse_args()
    
    try:
        export_onnx(args.model, args.output)
    except Exception as e:
        print(f"Export failed: {e}")
