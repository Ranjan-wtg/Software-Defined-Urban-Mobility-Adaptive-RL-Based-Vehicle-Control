# TVS Adaptive EV Control — Research & Competition Ready

This repository contains the publication-quality implementation for a **Software-Defined Urban Mobility** strategy, specifically designed for electric scooters like the **TVS iQube**. 

The core innovation is replacing static drive modes (Eco/Commute/Sport) with a **Reinforcement Learning (RL)** agent that dynamically adjusts vehicle behavior based on real-time context.

##  Competition "Killer Features"

To ensure project dominance, we've implemented high-impact research deliverables:

- ** Explainable AI (XAI)**: Context-aware decision logic visualizing "Why" the agent chooses specific behaviors.
- ** ECU-Ready Deployment**: Core intelligence exported to **ONNX format** for immediate integration into real-time embedded hardware.
- ** Comprehensive Ablation**: Proven **1.9x efficiency Gain** over industry-standard Eco/Sport modes.
- ** 1000-Episode Stress Test**: Validated robustness across extreme long-tail scenarios (steep hills, max payloads, dense congestion).
- ** Publication Visuals**: High-DPI (300) research plots including **Reward DNA Decomposition** and **Pareto Frontiers**.

##  Repository Structure

```
├── config.yaml          # All vehicle, environment, and training params
├── src/
│   ├── vehicle.py       # Physics-based EV dynamics model
│   ├── traffic.py       # Urban scenario generator
│   ├── environment.py   # Gymnasium environment wrapper
│   ├── reward.py        # Multi-objective reward (with XAI logic)
│   ├── baselines.py     # Static mode benchmark controllers
│   ├── train.py         # PPO training pipeline
│   ├── evaluate.py      # Detailed telemetry extraction
│   ├── visualize.py     # Publication-grade plotting suite
│   ├── ablation.py      # Strategic benchmark script
│   ├── export.py        # ONNX deployment exporter
│   └── stress_test.py   # 1000-episode robustness auditor
├── results/
│   ├── plots/           # Generated research visuals
│   └── stress_test_results.csv
└── models/
    ├── best_model/      # Trained PPO agents
    └── onnx/            # ECU-ready model binaries
```

##  Performance Benchmarks (20-Episode Average)

| Controller | Energy (Wh/km) | Ride Comfort | Adaptability |
|---|---|---|---|
| **RL Adaptive** | **3.06** | **0.85** | **Dynamic** |
| Eco Baseline | 5.98 | 0.00 | Static |
| Sport Mode | 25.81 | 0.00 | Fixed |


---
**Developed by Team Data Trio**  
Vellore Institute of Technology, Chennai
