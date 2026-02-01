# A Polymorphic Intent-Based Networking System for 6G

This repository contains the final implementation and results for a Polymorphic Intent-Based Networking (IBN) system designed for 6G Multi-access Edge Computing (MEC) and Handover (HO).

The system uses Multi-Agent Reinforcement Learning (MARL) to optimize network performance based on dynamic user intents (Polymorphism).

## 📂 Project Structure

- `run/`: Executable scripts for benchmarking and training.
  - `benchmark.py`: Reproduces final thesis results.
  - `train.py`: Cooperative MARL training script.
- `models/`: Professional PPO policy checkpoints.
  - `mec_policy.pth`: Final MEC agent weights.
  - `ho_policy.pth`: Final HO agent weights.
- `data/`: Analysis results and visualizations.
  - `synergy_analysis.csv`: Multi-agent ablation results.
  - `polymorphism_analysis.csv`: Intent-shifting results.
  - `training_metrics.csv`: MARL training logs.
  - `plots/reliability_benchmark.png`: System performance comparison.
- `src/`: Core implementation logic.
- `docs/`: Technical documentation.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch
- NumPy, Pandas, Matplotlib, Seaborn
- tqdm, joblib

### Reproducing Thesis Results
To run the final benchmark and reproduce the synergy and polymorphic performance metrics:

```powershell
python run/benchmark.py
```

This will load the final models from `models/` and export updated results and plots to `data/`.

## 🛡️ Performance Highlights (Set B)
- **MEC Offloading Success Rate**: ~90% under high-stress conditions.
- **Polymorphic Reliability (Max-Power)**: ~70% success rate with throughput focus.
- **Synergy**: Integrated MARL agents outperform baseline heuristics by >4x in service reliability.
