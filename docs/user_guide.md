# TeleLLM User Guide

**Quick Start Instructions for Reproducing Project Results**

---

## 1. Environment Setup

Ensure you have Python 3.10+ installed.

```bash
# Install dependencies
pip install torch numpy matplotlib pandas pydantic joblib
```

*Note: The system implements a custom high-fidelity network simulation. OpenAI Gym is not required.*

---

## 2. Running Benchmarks

The primary tool for performance evaluation is `run/benchmark.py`. This script executes a comprehensive suite of tests including Multi-Agent RL (MARL) synergy and Intent-Based Networking (IBN) polymorphism.

**Command:**
```bash
python run/benchmark.py
```

**What happens:**
1.  Loads the pre-trained PPO checkpoints:
    *   MEC Agent: `models/mec_policy.pth`
    *   HO Agent: `models/ho_policy.pth`
2.  Executes multiple test scenarios:
    *   **Synergy Analysis**: Evaluates coordination between HO and MEC agents.
    *   **IBN Polymorphism**: Validates agent adaptation to different intent vectors (Latency vs. Energy focus).
3.  **Output:**
    *   **Data**: `data/synergy_analysis.csv`, `data/polymorphism_analysis.csv`
    *   **Plots**: `data/plots/reliability_benchmark.png`

---

## 3. Training the Agents

To retrain the MARL agents from scratch, use the `run/train.py` script.

**Command:**
```bash
python run/train.py
```

**Configuration:**
Training parameters (learning rate, discount factor, etc.) are defined within the script. The training process uses parallelized simulation environments for efficiency.

**Output:**
*   Checkpoints are saved periodically to the `models/` directory.
*   Training metrics are recorded in `data/training_metrics.csv`.

---

## 4. Explainability and Logs

The TeleLLM Orchestrator generates detailed logs to provide transparency into its decision-making process.

*   **Reasoning Logs**: Located in `logs/` (e.g., `logs/urllc_storm_reasoning.json`).
*   **Performance Metrics**: Step-by-step metrics are captured during benchmark execution.

Example Logic Trace:
> "Metric: Latency=85ms (>60ms budget). State: CONGESTED. Action: Switch to SURVIVAL mode. Parameter Change: HOM=0dB, TTT=0s."
