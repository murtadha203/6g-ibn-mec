# Handover Optimization Agent (PPO)

The **Handover Optimization Agent** is a specialized Proximal Policy Optimization (PPO) agent designed for radio resource management in dynamic network environments.

---

## 1. Design Philosophy: Volatility Suppression

To prevent excessive "ping-pong" handovers, the agent incorporates a volatility suppression mechanism within its reward structure.

### Objective Function
The agent optimizes a multi-objective reward function:
$$ R = R_{QoS} - \lambda \cdot C_{Switch} $$

Where:
*   $R_{QoS}$ represents Quality of Service metrics (RSRP, SINR, Throughput).
*   $C_{Switch}$ is a penalty applied to each handover event.
*   $\lambda$ is the stability weighting factor, adjusted based on the current system intent (e.g., Latency vs. Reliability).

**Behavioral Outcome:** The agent learns to maintain connection with a cell as long as performance remains within acceptable thresholds, switching only when significant gains or reliability concerns arise.

---

## 2. Neural Architecture (Actor-Critic)

The agent utilizes an Actor-Critic architecture with a shared feature extraction layer.

### A. Observation Space
*   **Dimensions**: 42-dimensional vector (for a 7-cell hexagonal topology).
*   **Key Inputs**: 
    *   **Per-Cell (35)**: Radio metrics (RSRP, SINR), Relative Distance, Relative Bearing, and Serving Status for each of the 7 cells.
    *   **Global (7)**: UE Velocity, Battery, Handover History (Time since HO, Recent HO count), and Intent Weights ($\alpha, \beta, \gamma$).
*   **Output**: Latent state representation.

### B. Policy Head (Actor)
*   **Input**: Latent state.
*   **Output**: Probability distribution over the action space (Target Cells).
*   **Role**: Discrete cell selection.

### C. Value Head (Critic)
*   **Input**: Latent state.
*   **Output**: Scalar value $V(s)$.
*   **Role**: Estimates the expected cumulative reward for state $s$.

---

## 3. Training Process

The agent is trained using a multi-phase curriculum to ensure robust convergence:

1.  **Phase 1: Exploration**
    *   Focus on discovering the observation-action space.
    *   High entropy bonus to prevent premature convergence.

2.  **Phase 2: Performance Optimization**
    *   Emphasis on Quality of Service (QoS).
    *   Refining cell selection for maximum throughput.

3.  **Phase 3: Stability Alignment**
    *   Increased handover penalty.
    *   Injection of measurement noise (shadowing) to enhance generalizability.

---

## 4. Deployment Model: Tactical Execution

In the final architecture, the HOAgentPPO is deployed as a pre-trained, frozen model within a Tactical Execution wrapper. This layer executes optimized actions while allowing higher-level orchestration layers to adjust environmental parameters (e.g., cell individual offsets or time-to-trigger).

### File Location
[`src/telellm/tactical_frozen.py`](file:///c:/Users/mu0rt/OneDrive/Desktop/graduation%20project/src/telellm/tactical_frozen.py)
