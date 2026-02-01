The reward function is the foundational component that shapes the agent's behavior. The project utilizes a multi-objective reward structure to balance Quality of Service (QoS), energy efficiency, and network stability.

---

## 1. Multi-Objective Reward Equation

$$ R_{total} = w_{lat} \cdot R_{latency} + w_{eng} \cdot R_{energy} + w_{stab} \cdot R_{stability} + w_{rel} \cdot R_{reliability} $$

Where the weights ($w_{k}$) are dynamically adjusted based on the system intent (Intent-Based Networking) or fixed during specific training phases.

### A. Latency Reward ($R_{latency}$)
Encourages low-latency connections (High Throughput / Low Delay).
$$ R_{latency} = \frac{Throughput_{current}}{Throughput_{max}} $$

### B. Energy Reward ($R_{energy}$)
Penalizes high power consumption.
$$ R_{energy} = 1.0 - \frac{Power_{current}}{Power_{max}} $$

### C. Stability Reward ($R_{stability}$)
Penalizes high-frequency handover events to ensure network stability.
$$ R_{stability} = \begin{cases} -10.0 & \text{if Handover occurred} \\ +0.1 & \text{if Serving context maintained} \end{cases} $$
*Note: This penalty prevents "ping-pong" handovers, encouraging the agent to tolerate transient signal fluctuations in favor of connection stability.*

### D. Reliability Reward ($R_{reliability}$)
Penalizes Radio Link Failures (RLF) to maintain continuous connectivity.
$$ R_{reliability} = \begin{cases} -100.0 & \text{if RLF occurred} \\ +1.0 & \text{if Connectivity maintained} \end{cases} $$

---

## 3. Project Curriculum

To ensure robust policy convergence, a multi-phase curriculum learning strategy is employed:

| Phase | Designation | Primary Focus | Objective |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Exploration | State-space discovery | Map environmental dynamics and action outcomes. |
| **Phase 2** | Optimization | Performance maximization | Maximize $R_{latency}$ and $R_{energy}$ efficiency. |
| **Phase 3** | Stabilization | Volatility suppression | Enforce $R_{stability}$ constraints and noise robustness. |

---

## 4. Intent-Based Weighting

The agents are optimized using the **Proximal Policy Optimization (PPO)** objective, which improves training stability by clipping policy updates:

$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) ] $$

Where:
*   $r_t(\theta)$: Probability ratio between new and old policies.
*   $\hat{A}_t$: Advantage estimate computed via **Generalized Advantage Estimation (GAE)**.
*   $\epsilon$: Clipping coefficient (set to 0.2).

---

## 4. Intent-Based Weighting
During system operation, the high-level intent vector dynamically modulates the reward components:
*   **Latency-Critical**: $w_{lat} \uparrow$ (Focus on throughput).
*   **Energy-Saving**: $w_{eng} \uparrow$ (Focus on battery conservation).
*   **High-Reliability**: $w_{rel} \uparrow$ (Minimize failure probability).
