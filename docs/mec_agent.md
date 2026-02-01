# MEC Offloading Agent (PPO)

The **MEC Offloading Agent** is a specialized Proximal Policy Optimization (PPO) agent designed to manage computation offloading decisions in a Multi-access Edge Computing (MEC) environment.

---

## 1. Design Philosophy: Resource-Aware Offloading

The agent optimizes the trade-off between local processing and remote execution (Edge or Cloud) based on the current network state, server load, and terminal battery level.

### Objective Function
The agent seeks to maximize a multi-objective reward:
$$ R = w_{lat} \cdot R_{latency} + w_{eng} \cdot R_{energy} $$

Where:
*   $R_{latency}$: Encourages decisions that meet task deadlines.
*   $R_{energy}$: Penalizes high transmission power and local CPU consumption.
*   $w_{k}$: Dynamic weights provided by the high-level intent vector.

**Decision Logic:** The agent learns to offload computation to the MEC server when channel conditions are favorable and the server has available capacity, reverting to local processing during high congestion or extreme battery constraints.

---

## 2. Neural Architecture

Utilizes an Actor-Critic framework shared with the shared PPO core logic.

### A. Observation Space
*   **Dimensions**: 19-dimensional vector (Scale-invariant).
*   **Key Inputs**: 
    *   Radio metrics (RSRP, SINR, Throughput).
    *   MEC server load factors.
    *   Task characteristics (CPU cycles, data size, deadline).
    *   UE context (Speed, Battery, Time since last task).
    *   Current intent weights and last task result.

### B. Policy Network (Actor-Critic)
*   **Input Layer**: 19 units.
*   **Hidden Layers**: Two fully connected layers with 256 neurons each and Tanh activation functions.
*   **Output (Actor)**: 3 units (Action logits for Local, Edge, Cloud).
*   **Output (Critic)**: 1 unit (State value estimation).

---

## 3. Training and Curriculum

The agent is trained through a curriculum focused on intent adaptation:

1.  **Phase 1: Basic Connectivity**
    *   Learn the basic mechanics of transmission and local execution.
2.  **Phase 2: Intent Shifting**
    *   Training with varying $w_{lat}$ and $w_{eng}$ to induce polymorphic behavior.
3.  **Phase 3: Stress Testing**
    *   Introduction of high-load scenarios and failing backhaul links.

---

## 4. Deployment

The MEC agent is deployed as a pre-trained policy within the `MECAgentPPO` class. It receives context updates from the `NetworkSimulation` and produces offloading decisions for every task arrival.

### File Location
[`src/agents/mec_agent_ppo.py`](file:///c:/Users/mu0rt/OneDrive/Desktop/graduation%20project/src/agents/mec_agent_ppo.py)
