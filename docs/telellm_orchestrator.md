# TeleLLM Orchestrator: System Design

**Project:** Multi-Agent Intelligent Orchestration
**Module:** `src/telellm/orchestrator.py`

## 1. The Concept: "TeleLLM"

The **TeleLLM** architecture introduces a hierarchical control system where a Large Language Model (LLM) act as a high-level cognitive controller for low-level Deep Reinforcement Learning (DRL) agents.

- **Execution Layer (DRL Agents):** Reactive agents (millisecond scale) trained via PPO. They handle immediate handover and offloading decisions.
- **Control Layer (TeleLLM):** A cognitive controller (second/minute scale) that monitors system-wide performance and dynamically switches the system's **Control Mode**.

---

## 2. Architecture: LangGraph Workflow

The Orchestrator is implemented as a cyclic graph using **LangGraph**.

```mermaid
graph TD
    A[Monitor Node] -->|Metrics| B[Symbolic Estimator]
    B -->|Current State| C[Strategist (LLM)]
    C -->|Proposed Mode| D[Safety Shield]
    D -->|Verified Mode| E[Simulation Node]
    E -->|New Metrics| A
```

### Components

#### 1. Symbolic Estimator
*   **Role:** Information processing.
*   **Action:** Maps raw numerical metrics (Latency, Battery, Signal Quality) to semantic system states (e.g., `TrafficState.SURGE`, `EnergyState.CRITICAL`).
*   **Rationale:** Reasoning over symbolic states enhances the interpretability and reliability of the LLM's strategic decisions.

#### 2. Strategist Node
*   **Role:** Strategic decision-making.
*   **Input:** Semantic State + Historical Trends.
*   **Logic:** Employs Chain-of-Thought reasoning to select the optimal `ControlMode` based on current system conditions and high-level objectives.
*   **Output:** `ControlMode` proposal (BALANCED, GREEN, SURVIVAL).

#### 3. Safety Shield
*   **Role:** Constraint enforcement.
*   **Action:** Validates the proposed `ControlMode` against predefined safety constraints and physical system limits.
*   **Rules:**
    *   *Stability:* Prevents excessive mode oscillations.
    *   *Hard Constraints:* Enforces specialized modes during critical battery or signal events.
*   **Output:** Final, verified `ControlMode`.

---

## 3. Control Modes

The system operates in one of three distinct modes, effectively "changing the physics" of the environment for the DRL agent.

| Mode | Handover Margin (Hysteresis) | Time-To-Trigger (TTT) | Philosophy |
| :--- | :--- | :--- | :--- |
| **BALANCED** | **3 dB** | **0.16s** | Standard operation. Trade-off between signal quality and stability. |
| **GREEN** | **12 dB** | **1.0s** | "Lazy" switching. Stick to the current cell unless it is significantly worse. Saves energy. |
| **SURVIVAL** | **0 dB** | **0.0s** | "Paranoid" switching. Switch immediately if *any* neighbor is better. Maximizes reliability. |

---

## 4. Autonomous Validation Loop (`run_telellm.py`)

The system's autonomous governance is validated through specialized stress-test scenarios where the environment dynamically transitions between distinct seasonal difficulty levels.

### Validation Scenarios
1.  **URLLC Storm**: Injects a massive traffic spike (5x load) to test `SURVIVAL` mode transitions.
2.  **Energy Blackout**: Rapid battery depletion to validate `GREEN` mode power conservation.
3.  **Hardware Failure**: Disables primary base stations to test immediate reliability-focused re-optimization.

### Explainability Output
The orchestrator generates a `logs/telellm_reasoning.json` file, which records the internal reasoning of the strategists at every decision interval. This provides a transparent audit trail for verifying the neuro-symbolic logic.

---

### File Locations
- **Orchestrator Core**: [`src/telellm/orchestrator.py`](file:///c:/Users/mu0rt/OneDrive/Desktop/graduation%20project/src/telellm/orchestrator.py)
- **Validation Script**: [`src/telellm/run_telellm.py`](file:///c:/Users/mu0rt/OneDrive/Desktop/graduation%20project/src/telellm/run_telellm.py)
