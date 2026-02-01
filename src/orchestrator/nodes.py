from .state import SystemState
import json
import random

def context_node(state: SystemState) -> SystemState:
    """
    Mock Tele-LLM: Analyzes simulation metrics and makes structured decisions.
    
    This node acts as the "Brain" of the system.
    Input: Metrics Dictionary (from Simulation)
    Output: Structured Decision (Action, Reasoning, New Weights)
    """
    metrics = state.get("logs", {})
    
    if not isinstance(metrics, dict):
        print(f"[Context] Warning: Metrics not a dict: {metrics[:50]}...")
        metrics = {"deadline_rate": 50.0, "avg_rsrp": -85.0, "handovers": 50, "ping_pongs": 0}
        
    current = state["context_weights"]
    
    # Extract Metrics
    deadline_rate = metrics.get("deadline_rate", 0.0)
    avg_rsrp = metrics.get("avg_rsrp", -100.0)
    ho_count = metrics.get("handovers", 0)
    ping_pongs = metrics.get("ping_pongs", 0)
    
    print(f"\n[Context] Analyzing Metrics: DL={deadline_rate:.1f}%, RSRP={avg_rsrp:.1f}, HO={ho_count}, PP={ping_pongs}")
    
    # --------------------------------------------------------------------------
    #  LLM REASONING CORE (Structured Logic)
    # --------------------------------------------------------------------------
    # In a real Tele-LLM, this section would be an API call to GPT-4o/Claude 3.5
    # constructing a prompt with the metrics and asking for JSON output.
    # Here we mock the reasoning logic.
    
    # --------------------------------------------------------------------------
    #  PATH C: INTENT-BASED OUTPUT (O-RAN Compliance)
    # --------------------------------------------------------------------------
    # Instead of outputting raw weights, we output High-Level Intents.
    # The 'rewards.py' module maps these to weights.
    
    from src.rewards import get_weights_from_intent
    
    intent_str = "MAINTAIN" # Default
    
    # Logic: Analyze gaps
    if deadline_rate < 80.0:
        intent_str = "GUARANTEE_RELIABILITY"
        decision["reasoning"] = f"CRITICAL: Reliability {deadline_rate:.1f}% < 80%. Enforcing URLLC constraints."
    elif ho_count > 150 or ping_pongs > 20: 
        intent_str = "STABILIZE_CONNECTION"
        decision["reasoning"] = f"Unstable mobility (HO={ho_count}). Enforcing Hysteresis."
    elif deadline_rate > 98.0 and (metrics.get('total_energy', 0) > 15.0):
        intent_str = "MINIMIZE_ENERGY" # Green Mode
        decision["reasoning"] = "Excellent QoS. Activating Green Mode to save energy."
    elif deadline_rate < 90.0:
        intent_str = "MAXIMIZE_THROUGHPUT" # Standard boosting
        decision["reasoning"] = "Performance dip detected. Boosting throughput priority."
    else:
        # Default behavior: If we woke up but things look OK, maybe just slight tune?
        # Or remain in current mode?
        # Let's default to Stabilize if unsure, or keep current.
        intent_str = "STABILIZE_CONNECTION"
        decision["reasoning"] = "Drift detected but ambiguous. Defaulting to Stability."

    decision["action"] = intent_str
    
    # Retrieve Weights from Intent Map
    new_weights = get_weights_from_intent(intent_str)
    decision["updates"] = new_weights
    
    # Apply Updates
    print(f"[Context] O-RAN Intent Issued: {intent_str}")
    print(f"[Context] Reasoning: {decision['reasoning']}")
    
    for k, v in decision["updates"].items():
        state["context_weights"][k] = v
        # print(f"   -> Set {k} = {v}") # Verbose


    # Save history
    if "history" not in state:
        state["history"] = []
    
    state["history"].append({
        "epoch": state.get("epoch", 0),
        "metrics": metrics,
        "decision": decision,
        "weights_snapshot": state["context_weights"].copy()
    })
    
    return state


def simulation_node(state: SystemState) -> SystemState:
    """
    Runs a batch of simulation steps using the Benchmark Suite.
    Acts as the interface between LangGraph and the Physics Engine.
    """
    # Import from the refactored benchmark suite (Orchestrator API)
    from experiments.benchmark_suite import run_simulation_batch
    
    config = {
        "context_weights": state["context_weights"],
        "epoch": state["epoch"]
    }
    steps = state["total_steps"]
    
    print(f"\n[Simulation] Starting Epoch {state['epoch'] + 1}...")
    
    # Run the batch
    # This calls the actual physics engine with the trained DQN agents
    # Returns Dict[str, Any]
    metrics = run_simulation_batch(config, steps_per_epoch=steps)
    
    # Update state
    state["logs"] = metrics
    state["epoch"] += 1
    
    if state["epoch"] >= state["max_epochs"]:
        state["is_done"] = True
    
    dl = metrics.get('deadline_rate', 0)
    en = metrics.get('total_energy', 0)
    ho = metrics.get('handovers', 0)
    
    # --------------------------------------------------------------------------
    #  PATH C: EVENT-DRIVEN DRIFT DETECTION (System 2 Trigger)
    # --------------------------------------------------------------------------
    # Check for anomalies or performance degradation
    needs_intervention = False
    intervention_reason = ""
    
    # Condition 1: Performance Collapse (Deadline < 80%)
    if dl < 80.0:
        needs_intervention = True
        intervention_reason += f"Low Reliability ({dl:.1f}%). "
        
    # Condition 2: Instability (High Handovers)
    if ho > 150: # Arbitrary threshold for per-epoch HOs
        needs_intervention = True
        intervention_reason += f"Instability ({ho} HOs). "
        
    # Condition 3: Energy Waste (High Reliability but Energy Spike) -> Opportunity
    # Only wake up if we can optimize significantly
    if dl > 98.0 and en > 25.0:
        needs_intervention = True
        intervention_reason += f"Energy Inefficiency ({en:.1f}J).Opt Opportunity. "
        
    # Apply Flag
    state["needs_intervention"] = needs_intervention
        
    if needs_intervention:
        print(f"[Simulation] DRIFT DETECTED: {intervention_reason} -> WAKING UP ORCHESTRATOR")
    else:
        print(f"[Simulation] System Stable. Skipping Orchestrator (System 1 Autopilot).")
        
    print(f"[Simulation] Epoch Complete. Result: Deadline={dl:.1f}%, Energy={en:.1f}J, HO={ho}")
    return state
