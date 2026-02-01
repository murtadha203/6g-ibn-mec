import numpy as np
from typing import Dict, Any, List

# ============================================================================
#  UTILITY FUNCTIONS
# ============================================================================

def normalize_reward(reward: float, min_r: float = -10.0, max_r: float = 20.0) -> float:
    """Clip reward to expected range for training stability."""
    return np.clip(reward, min_r, max_r)

def compute_shaped_reward(
    immediate_reward: float,
    potential_before: float,
    potential_after: float,
    discount: float = 0.99,
) -> float:
    """Potential-based reward shaping."""
    shaping = discount * potential_after - potential_before
    return immediate_reward + shaping

# ============================================================================
#  O-RAN INTENT MAPPING (PATH C: COMPLIANCE)
# ============================================================================

INTENT_MAP = {
    # 1. Performance Focus (eMBB)
    "MAXIMIZE_THROUGHPUT": {
        "alpha": 0.8, "beta": 0.1, "gamma": 0.1,
        "handover_margin_db": 1.0, "time_to_trigger_s": 0.04
    },
    
    # 2. Energy Efficiency (Green Mode)
    "MINIMIZE_ENERGY": {
        "alpha": 0.1, "beta": 0.8, "gamma": 0.1, 
        "handover_margin_db": 3.0, "time_to_trigger_s": 0.16
    },
    
    # 3. Reliability Focus (URLLC)
    "GUARANTEE_RELIABILITY": {
        "alpha": 0.2, "beta": 0.2, "gamma": 0.6,
        "handover_margin_db": 2.0, "time_to_trigger_s": 0.08
    },
    
    # 4. Stability (Anti-PingPong)
    "STABILIZE_CONNECTION": {
        "alpha": 0.33, "beta": 0.33, "gamma": 0.34,
        "handover_margin_db": 6.0, "time_to_trigger_s": 0.32
    }
}

def get_weights_from_intent(intent: str) -> Dict[str, Any]:
    """Retrieve partial weight config from intent string."""
    return INTENT_MAP.get(intent, INTENT_MAP["STABILIZE_CONNECTION"])


# ============================================================================
#  MEC OFFLOADING REWARD FUNCTION
# ============================================================================

def compute_mec_reward(
    task_info: Dict[str, Any],
    preference_weights: Dict[str, float],
    training_mode: bool = False,
) -> float:
    """
    Compute reward for a MEC offloading decision.
    
    Args:
        task_info: Task execution results (latency, deadline, energy, etc.)
        preference_weights: Context-based weights for latency/energy
        training_mode: If True, use aggressive rewards to encourage edge offloading
                      (for overnight training). If False, use stable production rewards.
    
    Target Range: [-10.0, +20.0]
    """
    # 1. Extract Metrics
    latency = task_info["latency_s"]
    deadline = task_info["deadline_s"]
    deadline_met = task_info["deadline_met"]
    energy_consumed = task_info["energy_j"]
    target = task_info.get("offload_target", "local")
    
    # 2. Extract Weights
    raw_alpha = preference_weights.get("latency_weight", 0.5)
    raw_beta = preference_weights.get("energy_weight", 0.5)
    
    
    gamma = 0.4 
    remaining = 1.0 - gamma
    total_raw = raw_alpha + raw_beta + 1e-9
    
    alpha = (raw_alpha / total_raw) * remaining
    beta = (raw_beta / total_raw) * remaining
    
    # 3. Latency Reward (with Speed Bonus)
    # Calculate time saved ratio for speed bonus
    time_saved_ratio = max(0.0, (deadline - latency) / (deadline + 1e-9))
    
    if training_mode:
        # Context-aware reward shaping based on network conditions
        cell_load = task_info.get("cell_congestion", 0.0)
        rsrp_dbm = task_info.get("rsrp_dbm", -80.0)
        
        if deadline_met:
            if target == "edge":
                # SMART AGGRESSION: Reward using edge when conditions are GOOD
                if cell_load < 0.7 and rsrp_dbm > -95:
                    # Optimal conditions: low load and strong signal
                    r_latency = 15.0 + 6.0 * time_saved_ratio  # Max +21
                elif cell_load >= 0.85:
                    # Penalty for offloading under high congestion
                    r_latency = -5.0
                elif rsrp_dbm < -105:
                    # Penalty for offloading at weak signal strength
                    r_latency = -10.0
                else:
                    # Moderate conditions: OK but not great
                    r_latency = 8.0 + 3.0 * time_saved_ratio
                    
            elif target == "cloud":
                # Cloud offloading: less preferred than edge
                r_latency = 4.0 + 2.0 * time_saved_ratio
                
            else:
                # LOCAL PROCESSING
                if cell_load > 0.8 or rsrp_dbm < -100:
                    # Reward conservative decision under poor network conditions
                    r_latency = 10.0
                elif cell_load < 0.5 and rsrp_dbm > -85:
                    # Opportunity cost: network was available but not utilized
                    r_latency = 1.0
                else:
                    # Reasonable decision in moderate conditions
                    r_latency = 5.0
        else:
            # FAILURE CASE
            violation_ratio = min(1.0, (latency - deadline) / (deadline + 1e-9))
            if target == "edge":
                if cell_load > 0.75 or rsrp_dbm < -100:
                    # High penalty for offloading to congested/weak network
                    r_latency = -10.0 - 5.0 * violation_ratio
                else:
                    # Softer penalty: reasonable attempt that failed
                    r_latency = -3.0 - 2.0 * violation_ratio
            else:
                # Local processing failed (device overload?)
                r_latency = -5.0 - 3.0 * violation_ratio
        
        r_latency = np.clip(r_latency, -15.0, 21.0)
        
    else:
        # Production mode: balanced rewards for evaluation and benchmarking
        
        if deadline_met:
            if target == "edge":
                latency_ratio = latency / (deadline + 1e-9)
                r_latency = 10.0 + 5.0 * (1.0 - latency_ratio)
            else:
                r_latency = 1.0
        else:
            violation = latency - deadline
            r_latency = -5.0 * (violation / (deadline + 1e-9))
        
        r_latency = np.clip(r_latency, -5.0, 15.0)
    
    # 4. Energy Reward
    reference_energy_j = 2.0  # Approx cost of local VR processing
    energy_ratio = energy_consumed / reference_energy_j
    
    r_energy = 3.0 * (1.0 - energy_ratio)
    r_energy = np.clip(r_energy, -3.0, 3.0)
    
    # 5. Reliability Reward
    if training_mode:
        # Training mode: Extra bonus for edge reliability
        if deadline_met and target == "edge":
            r_reliability = 6.0  # Increased from 5.0
        elif deadline_met and target == "cloud":
            r_reliability = 3.0
        elif deadline_met:
            r_reliability = 0.5  # Reduced from 1.0 for local
        else:
            r_reliability = -4.0  # Slightly softer penalty
    else:
        # Production mode: Standard reliability rewards
        if deadline_met and target == "edge":
            r_reliability = 5.0
        elif deadline_met:
            r_reliability = 1.0
        else:
            r_reliability = -5.0
    
    # 6. Combine
    reward = alpha * r_latency + beta * r_energy + gamma * r_reliability
    
    # Final clip to wider range
    return np.clip(reward, -10.0, 20.0)

# ============================================================================
#  HANDOVER (HO) REWARD FUNCTION
# ============================================================================

def compute_ho_reward(
    decision: Dict[str, Any],
    radio_state_before: Dict[str, Any],
    radio_state_after: Dict[str, Any],
    serving_cell_before: int,
    serving_cell_after: int,
    handover_history: List[float],
    preference_weights: Dict[str, float],
    time_s: float,
) -> float:
    """
    Compute reward for a handover decision.
    Target Range: [-5.0, +5.0]
    
    Weights Mapping:
    - Alpha (Performance) + Beta (Energy) -> Quality Reward (Signal, Throughput)
    - Gamma (Reliability) -> Stability Reward (Ping-Pong Penalty)
    """
    target_cell = decision.get("handover_target", serving_cell_before)
    
    # 1. Weights Extraction
    # The Context Node sends: alpha (Perf), beta (Energy), gamma (Reliability)
    # We map 'Reliability' (Gamma) directly to Stability/Ping-Pong avoidance.
    # We map 'Performance' (Alpha) and 'Energy' (Beta) to Signal Quality (better signal = less energy & more SPEED).
    
    raw_alpha = preference_weights.get("alpha", 0.33)
    raw_beta = preference_weights.get("beta", 0.33)
    raw_gamma = preference_weights.get("gamma", 0.34)
    
    # Normalize to ensure sum = 1.0 (Safety check)
    total_w = raw_alpha + raw_beta + raw_gamma + 1e-9
    w_quality = (raw_alpha + raw_beta) / total_w  # Quality matters for Perf AND Energy
    w_stability = raw_gamma / total_w             # Stability matters for Reliability
    
    # 2. Quality Metrics (RSRP, SINR, Throughput)
    # RSRP Diff
    rsrp_before = radio_state_before["rsrp_dbm"][serving_cell_before]
    rsrp_after = radio_state_after["rsrp_dbm"][serving_cell_after]
    rsrp_diff = rsrp_after - rsrp_before
    # FIX: Widen clip range to allow strong penalties for signal loss
    r_rsrp = np.clip(rsrp_diff / 10.0, -5.0, 1.0)
    
    # SINR Diff
    sinr_before = radio_state_before["sinr_db"][serving_cell_before]
    sinr_after = radio_state_after["sinr_db"][serving_cell_after]
    sinr_diff = sinr_after - sinr_before
    # FIX: Widen clip range to allow strong penalties for interference
    r_sinr = np.clip(sinr_diff / 5.0, -5.0, 1.0)
    
    # Throughput Diff (Absolute Norm)
    thr_before = radio_state_before["throughput_bps"][serving_cell_before]
    thr_after = radio_state_after["throughput_bps"][serving_cell_after]
    max_capacity = 1e9 # 1 Gbps
    thr_diff_norm = (thr_after - thr_before) / max_capacity
    r_thr = np.clip(thr_diff_norm * 5.0, -1.0, 1.0)
    
    # Combined Quality Score
    r_quality = 0.3 * r_rsrp + 0.3 * r_sinr + 0.4 * r_thr
    
    # Outage Penalty (CRITICAL: Make staying in bad signal very expensive)
    # If RSRP < -95 dBm, agent MUST consider switching
    if rsrp_after < -95.0:
        r_quality -= 10.0  # Strong penalty to force action
    
    # 3. Stability Metrics (Ping-Pong)
    did_handover = (serving_cell_after != serving_cell_before)
    r_stability = 0.0
    
    if did_handover:
        r_stability = -0.1  # Base cost of switching
        
        # Ping-Pong Detection (2s window)
        recent_hos = [t for t in handover_history if (time_s - t <= 2.0) and (t < time_s)]
        
        if len(recent_hos) > 0:
            # Penalize linearly with number of recent HOs
            
            r_stability -= 0.2 * len(recent_hos)
            
            # Additional penalty if Gamma is high (Strict Mode)
            if w_stability > 0.5:
                 r_stability -= 2.0 # Reduced from 5.0
    
    # 4. Final Combination
    # Quality is usually +0.5 to +2.0 without penalties
    # Stability is -0.1 to -10.0
    # If Gamma=0.9: Reward = 0.1 * Quality + 0.9 * Stability
    #              Reward = 0.1*(2.0) + 0.9*(-5.0) = 0.2 - 4.5 = -4.3 (Big Penalty)
    # If Gamma=0.1: Reward = 0.9*(2.0) + 0.1*(-5.0) = 1.8 - 0.5 = 1.3 (Net Positive)
    
    # Scale up components to observable range
    reward = w_quality * (r_quality * 10.0) + w_stability * (r_stability * 5.0)
    
    return normalize_reward(reward)

# ============================================================================

# ============================================================================

if __name__ == "__main__":
    print("=== MEC Reward Test ===")
    task_success = {
        "task_id": 1,
        "offload_target": "edge",
        "latency_s": 0.03,
        "deadline_s": 0.06,
        "deadline_met": True,
        "energy_j": 0.05,
    }
    prefs_vr = {"latency_weight": 0.8, "energy_weight": 0.2}
    
    r_mec = compute_mec_reward(task_success, prefs_vr)
    print(f"MEC Reward (VR Success): {r_mec:.3f}")
    
    print("\n=== HO Reward Test ===")
    radio_before = {
        "rsrp_dbm": [-95.0, -80.0, -100.0],
        "sinr_db": [-2.0, 15.0, -5.0],
        "throughput_bps": [50e6, 500e6, 10e6],
    }
    radio_after = {
        "rsrp_dbm": [-95.0, -80.0, -100.0],
        "sinr_db": [-2.0, 15.0, -5.0],
        "throughput_bps": [50e6, 500e6, 10e6],
    }
    
    decision = {"handover_target": 1}
    prefs_ho = {"alpha": 0.33, "beta": 0.33, "gamma": 0.34}
    
    r_ho = compute_ho_reward(
        decision, radio_before, radio_after,
        serving_cell_before=0, serving_cell_after=1,
        handover_history=[], preference_weights=prefs_ho, time_s=10.0
    )
    print(f"HO Reward (Major Improvement): {r_ho:.3f}")
