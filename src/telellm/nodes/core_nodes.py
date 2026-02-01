
from typing import Dict, Any
from ..symbolic_estimator import SymbolicEstimator
from ..safety_shield import SafetyShield
from ..tactical_frozen import TacticalExecutor
from ..schemas import ControlMode

class EstimatorNode:
    """
    Wraps SymbolicEstimator to translate Raw Metrics -> Symbolic State.
    """
    def __init__(self):
        self.estimator = SymbolicEstimator()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        metrics = state.get("metrics", {})
        sym_state = self.estimator.estimate(metrics)
        
        # Flatten enums to strings for easier LLM parsing/Logging if needed
        # But Strategist expects values (strings) or Enums? 
        # SymbolicEstimator returns strings (values) based on previous view.
        return {"symbolic_state": sym_state}

class ShieldNode:
    """
    Wraps SafetyShield to enforce transition constraints.
    """
    def __init__(self):
        self.shield = SafetyShield()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        proposed_mode = state.get("proposed_mode")
        current_mode = state.get("current_mode")
        # Ensure current_mode is Enum
        if isinstance(current_mode, str):
            current_mode = ControlMode(current_mode)
            
        step = state.get("step", 0) # Use step as epoch proxy
        
        # Update shield's internal tracking (it tracks current mode internally too)
        # But safely, we pass the proposed transition.
        # The SafetyShield class we viewed had `propose_transition(proposed_mode, current_epoch)`
        # It also tracked `self.current_mode`. We should sync them.
        self.shield.current_mode = current_mode
        
        final_mode = self.shield.propose_transition(proposed_mode, step)
        
        return {"final_mode": final_mode.value, "shield_active": (final_mode != proposed_mode)}

class ConfiguratorNode:
    """
    Applies the Final Mode to the TacticalExecutor (Tactical Agent).
    """
    def __init__(self, tactical_executor: TacticalExecutor):
        self.tactical = tactical_executor

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        mode_val = state.get("final_mode")
        mode = ControlMode(mode_val)
        
        # Apply configuration
        self.tactical.apply_mode(mode)
        
        return {"config_applied": True, "active_params": {
            "margin": self.tactical.agent.handover_margin_db,
            "ttt": self.tactical.agent.time_to_trigger_s
        }}
