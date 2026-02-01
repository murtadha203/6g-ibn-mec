import time
from .schemas import ControlMode

class SafetyShield:
    """
    Symbolic Safety Layer.
    Enforces rigorous control constraints:
    1. Transition Topology (DAG validity)
    2. Dwell Time (Hysteresis)
    """
    def __init__(self, min_dwell_epochs=3):
        self.current_mode = ControlMode.BALANCED
        self.last_switch_epoch = 0
        self.min_dwell_epochs = min_dwell_epochs
        
        # Transition Graph (Adjacency List)
        # Key: Current Mode -> Value: Set of Allowed Next Modes
        self.transition_graph = {
            ControlMode.BALANCED: {ControlMode.SURVIVAL, ControlMode.GREEN},
            ControlMode.GREEN: {ControlMode.BALANCED}, # Must go via Balanced to Survival
            ControlMode.SURVIVAL: {ControlMode.BALANCED} # Gradual recovery
        }

    def propose_transition(self, proposed_mode: ControlMode, current_epoch: int) -> ControlMode:
        """
        Evaluate a proposed mode switch against safety rules.
        Returns: The SAFE mode (either new or old).
        """
        # Rule 1: Identity Matrix (Same mode is always safe)
        if proposed_mode == self.current_mode:
            return self.current_mode
            
        # Rule 2: Dwell Time
        if (current_epoch - self.last_switch_epoch) < self.min_dwell_epochs:
            print(f"[SHIELD] Blocked switch {self.current_mode}->{proposed_mode}. Dwell time active.")
            return self.current_mode
            
        # Rule 3: Topology Check
        if proposed_mode not in self.transition_graph[self.current_mode]:
            print(f"[SHIELD] Blocked Dangerous Transition {self.current_mode}->{proposed_mode}. Redirecting to BALANCED.")
            
            # For strict safety, we just REJECT. But we could propose intermediate.
            # Here we reject.
            return self.current_mode
            
        # If passed
        print(f"[SHIELD] Accepted Transition {self.current_mode}->{proposed_mode}.")
        self.current_mode = proposed_mode
        self.last_switch_epoch = current_epoch
        return proposed_mode
