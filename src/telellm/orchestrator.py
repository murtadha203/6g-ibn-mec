
from typing import Dict, Any
from .schemas import ControlMode
from .tactical_frozen import TacticalExecutor
from .nodes.core_nodes import EstimatorNode, ShieldNode, ConfiguratorNode
from .nodes.strategist import StrategistNode

class TeleLLMOrchestrator:
    """
    The TeleLLM Orchestrator.
    
    Implemented through a Neuro-Symbolic Control Loop:
    1. Symbolic Estimator: Maps high-level metrics to state representation.
    2. Strategist: Determines optimal control policy based on system state.
    3. Safety Shield: Enforces safety constraints on proposed modes.
    4. Configurator: Applies selected mode parameters to the execution layer.
    5. Tactical Execution: Executes actions within the simulation environment.
    """
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        # Initialize Orchestrator components
        
        # 1. Initialize Tactical Layer (Tier 1)
        self.tactical = TacticalExecutor(checkpoint_path, device=device)
        
        # 2. Initialize Strategic Nodes (Tier 2)
        self.estimator_node = EstimatorNode()
        self.strategist_node = StrategistNode()
        self.shield_node = ShieldNode()
        self.configurator_node = ConfiguratorNode(self.tactical)
        
        # State Tracking
        self.state = {
            "step": 0,
            "current_mode": ControlMode.BALANCED.value,
            "history": []
        }
        
    def step(self, metrics: Dict[str, float], observation: Any) -> int:
        """
        Execute one control step.
        
        Args:
            metrics: Raw simulation metrics (RSRP, Power, Latency, etc.)
            observation: Agent's observation vector (raw).
            
        Returns:
            action: The selected cell index (Tier 1 output).
        """
        self.state["step"] += 1
        self.state["metrics"] = metrics
        
        # --- Perception (Symbolic Estimator) ---
        est_update = self.estimator_node.run(self.state)
        self.state.update(est_update)
        
        # --- Cognition (Strategist) ---
        strat_update = self.strategist_node.run(self.state)
        self.state.update(strat_update)
        
        # --- Safety (Shield) ---
        shield_update = self.shield_node.run(self.state)
        self.state.update(shield_update)
        
        # --- Configuration (Actuation) ---
        config_update = self.configurator_node.run(self.state)
        self.state.update(config_update)
        
        # Update current mode for next step
        self.state["current_mode"] = self.state["final_mode"]
        
        # --- Tactical Action ---
        # The TacticalExecutor is now configured. Get the action.
        action = self.tactical.act(observation)
        
        return action

    def get_debug_info(self) -> Dict[str, Any]:
        """Return explainability signals."""
        return {
            "mode": self.state.get("final_mode"),
            "symbolic_state": self.state.get("symbolic_state"),
            "reasoning": self.state.get("reasoning"),
            "shield_active": self.state.get("shield_active")
        }
