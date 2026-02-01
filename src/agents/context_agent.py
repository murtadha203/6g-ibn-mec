import numpy as np
import json
import os
from typing import Dict, Any, List
from .base_agent import BaseAgent

class ContextAgent(BaseAgent):
    """
    Context Awareness Agent (Rule-Based).
    
    Role:
    - Analyzes global simulation state.
    - Detects critical conditions (High Mobility, Low Battery, Congestion).
    - Outputs preference weights (alpha, beta, gamma) for other agents.
    - Does NOT take direct network actions.
    """
    
    def __init__(self, agent_id: str = "context_0", config: Dict[str, Any] = None):
        super().__init__(agent_id, config or {})
        
        # Default weights if no specific context is detected
        # Balanced profile
        self.default_weights = {
            "alpha": 0.33, # Latency
            "beta": 0.33,  # Energy
            "gamma": 0.34  # Reliability
        }
        
        # Thresholds (Physical)
        self.battery_low_threshold = 200.0  # Joules (20% of 1000J)
        self.speed_high_threshold = 15.0    # m/s (~54 km/h)
        self.congestion_sinr_threshold = 5.0 # dB

    def get_observation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features relevant to context detection.
        """
        return {
            "battery": context.get("ue_battery_joules", 1000.0), 
            "speed": context.get("ue_speed_mps", 0.0),
            "sinr": context.get("serving_sinr_db", 20.0),
            "service_type": "VR" 
        }

    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine context flags and preference weights.
        Returns a dictionary, not a tensor, because this is a Coordinator agent.
        """
        battery = observation["battery"]
        speed = observation["speed"]
        sinr = observation["sinr"]
        
        # 1. Detect Flags
        flags = {
            "low_battery": battery < self.battery_low_threshold,
            "high_mobility": speed > self.speed_high_threshold,
            "congestion": sinr < self.congestion_sinr_threshold
        }
        
        # 2. Compute Weights (Rule-Based Logic)
        weights = self.default_weights.copy()
        
        # Priority Logic:
        # Safety (Mobility) > Survival (Battery) > Performance (Latency)
        
        if flags["high_mobility"] or flags["congestion"]:
            # CRITICAL: Connection is unstable. Prioritize Reliability (Gamma).
            weights["gamma"] = 0.8
            weights["alpha"] = 0.1
            weights["beta"] = 0.1
            
        elif flags["low_battery"]:
            # WARNING: Dying. Prioritize Energy (Beta).
            weights["beta"] = 0.8
            weights["alpha"] = 0.1
            weights["gamma"] = 0.1
            
        else:
            # NORMAL: Prioritize Application Performance (Alpha).
            weights["alpha"] = 0.6
            weights["beta"] = 0.2
            weights["gamma"] = 0.2
            
        # Normalize to ensure sum is 1.0
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total
            
        return {
            "weights": weights,
            "flags": flags,
            "policy_mode": self._determine_mode(flags)
        }

    def _determine_mode(self, flags: Dict[str, bool]) -> str:
        if flags["high_mobility"]: return "robust"
        if flags["low_battery"]: return "power_save"
        if flags["congestion"]: return "conservative"
        return "performance"

    # =========================================================================
    # IMPLEMENTING ABSTRACT METHODS (REQUIRED TO PREVENT CRASH)
    # =========================================================================

    def save(self, path: str):
        """
        Context Agent is rule-based, so we only save the config/thresholds.
        We do not have Neural Network weights to save.
        """
        data = {
            "config": self.config,
            "thresholds": {
                "battery": self.battery_low_threshold,
                "speed": self.speed_high_threshold
            }
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self, path: str):
        """Load configuration if available."""
        if not os.path.exists(path):
            print(f"ContextAgent: No file found at {path}, keeping defaults.")
            return
            
        with open(path, 'r') as f:
            data = json.load(f)
            # Update thresholds if found
            if "thresholds" in data:
                self.battery_low_threshold = data["thresholds"].get("battery", 200.0)
                self.speed_high_threshold = data["thresholds"].get("speed", 15.0)
