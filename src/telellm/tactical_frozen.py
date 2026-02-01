
import torch
import numpy as np
from typing import Dict, Any, Optional
from ..agents.ho_agent_ppo import HOAgentPPO
from .schemas import ControlMode

class TacticalExecutor:
    """
    Tier 1: Tactical Execution Layer.
    
    Wraps the pre-trained PPO Agent in a frozen inference shell.
    Enables dynamic reconfiguration of control parameters (Hysteresis, TTT)
    by higher-level orchestrators based on the active ControlMode.
    """
    
    # Configuration profiles for different operational modes
    MODE_CONFIGS = {
        ControlMode.BALANCED: {
            "handover_margin_db": 3.0,
            "time_to_trigger_s": 0.16,
        },
        ControlMode.SURVIVAL: {
            "handover_margin_db": 0.0,
            "time_to_trigger_s": 0.0,
        },
        ControlMode.GREEN: {
            "handover_margin_db": 12.0,
            "time_to_trigger_s": 1.0,
        }
    }

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """
        Initialize the Frozen PPO Agent.
        
        Args:
            checkpoint_path: Path to the .pth weight file.
            device: 'cpu' or 'cuda'.
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Initialize Agent (Architecture must match training)
        self.agent = HOAgentPPO(
            agent_id="tactical_executor",
            num_cells=7,      # Renaissance Hex Topology
            frame_stack=1,    
            lr=0.0            # Learning rate 0 (Frozen)
        )
        
        # Load Weights
        self._load_weights()
        
        # Set to Eval Mode
        self.agent.network.eval()
        
        # Current State
        self.current_mode = ControlMode.BALANCED
        self.apply_mode(self.current_mode)
        
        self.apply_mode(self.current_mode)
        
        print(f"[TacticalExecutor] Online. Mode: {self.current_mode.value}. Status: Frozen.")

    def _load_weights(self):
        """Safely load the pre-trained weights."""
        try:
            # map_location ensures we can load CUDA models on CPU if needed
            self.agent.load(self.checkpoint_path)
            self.agent.load(self.checkpoint_path)
            print(f"[TacticalExecutor] Weights loaded from {self.checkpoint_path}")
        except Exception as e:
            print(f"[TacticalExecutor] CRITICAL: Failed to load weights! {e}")
            raise e

    def apply_mode(self, mode: ControlMode, overrides: Optional[Dict[str, float]] = None):
        """
        Reconfigure the agent's parameters based on the Strategic Mode.
        
        Args:
            mode: The ControlMode (BALANCED, SURVIVAL, GREEN).
            overrides: Optional dict to fine-tune specific params (e.g., from LLM).
        """
        self.current_mode = mode
        
        # Start with base profile for the mode
        config = self.MODE_CONFIGS[mode].copy()
        
        # Apply intelligent overrides
        if overrides:
            config.update(overrides)
            
        # Apply to Agent
        self.agent.handover_margin_db = config.get("handover_margin_db", 3.0)
        self.agent.time_to_trigger_s = config.get("time_to_trigger_s", 0.16)
        
        # Log the adaptation
        # Parameters updated successfully

    def act(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Execute Tactical Action.
        
        Args:
            observation: Raw observation from environment.
            deterministic: If True, use argmax (Greedy). If False, sample (Stochastic).
            
        Returns:
            action: The selected cell index.
        """
        # Observation processing for inference.
        # Deterministic behavior is typically used for final validation.
        
        if deterministic:
            # Manual Forward Pass for Greedy Action
            with torch.no_grad():
                # Normalize if agent was trained with normalization
                # The BasePPOAgent handles this internally in select_action, 
                # but we need to replicate or access it.
                # Accessing self.agent.obs_rms directly:
                if self.agent.normalize_obs:
                     # Note: We DON'T update RMS during inference, just normalize
                     norm_obs = self.agent.obs_rms.normalize(observation)
                else:
                     norm_obs = observation
                     
                obs_t = torch.FloatTensor(norm_obs).unsqueeze(0).to(self.device)
                logits = self.agent.network.actor(obs_t)
                action = torch.argmax(logits, dim=1).item()
                return action
        else:
            return self.agent.select_action(observation)
