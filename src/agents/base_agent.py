from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    Enforces a standard interface for:
    - Context Agent
    - Handover (HO) Agent
    - MEC Offloading Agent
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.step_counter = 0
        self.training = True  # Default to training mode (exploration enabled)
        
    @abstractmethod
    def get_observation(self, context: Dict[str, Any]) -> Any:
        """
        Extract relevant features from the global simulation context.
        Return type depends on agent:
        - RL agents: numpy array or torch tensor
        - Rule-based agents: dict or any structured format
        """
        pass

    @abstractmethod
    def select_action(self, observation: Any) -> Any:
        """
        Select an action based on the current observation.
        If self.training is True, should explore (e.g., Epsilon-Greedy).
        If self.training is False, should be deterministic.
        """
        pass

    def update(self, observation: Any, action: Any, reward: float, next_observation: Any, done: bool):
        """
        Update the agent's policy based on the transition (s, a, r, s').
        Override this for learning agents (RL).
        """
        pass

    def reset(self):
        """Reset agent internal state at the start of an episode."""
        self.step_counter = 0

    def set_mode(self, training: bool):
        """Switch between Training (Exploration) and Evaluation (Deterministic) modes."""
        self.training = training

    @abstractmethod
    def save(self, path: str):
        """Save the agent model/weights to a file."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load the agent model/weights from a file."""
        pass
