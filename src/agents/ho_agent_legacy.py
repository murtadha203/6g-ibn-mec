import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import math  # [Renaissance] For bearing calculation
from collections import deque
from typing import Dict, Any, Tuple, List

# Handle both package import and standalone execution
try:
    from .base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent


# Neural Network Architecture

class QNetwork(nn.Module):
    """
    Deep Q-Network for estimating Q-values for each cell action.
    
    Architecture:
        Input: [batch_size, obs_dim=18]
        FC1: [18 -> 128] + ReLU
        FC2: [128 -> 128] + ReLU
        FC3: [128 -> 64] + ReLU
        Output: [64 -> num_cells] (Q-values)
    """
    
    def __init__(self, obs_dim: int, num_actions: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# Experience Replay Buffer

class ReplayBuffer:
    """
    Stores and samples experience tuples for off-policy learning.
    """
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, obs: np.ndarray, action: int, reward: float, 
            next_obs: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.buffer.append((obs, action, reward, next_obs, done))
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# Handover Agent

class HOAgent(BaseAgent):
    """
    Handover Agent using Deep Q-Network (DQN) for cell selection.
    
    Responsibilities:
    - Observe radio conditions (RSRP, SINR, throughput) across all cells
    - Select target cell to maximize quality while minimizing handovers
    - Learn from experience using off-policy RL
    - Coordinate with ContextAgent for adaptive preferences
    """
    
    def __init__(
        self, 
        agent_id: str = "ho_agent_0",
        config: Dict[str, Any] = None,
        num_cells: int = 7,  # Default to 7 for Renaissance hex grid
    ):
        super().__init__(agent_id, config or {})
        
        self.num_cells = num_cells
        self.num_cells = num_cells
        # Observation Space:
        # Per-cell (5): RSRP, SINR, RelDist, Bearing, IsServing
        # Global (7): Speed, Batt, TimeSinceHO, HOCount, Alpha, Beta, Gamma
        self.obs_dim = 5 * num_cells + 7
        
        # Hyperparameters (can be overridden via config)
        self.lr = config.get("learning_rate", 0.001) if config else 0.001
        self.gamma = config.get("gamma", 0.99) if config else 0.99
        self.batch_size = config.get("batch_size", 64) if config else 64
        self.buffer_size = config.get("buffer_size", 10000) if config else 10000
        self.target_update_freq = config.get("target_update_freq", 500) if config else 500
        
        # Epsilon-greedy exploration
        self.epsilon = config.get("epsilon_start", 1.0) if config else 1.0
        self.epsilon_min = config.get("epsilon_min", 0.01) if config else 0.01
        self.epsilon_decay = config.get("epsilon_decay", 0.98) if config else 0.98
        
        # Handover control parameters
        self.handover_margin_db = 3.0
        self.time_to_trigger_s = 0.16
        
        # Q-Networks (online and target)
        self.q_network = QNetwork(self.obs_dim, self.num_cells)
        self.target_network = QNetwork(self.obs_dim, self.num_cells)
        self._update_target_network()  # Initialize target with same weights
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=self.buffer_size)
        
        # Training metrics
        self.update_counter = 0
        self.episode_counter = 0
        self.last_handover_time = 0.0
        self.loss_history = []
    
    def set_context_weights(self, weights: Dict[str, float]):
        """Update context preference weights."""
        self.context_weights = weights
        # [PHASE 17] Extract HOM/TTT parameters
        self.handover_margin_db = weights.get("handover_margin_db", 3.0)
        self.time_to_trigger_s = weights.get("time_to_trigger_s", 0.16)

    # Observation Processing
    
    def get_observation(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract scale-invariant features from simulation context.
        
        Feature Vector (Size = 5 * num_cells + 6):
        
        Per-Cell Features (for each cell i):
        1. RSRP (norm): (dbm + 100) / 50
        2. SINR (norm): (db + 10) / 30
        3. Rel Dist: distance_to_ue / current_isd (clipped to 3.0)
        4. Rel Bearing: angle_to_ue / pi [-1, 1]
        5. Is Serving: 1.0 if serving, else 0.0
        
        Global Features:
        1. UE Speed (norm): mps / 30
        2. UE Battery (norm): joules / 1000
        3. Time since HO (norm): sec / 5
        4. Recent HO Count (norm): count / 5
        5. Alpha (Latency weight)
        6. Beta (Energy weight)
        """
        # 1. Context Extraction
        rsrp_dbm = context["rsrp_dbm"]
        sinr_db = context["sinr_db"]
        ue_pos = context.get("ue_position", (0, 0))
        bs_positions = context.get("bs_positions", [])
        current_isd = context.get("current_isd", 500.0)  # Default if missing
        serving_id = context["serving_cell_id"]
        
        
        # For robustness, if missing, assume 3-cell line or generic
        if not bs_positions and self.num_cells == 3:
            # Legacy fallback
            bs_positions = [(0, 0), (500, 0), (1000, 0)] 
            
        features = []
        
        # 2. Per-Cell Features
        for i in range(self.num_cells):
            # Safe access with padding if mismatch
            r_val = rsrp_dbm[i] if i < len(rsrp_dbm) else -140.0
            s_val = sinr_db[i] if i < len(sinr_db) else -20.0
            
            # RSRP Norm
            rsrp_norm = (r_val + 100.0) / 50.0
            
            # SINR Norm
            sinr_norm = (s_val + 10.0) / 30.0
            
            
            if i < len(bs_positions):
                bx, by = bs_positions[i]
                dx = bx - ue_pos[0]
                dy = by - ue_pos[1]
                dist = math.hypot(dx, dy)
                angle = math.atan2(dy, dx)
                
                # Normalize distance by ISD (Scale Invariance!)
                # Clip to 3.0 (e.g. 3 ISDs away is "far")
                dist_norm = min(dist / max(current_isd, 1.0), 3.0) 
                
                # Normalize bearing to [-1, 1]
                bearing_norm = angle / math.pi
            else:
                dist_norm = 3.0
                bearing_norm = 0.0
            
            # Is Serving
            is_serving = 1.0 if i == serving_id else 0.0
            
            features.extend([rsrp_norm, sinr_norm, dist_norm, bearing_norm, is_serving])
            
        # 3. Global Features
        ue_speed = context.get("ue_speed_mps", 0.0)
        speed_norm = min(ue_speed / 30.0, 1.0)
        
        battery = context.get("ue_battery_joules", 1000.0)
        batt_norm = battery / 1000.0
        
        # HO History
        current_time = context.get("time_s", 0.0)
        ho_history = context.get("handover_history", [])
        
        if ho_history:
            t_since = min(current_time - ho_history[-1], 5.0)
        else:
            t_since = 5.0
        time_ho_norm = t_since / 5.0
        
        recent_count = sum(1 for t in ho_history if current_time - t <= 2.0)
        count_ho_norm = min(recent_count / 5.0, 1.0)
        
        # Context Weights (Intent Weights)
        intent = context.get("intent_weights")
        if intent:
            alpha = intent.get('latency', 0.33)
            beta = intent.get('energy', 0.33)
            gamma = intent.get('throughput', 0.34)
        elif getattr(self, "context_weights", None):
            alpha = self.context_weights.get("alpha", 0.5)
            beta = self.context_weights.get("beta", 0.5)
            gamma = self.context_weights.get("gamma", 0.0)
        else:
            user_pref = context.get("user_pref", {})
            alpha = user_pref.get("latency_weight", 0.5)
            beta = user_pref.get("energy_weight", 0.5)
            gamma = 1.0 - alpha - beta if (alpha + beta) < 1.0 else 0.0
            
        features.extend([speed_norm, batt_norm, time_ho_norm, count_ho_norm, alpha, beta, gamma])
        
        return np.array(features, dtype=np.float32)
    
    # Action Selection
    
    def select_action(self, observation: np.ndarray) -> int:
        """
        Select cell action using epsilon-greedy strategy.
        
        Training mode: Explore with probability epsilon
        Evaluation mode: Always exploit (greedy)
        """
        if self.training and random.random() < self.epsilon:
            # Explore: random cell selection
            return random.randint(0, self.num_cells - 1)
        else:
            # Exploit: select cell with highest Q-value
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                q_values = self.q_network(obs_tensor)
                return q_values.argmax().item()
    
    # Learning
    
    def update(
        self, 
        observation: np.ndarray, 
        action: int, 
        reward: float, 
        next_observation: np.ndarray, 
        done: bool
    ):
        """
        Add experience to replay buffer and perform DQN update.
        """
        # Store transition
        self.replay_buffer.add(observation, action, reward, next_observation, done)
        
        # Only train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample minibatch
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs_batch)
        action_tensor = torch.LongTensor(action_batch)
        reward_tensor = torch.FloatTensor(reward_batch)
        next_obs_tensor = torch.FloatTensor(next_obs_batch)
        done_tensor = torch.FloatTensor(done_batch)
        
        # Compute current Q-values
        current_q_values = self.q_network(obs_tensor)
        current_q = current_q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_tensor)
            max_next_q = next_q_values.max(1)[0]
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q
        
        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track metrics
        self.loss_history.append(loss.item())
        self.update_counter += 1
        
        # Periodically update target network
        if self.update_counter % self.target_update_freq == 0:
            self._update_target_network()
    
    def _update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    # Training Control
    
    def reset(self):
        """Reset episode-specific state."""
        super().reset()
        self.last_handover_time = 0.0
        
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_counter += 1
    
    def set_epsilon_schedule(self, epoch: int, total_epochs: int = 500) -> None:
        """
        Set epsilon based on training phase for Grand Slam overnight training.
        
        HO Agent uses more conservative schedule to preserve ping-pong stability:
        Phase 1 (Epochs 1-100): Exploration - Medium-high epsilon (1.0 -> 0.20)
        Phase 2 (Epochs 101-350): Learning Zone - Medium-low epsilon (0.20 -> 0.08)
        Phase 3 (Epochs 351-500): Fine Tuning - Low epsilon (0.08 -> 0.02)
        
        Args:
            epoch: Current training epoch (1-indexed)
            total_epochs: Total number of epochs (default 500)
        """
        if epoch <= 100:
            # Phase 1: Exploration - Decay from 1.0 to 0.20 (less aggressive than MEC)
            progress = epoch / 100.0
            self.epsilon = max(0.20, 1.0 - 0.80 * progress)
        elif epoch <= 350:
            # Phase 2: Learning Zone - Slow decay from 0.20 to 0.08
            progress = (epoch - 100) / 250.0
            self.epsilon = max(0.08, 0.20 - 0.12 * progress)
        else:
            # Phase 3: Fine Tuning - Slow decay from 0.08 to 0.02
            progress = (epoch - 350) / 150.0
            self.epsilon = max(0.02, 0.08 - 0.06 * progress)
    
    # Persistence
    
    def save(self, path: str):
        """Save agent state to disk."""
        save_dict = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode_counter": self.episode_counter,
            "update_counter": self.update_counter,
            "config": self.config,
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_dict, path)
        print(f"HOAgent saved to {path}")
    
    def load(self, path: str):
        """Load agent state from disk."""
        if not os.path.exists(path):
            print(f"HOAgent: No checkpoint found at {path}")
            return
        
        checkpoint = torch.load(path, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.episode_counter = checkpoint.get("episode_counter", 0)
        self.update_counter = checkpoint.get("update_counter", 0)
        
        print(f"HOAgent loaded from {path} (Episode {self.episode_counter})")
        
    # Utilities
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return training metrics for logging/visualization."""
        return {
            "epsilon": self.epsilon,
            "episode": self.episode_counter,
            "updates": self.update_counter,
            "avg_loss": np.mean(self.loss_history[-100:]) if self.loss_history else 0.0,
            "buffer_size": len(self.replay_buffer),
        }


