import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque
from typing import Dict, Any, Tuple, List, Optional

# Handle both package import and standalone execution
try:
    from .base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent


# Neural Network Architecture

class QNetwork(nn.Module):
    """
    Deep Q-Network for estimating Q-values for each offloading target.
    
    Architecture:
        Input: [batch_size, obs_dim=22]
        FC1: [22 -> 128] + ReLU
        FC2: [128 -> 128] + ReLU
        FC3: [128 -> 64] + ReLU
        Output: [64 -> 3] (Q-values: local, edge, cloud)
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


# MEC Offloading Agent

class MECAgent(BaseAgent):
    """
    MEC Offloading Agent using Deep Q-Network (DQN) for task offloading decisions.
    
    Responsibilities:
    - Observe task characteristics (size, deadline, service type)
    - Observe radio conditions and UE context
    - Select offloading target to minimize latency and energy while meeting deadlines
    - Learn from experience using off-policy RL
    - Coordinate with ContextAgent for adaptive preferences
    
    Key Difference from HO Agent:
    - Only makes decisions when tasks arrive (sparse updates)
    - Task-aware observation space
    - Energy-latency tradeoff optimization
    """
    
    def __init__(
        self, 
        agent_id: str = "mec_agent_0",
        config: Dict[str, Any] = None,
        num_actions: int = 3,  # local, edge, cloud
    ):
        super().__init__(agent_id, config or {})
        
        self.num_actions = num_actions
        # Observation dimensions (scaled-invariant)
        self.obs_dim = 19
        
        # Action mapping
        self.action_map = {0: "local", 1: "edge", 2: "cloud"}
        self.action_reverse_map = {"local": 0, "edge": 1, "cloud": 2}
        
        # Hyperparameters (same as HO Agent for consistency)
        self.lr = config.get("learning_rate", 0.001) if config else 0.001
        self.gamma = config.get("gamma", 0.99) if config else 0.99
        self.batch_size = config.get("batch_size", 64) if config else 64
        self.buffer_size = config.get("buffer_size", 10000) if config else 10000
        self.target_update_freq = config.get("target_update_freq", 500) if config else 500
        
        # Epsilon-greedy exploration
        self.epsilon = config.get("epsilon_start", 1.0) if config else 1.0
        self.epsilon_min = config.get("epsilon_min", 0.01) if config else 0.01
        self.epsilon_decay = config.get("epsilon_decay", 0.98) if config else 0.98  # Faster decay
        
        # Q-Networks (online and target)
        self.q_network = QNetwork(self.obs_dim, self.num_actions)
        self.target_network = QNetwork(self.obs_dim, self.num_actions)
        self._update_target_network()  # Initialize target with same weights
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=self.buffer_size)
        
        # Training metrics
        self.update_counter = 0
        self.episode_counter = 0
        self.task_counter = 0
        self.loss_history = []
        
        # State tracking for observation
        self.last_offload_target = "local"
        self.last_task_success = True
        self.last_task_time = 0.0
    
    def set_context_weights(self, weights: Dict[str, float]):
        """Update context preference weights."""
        self.context_weights = weights

    # Observation Processing
    
    def get_observation(self, context: Dict[str, Any], task: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Extract and normalize features from simulation context and task.
        
        Observation Vector (19 features):
        - Task data size (1): Normalized by 100 Mbits
        - Task CPU cycles (1): Normalized by 5 Gigacycles
        - Task deadline (1): Normalized by 1 second
        - Service type (3): One-hot [VR, EV, IoT]
        - Serving throughput (1): Normalized by 1 Gbps
        - Serving RSRP (1): Normalized to [-1, 1]
        - Serving SINR (1): Normalized to [-1, 1]
        - Battery level (1): Normalized by 1000 J
        - UE speed (1): Normalized by 30 m/s
        - Time since last task (1): Normalized by 10 seconds
        - Latency weight (1): Alpha from ContextAgent
        - Energy weight (1): Beta from ContextAgent
        - Reliability weight (1): Gamma from ContextAgent
        # - Serving cell (removed for scale invariance)
        - Last offload target (3): One-hot [local, edge, cloud]
        - Last task success (1): Binary {0, 1}
        """
        features = []
        
        # 1. Task characteristics (6 features)
        if task is not None:
            # Actual task data
            task_data_bits = task.get("data_size_bits", 0.0)
            task_cpu_cycles = task.get("cpu_cycles", 0.0)
            task_deadline_s = task.get("deadline_s", 0.0) - context.get("time_s", 0.0)
            service_type = task.get("service_type", "VR")
            
            # Normalize
            task_data_norm = task_data_bits / 100e6  # / 100 Mbits
            task_cpu_norm = task_cpu_cycles / 5e9    # / 5 Gigacycles
            task_deadline_norm = min(task_deadline_s / 1.0, 2.0)  # / 1 second, clip at 2
            
            # Service type one-hot
            service_onehot = [
                1.0 if service_type == "VR" else 0.0,
                1.0 if service_type == "EV" else 0.0,
                1.0 if service_type == "IoT" else 0.0
            ]
        else:
            # No task: use default/zero values
            task_data_norm = 0.0
            task_cpu_norm = 0.0
            task_deadline_norm = 1.0
            service_onehot = [0.0, 0.0, 0.0]
        
        features.extend([task_data_norm, task_cpu_norm, task_deadline_norm])
        features.extend(service_onehot)
        
        # 2. Radio conditions (3 features)
        serving_throughput_bps = context.get("serving_throughput_bps", 0.0)
        serving_rsrp_dbm = context.get("serving_rsrp_dbm", -100.0)
        serving_sinr_db = context.get("serving_sinr_db", 0.0)
        
        thr_norm = serving_throughput_bps / 1e9  # / 1 Gbps
        rsrp_norm = (serving_rsrp_dbm + 100.0) / 50.0  # [-150, -50] -> [-1, 1]
        sinr_norm = (serving_sinr_db + 10.0) / 30.0    # [-10, 20] -> [-0.33, 1]
        
        features.extend([thr_norm, rsrp_norm, sinr_norm])
        
        # 3. UE context (3 features)
        battery = context.get("ue_battery_joules", 1000.0)
        ue_speed = context.get("ue_speed_mps", 0.0)
        current_time = context.get("time_s", 0.0)
        
        battery_norm = battery / 1000.0
        speed_norm = min(ue_speed / 30.0, 1.0)
        time_since_last_task = min(current_time - self.last_task_time, 10.0)
        time_since_task_norm = time_since_last_task / 10.0
        
        features.extend([battery_norm, speed_norm, time_since_task_norm])
        
        # 4. Intent Weights (3 features)
        # [PHASE 4] Explicit IBN Support: w_lat, w_eng, w_thr
        intent = context.get("intent_weights")
        if intent:
            # Check keys
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
        
        features.extend([alpha, beta, gamma])
        
        # 5. Serving cell one-hot (Removed for Renaissance Scale Invariance)
        # serving_cell_id = context.get("serving_cell_id", 0)
        # features.extend(...) 
        # We rely on radio metrics (RSRP/Throughput) rather than Cell ID.
        
        # 6. Last decision context (4 features)
        last_target_onehot = [
            1.0 if self.last_offload_target == "local" else 0.0,
            1.0 if self.last_offload_target == "edge" else 0.0,
            1.0 if self.last_offload_target == "cloud" else 0.0
        ]
        last_success = 1.0 if self.last_task_success else 0.0
        
        features.extend(last_target_onehot)
        features.append(last_success)
        
        # Convert to numpy array
        observation = np.array(features, dtype=np.float32)
        
        # Sanity check
        assert len(observation) == self.obs_dim, f"Observation size mismatch: {len(observation)} != {self.obs_dim}"
        
        return observation
    
    # Action Selection
    
    def select_action(self, observation: np.ndarray, context: Dict[str, Any] = None) -> int:
        """
        Select offloading target using epsilon-greedy strategy.
        
        Args:
            observation: Feature vector for the neural network
            context: Optional context dict for physics guardrails (RSRP, etc.)
        
        Returns:
            action (int): 0=local, 1=edge, 2=cloud
        """
        # PHASE 19: PHYSICS GUARDRAIL
        # Force local processing if signal is extremely weak
        # Prevents agent from learning impossible "offload at -115 dBm" behaviors
        if context is not None:
            rsrp_dbm = context.get("serving_rsrp_dbm", -80.0)
            if rsrp_dbm < -110.0:
                # Signal too weak for reliable offloading
                # Force local processing (action = 0)
                return 0
        
        if self.training and random.random() < self.epsilon:
            # Explore: random target selection
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: select target with highest Q-value
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                q_values = self.q_network(obs_tensor)
                return q_values.argmax().item()
    
    def get_action_name(self, action: int) -> str:
        """Convert action index to string name."""
        return self.action_map.get(action, "local")
    
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
        
        NOTE: This should only be called when a task was processed.
        """
        # Store transition
        self.replay_buffer.add(observation, action, reward, next_observation, done)
        self.task_counter += 1
        
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
    
    # ========================================================================
    #  STATE TRACKING
    # ========================================================================
    
    def record_task_result(self, offload_target: str, success: bool, time: float):
        """Update internal state after task completion."""
        self.last_offload_target = offload_target
        self.last_task_success = success
        self.last_task_time = time
    
    # Training Control
    
    def reset(self):
        """Reset episode-specific state."""
        super().reset()
        self.last_offload_target = "local"
        self.last_task_success = True
        self.last_task_time = 0.0
        
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_counter += 1
    
    def set_epsilon_schedule(self, epoch: int, total_epochs: int = 500) -> None:
        """
        Set epsilon based on training phase for Grand Slam overnight training.
        
        Phase 1 (Epochs 1-100): Exploration - High epsilon (1.0 -> 0.30)
        Phase 2 (Epochs 101-350): Learning Zone - Medium epsilon (0.30 -> 0.10)
        Phase 3 (Epochs 351-500): Fine Tuning - Low epsilon (0.10 -> 0.02)
        
        Args:
            epoch: Current training epoch (1-indexed)
            total_epochs: Total number of epochs (default 500)
        """
        if epoch <= 100:
            # Phase 1: Exploration
            progress = epoch / 100.0
            self.epsilon = max(0.30, 1.0 - 0.70 * progress)
        elif epoch <= 350:
            # Phase 2: Learning Zone
            progress = (epoch - 100) / 250.0
            self.epsilon = max(0.10, 0.30 - 0.20 * progress)
        else:
            # Phase 3: Fine Tuning
            progress = (epoch - 350) / 150.0
            self.epsilon = max(0.02, 0.10 - 0.08 * progress)
    
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
            "task_counter": self.task_counter,
            "config": self.config,
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_dict, path)
        print(f"MECAgent saved to {path}")
    
    def load(self, path: str):
        """Load agent state from disk."""
        if not os.path.exists(path):
            print(f"MECAgent: No checkpoint found at {path}")
            return
        
        checkpoint = torch.load(path, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.episode_counter = checkpoint.get("episode_counter", 0)
        self.update_counter = checkpoint.get("update_counter", 0)
        self.task_counter = checkpoint.get("task_counter", 0)
        
        print(f"MECAgent loaded from {path} (Episode {self.episode_counter}, Tasks {self.task_counter})")
        
    # Utilities
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return training metrics for logging/visualization."""
        return {
            "epsilon": self.epsilon,
            "episode": self.episode_counter,
            "updates": self.update_counter,
            "tasks_processed": self.task_counter,
            "avg_loss": np.mean(self.loss_history[-100:]) if self.loss_history else 0.0,
            "buffer_size": len(self.replay_buffer),
        }


# ============================================================================

# ============================================================================
