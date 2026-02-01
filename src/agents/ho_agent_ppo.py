from .ppo_core import BasePPOAgent
from .ho_agent_legacy import HOAgent
import numpy as np
from collections import deque

class HOAgentPPO(BasePPOAgent, HOAgent):
    """
    PPO implementation for the Handover (HO) Agent.
    Combines mobility-aware observation logic with a Proximal Policy Optimization core.
    Supports frame stacking for temporal feature perception (e.g., velocity).
    """
    def __init__(self, agent_id="ho_agent_ppo", config=None, num_cells=7, lr=3e-4, frame_stack=1, **kwargs):
        
        # Initialize Base params
        self.num_cells = num_cells
        # Observation dimensions (includes throughput, RSRP, and intent weights)
        self.raw_obs_dim = 5 * num_cells + 7 
        self.frame_stack = frame_stack
        self.obs_dim = self.raw_obs_dim * frame_stack
        
        # Stacking Buffer
        self.obs_queue = deque(maxlen=frame_stack)
        
        # Call BasePPOAgent init
        BasePPOAgent.__init__(self, obs_dim=self.obs_dim, action_dim=num_cells,
                              lr=lr, gamma=0.99, batch_size=64)
        
        # Call HOAgent init (mostly to set config and other state)
        # Note: We overwrite select_action and update, so DQN parts are ignored
        self.agent_id = agent_id
        self.config = config or {}
        
        # Initialize state tracking from ho_agent
        self.last_handover_time = 0.0
        self.handover_margin_db = 3.0
        self.time_to_trigger_s = 0.16
        
        # Metrics to mimic HOAgent interface for logging
        self.loss_history = []
        self.update_counter = 0
        self.episode_counter = 0

    def reset_stack(self):
        """Clear observation history (Call at start of episode)."""
        self.obs_queue.clear()
        
    def get_observation(self, context):
        """
        Process context into a stacked observation vector.
        """
        # Fetch single frame
        single_obs = HOAgent.get_observation(self, context)
        
        # 2. Stack Handling
        if self.frame_stack > 1:
            if len(self.obs_queue) == 0:
                # First step: Repeat frame
                for _ in range(self.frame_stack):
                    self.obs_queue.append(single_obs)
            else:
                self.obs_queue.append(single_obs)
                
            # Flatten to [18*4]
            return np.concatenate(self.obs_queue)
        else:
            return single_obs
       
    def select_action(self, observation: np.ndarray):
        """
        PPO Action Selection: Sample from policy.
        Returns: action (int)
        
        Note: PPO needs log_prob and value for update.
        In training loop, we must call select_action_with_info() instead.
        This method is kept for compatibility with evaluating code.
        """
        action, _, _ = BasePPOAgent.select_action(self, observation)
        return action

    def select_action_with_info(self, observation: np.ndarray):
        """
        Returns action, log_prob, value for training buffer.
        """
        return BasePPOAgent.select_action(self, observation)
        
    def get_metrics(self):
        return {
            "episode": self.episode_counter,
            "updates": self.update_counter,
            "buffer_size": len(self.obs_buffer)
        }

    def update(self, rollout_data, last_val=0.0):
        """
        PPO Update Step.
        Accepts aggregated rollout dictionary and runs optimization.
        Overwrites HOAgent.update (DQN).
        """
        self.obs_buffer = rollout_data['obs']
        self.action_buffer = rollout_data['act']
        self.logprob_buffer = rollout_data['logprob']
        self.reward_buffer = rollout_data['rew']
        self.value_buffer = rollout_data['val']
        self.done_buffer = rollout_data['done']
        
        metrics = self.finish_episode(last_val=last_val)
        self.update_counter += 1
        return metrics

    # Re-use save/load from BasePPOAgent (overwrites HOAgent's save/load)
    
    def load(self, path):
         BasePPOAgent.load(self, path)
         print(f"HOAgentPPO loaded from {path}")
         
    def save(self, path):
         BasePPOAgent.save(self, path)
         print(f"HOAgentPPO saved to {path}")
