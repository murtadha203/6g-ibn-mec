import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from typing import List, Tuple, Dict, Any

class RunningMeanStd:
    """Tracks running mean and count for normalization."""
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 1 else 1
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count
        
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

class ActorCritic(nn.Module):
    """
    Unified Actor-Critic architecture with three-layer MLP backends.
    Enhanced network capacity for robust feature representation.
    """
    def __init__(self, obs_dim: int, action_dim: int):
        super(ActorCritic, self).__init__()
        
        # Three-layer MLP for value estimation
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self):
        raise NotImplementedError
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_value(self, x):
        return self.critic(x)


class BasePPOAgent:
    """
    Implementation of the Proximal Policy Optimization (PPO) algorithm.
    Includes support for Generalized Advantage Estimation (GAE) and observation normalization.
    """
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99, 
                 gae_lambda: float = 0.95, clip_coef: float = 0.2, ent_coef: float = 0.05,
                 vf_coef: float = 0.5, batch_size: int = 2048, n_epochs: int = 10,
                 normalize_obs: bool = True):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef # Higher entropy for exploration
        self.vf_coef = vf_coef
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        self.device = torch.device("cpu") # Can toggle to cuda
        
        self.network = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Observation Normalization
        self.normalize_obs = normalize_obs
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        
        # Storage
        self.obs_buffer = []
        self.action_buffer = []
        self.logprob_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.done_buffer = []
        
        # Episode Counter
        self.episode_counter = 0
        
    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Select action for rollout.
        Returns: action, log_prob, value
        """
        # 1. Update RMS (Only during training rollouts?) 
        # Usually we update RMS during rollout.
        if self.normalize_obs:
            self.obs_rms.update(obs)
            norm_obs = self.obs_rms.normalize(obs)
        else:
            norm_obs = obs
            
        with torch.no_grad():
            obs_t = torch.FloatTensor(norm_obs).unsqueeze(0).to(self.device)
            action, log_prob, _, value = self.network.get_action_and_value(obs_t)
            
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, obs, action, log_prob, reward, value, done):
        """Store step data. Store NORMALIZED obs? Usually we store raw and normalize again?
        Standard implementations store normalized obs if normalization happens before.
        Since we computed action based on normalized obs, we should store that normalized obs.
        Or consistency matters.
        Here we store what we used.
        """
        if self.normalize_obs:
            norm_obs = self.obs_rms.normalize(obs)
            self.obs_buffer.append(norm_obs)
        else:
            self.obs_buffer.append(obs)
            
        self.action_buffer.append(action)
        self.logprob_buffer.append(log_prob)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)
        self.done_buffer.append(done)

    def finish_episode(self, last_val: float = 0.0):
        """
        Compute GAE and update policy.
        """
        if len(self.obs_buffer) == 0:
            return {}
            
        # Convert buffers to tensors
        obs_t = torch.FloatTensor(np.array(self.obs_buffer)).to(self.device)
        act_t = torch.LongTensor(np.array(self.action_buffer)).to(self.device)
        logprob_t = torch.FloatTensor(np.array(self.logprob_buffer)).to(self.device)
        
        # GAE Calculation
        params = self._compute_gae(last_val)
        returns = params['returns']
        advantages = params['advantages']
        
        # Flatten and Update
        b_obs = obs_t
        b_logprobs = logprob_t
        b_actions = act_t
        b_advantages = advantages
        b_returns = returns
        # b_values = torch.FloatTensor(np.array(self.value_buffer)).to(self.device)
        
        metrics = {}
        
        # Optimization Epochs
        for epoch in range(self.n_epochs):
            # Mini-batch indices
            indices = np.arange(len(self.obs_buffer))
            np.random.shuffle(indices)
            
            for start in range(0, len(self.obs_buffer), self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]
                
                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_logprobs = b_logprobs[mb_idx]
                mb_advantages = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]
                
                # Forward pass
                _, new_logprobs, entropy, new_values = self.network.get_action_and_value(mb_obs, mb_actions)
                
                # Ratio
                logratio = new_logprobs - mb_old_logprobs
                ratio = logratio.exp()
                
                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value Loss
                new_values = new_values.view(-1)
                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                
                # Entropy Loss
                entropy_loss = entropy.mean()
                
                # Total Loss
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
            metrics['loss'] = loss.item()
            metrics['pg_loss'] = pg_loss.item()
            metrics['v_loss'] = v_loss.item()
            metrics['entropy'] = entropy_loss.item()
        
        # Clear buffers
        self.clear_buffer()
        return metrics

    def _compute_gae(self, last_val: float) -> Dict[str, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE)."""
        rewards = np.array(self.reward_buffer)
        values = np.array(self.value_buffer + [last_val])
        dones = np.array(self.done_buffer)
        
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - 0.0 # Assuming rollout ends not done
            else:
                nextnonterminal = 1.0 - dones[t+1]
                
            nonterminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * values[t+1] * nonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
            
        returns = advantages + values[:-1]
        
        return {
            'returns': torch.FloatTensor(returns).to(self.device),
            'advantages': torch.FloatTensor(advantages).to(self.device)
        }

    def clear_buffer(self):
        self.obs_buffer = []
        self.action_buffer = []
        self.logprob_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.done_buffer = []
    
    def save(self, path):
        # Save Model AND Running Mean Std
        state = {
            'model': self.network.state_dict(),
            'obs_rms_mean': self.obs_rms.mean,
            'obs_rms_var': self.obs_rms.var,
            'obs_rms_count': self.obs_rms.count,
            'episode_counter': self.episode_counter
        }
        torch.save(state, path)
    
    def load(self, path):
        # Allow loading numpy/pickled objects (safe here as we created them)
        ckpt = torch.load(path, weights_only=False)
        if 'model' in ckpt:
            self.network.load_state_dict(ckpt['model'])
            self.obs_rms.mean = ckpt.get('obs_rms_mean', self.obs_rms.mean)
            self.obs_rms.var = ckpt.get('obs_rms_var', self.obs_rms.var)
            self.obs_rms.count = ckpt.get('obs_rms_count', self.obs_rms.count)
            self.episode_counter = ckpt.get('episode_counter', 0)
        else:
            # Revert to standard state loading
            self.network.load_state_dict(ckpt)
