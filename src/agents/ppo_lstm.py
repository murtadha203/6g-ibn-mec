import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(RecurrentActorCritic, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Feature Extractor
        self.feat_fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh()
        )
        
        # LSTM Core
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True)
        
        # Heads
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, hidden):
        """
        Forward pass for a sequence or single step.
        x shape: (batch, seq_len, obs_dim)
        hidden: (h, c) tuples
        """
        # Encode features
        batch_size, seq_len, _ = x.size()
        x_feat = self.feat_fc(x.view(-1, x.size(-1)))
        x_feat = x_feat.view(batch_size, seq_len, -1)
        
        # Recurrent Pass
        self.lstm.flatten_parameters()
        lstm_out, new_hidden = self.lstm(x_feat, hidden)
        
        return lstm_out, new_hidden

    def get_action_and_value(self, x, hidden, action=None):
        """
        Sampling Action (Inference or Training).
        x shape: (batch, 1, obs_dim) usually for inference.
        """
        lstm_out, new_hidden = self.forward(x, hidden)
        
        # Use last output for action
        # If seq_len > 1 (Training), we process all time steps
        logits = self.actor(lstm_out)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(lstm_out), new_hidden

class RecurrentPPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, hidden_dim=128, gamma=0.99, batch_size=64):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.device = torch.device("cpu")
        self.network = RecurrentActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Running Mean Std for normalization
        self.obs_mean = np.zeros(obs_dim)
        self.obs_var = np.ones(obs_dim)
        self.count = 1e-4

    def normalize(self, obs):
        self.count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.count
        m_a = self.obs_var * (self.count - 1)
        m_b = 0 # Simplified update
        self.obs_var = (m_a + delta*(obs - self.obs_mean)) / self.count
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)

    def get_initial_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

    def select_action(self, obs, hidden):
        """
        Step-by-step action selection.
        obs: (obs_dim,) numpy
        """
        # Normalize
        obs = self.normalize(obs)
        
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).view(1, 1, -1).to(self.device)
            action, logprob, _, value, new_hidden = self.network.get_action_and_value(obs_t, hidden)
            
        return action.item(), logprob.item(), value.item(), new_hidden

    def update(self, rollouts):
        """
        Recurrent PPO Update.
        rollouts: List of Trajectories.
        Each trajectory: {'obs': [], 'act': [], ...}
        """
        # Flatten trajectories but keep sequence structure?
        # Standard PPO shuffles minibatches. 
        # R-PPO must shuffle *Trajectories* or *Chunks*.
        
        # For simplicity: Pad trajectories to same length and batch them?
        # Or concat all sequences and train?
        pass # Implementation in Trainer or specialized logic
