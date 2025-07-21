import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Dict, List
import logging

class PolicyNetwork(nn.Module):
    """Policy network for PPO"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int] = [512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
            
        self.backbone = nn.Sequential(*layers)
        
        # Policy head
        self.mean_head = nn.Linear(prev_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and std of action distribution"""
        features = self.backbone(obs)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std)
        return mean, std
        
    def get_action_and_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return log probability"""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
        
    def get_log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of given action"""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)
        
    def get_entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Get entropy of action distribution"""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        return dist.entropy().sum(dim=-1)

class ValueNetwork(nn.Module):
    """Value network for PPO"""
    
    def __init__(self, obs_dim: int, hidden_sizes: List[int] = [512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state value"""
        return self.network(obs).squeeze(-1)

class PPOBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self, obs_dim: int, action_dim: int, buffer_size: int, device: torch.device):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        
        self.observations = torch.zeros((buffer_size, obs_dim), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        
        self.ptr = 0
        self.size = 0
        
    def store(self, obs: torch.Tensor, action: torch.Tensor, reward: float, 
              value: torch.Tensor, log_prob: torch.Tensor, done: bool):
        """Store experience in buffer"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get random batch from buffer"""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices],
            'dones': self.dones[indices]
        }
        
    def compute_advantages_and_returns(self, gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns"""
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        # Convert to numpy for easier computation
        rewards = self.rewards[:self.size].cpu().numpy()
        values = self.values[:self.size].cpu().numpy()
        dones = self.dones[:self.size].cpu().numpy()
        
        # Compute advantages using GAE
        advantages_np = generalized_advantage_estimation(rewards, values, gamma, lam)
        
        # Compute returns
        returns_np = advantages_np + values[:-1] if len(values) > len(advantages_np) else advantages_np + values
        
        advantages[:len(advantages_np)] = torch.tensor(advantages_np, device=self.device)
        returns[:len(returns_np)] = torch.tensor(returns_np, device=self.device)
        
        return advantages[:self.size], returns[:self.size]

def monte_carlo_advantage(rewards: np.ndarray, values: np.ndarray, gamma: float) -> np.ndarray:
    """Monte Carlo advantage estimation"""
    T = len(rewards)
    advantages = np.zeros(T)
    
    for t in range(T):
        Gt = 0
        for k in range(t, T):
            Gt += (gamma ** (k - t)) * rewards[k]
        advantages[t] = Gt - values[t]
    
    return advantages

def td_residual_advantage(rewards: np.ndarray, values: np.ndarray, gamma: float) -> np.ndarray:
    """TD(0) residual advantage estimation"""
    T = len(rewards)
    advantages = np.zeros(T)
    
    for t in range(T):
        advantages[t] = rewards[t] + gamma * values[t + 1] - values[t]
    
    return advantages

def generalized_advantage_estimation(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """Generalized Advantage Estimation (GAE)"""
    T = len(rewards)
    advantages = np.zeros(T)
    
    # Compute TD errors
    td_errors = np.zeros(T)
    for t in range(T):
        if t < T - 1:
            td_errors[t] = rewards[t] + gamma * values[t + 1] - values[t]
        else:
            td_errors[t] = rewards[t] - values[t]
    
    # Compute GAE
    gae = 0
    for t in reversed(range(T)):
        gae = td_errors[t] + gamma * lam * gae
        advantages[t] = gae
    
    return advantages

def compute_policy_loss(ratio: torch.Tensor, adv: torch.Tensor, entropy: torch.Tensor, 
                       epsilon: float, entropy_weight: float) -> torch.Tensor:
    """Compute PPO policy loss"""
    # Clipped surrogate objective
    unclipped_objective = ratio * adv
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    clipped_objective = clipped_ratio * adv
    
    policy_objective = torch.minimum(unclipped_objective, clipped_objective)
    mean_policy_objective = torch.mean(policy_objective)
    
    # Add entropy bonus
    entropy_bonus = entropy_weight * torch.mean(entropy)
    
    # PPO loss (negative because we want to maximize)
    policy_loss = -(mean_policy_objective + entropy_bonus)
    
    return policy_loss

def compute_value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Compute value function loss"""
    return F.mse_loss(values, returns)

class PPOAgent:
    """PPO Agent implementation"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.policy = PolicyNetwork(obs_dim, action_dim, 
                                  config.get('policy_hidden_sizes', [512, 256, 128])).to(self.device)
        self.value_net = ValueNetwork(obs_dim, 
                                    config.get('value_hidden_sizes', [512, 256, 128])).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), 
                                         lr=config.get('learning_rate', 3e-4))
        self.value_optimizer = optim.Adam(self.value_net.parameters(), 
                                        lr=config.get('learning_rate', 3e-4))
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('lam', 0.95)
        self.epsilon = config.get('epsilon', 0.2)
        self.entropy_weight = config.get('entropy_weight', 0.01)
        self.value_loss_weight = config.get('value_loss_weight', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Buffer
        self.buffer = PPOBuffer(obs_dim, action_dim, 
                              config.get('buffer_size', 10000), self.device)
        
    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from policy"""
        obs = obs.to(self.device)
        with torch.no_grad():
            action, log_prob = self.policy.get_action_and_log_prob(obs)
            value = self.value_net(obs)
        return action, log_prob, value
        
    def update(self, batch_size: int = 64, num_epochs: int = 10):
        """Update policy and value networks"""
        # Compute advantages and returns
        advantages, returns = self.buffer.compute_advantages_and_returns(self.gamma, self.lam)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(num_epochs):
            # Get batch
            batch = self.buffer.get_batch(batch_size)
            
            # Current policy evaluation
            current_log_probs = self.policy.get_log_prob(batch['observations'], batch['actions'])
            current_values = self.value_net(batch['observations'])
            current_entropy = self.policy.get_entropy(batch['observations'])
            
            # Compute ratio
            ratio = torch.exp(current_log_probs - batch['log_probs'])
            
            # Get corresponding advantages and returns
            batch_indices = torch.randint(0, len(advantages), (batch_size,), device=self.device)
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            
            # Compute losses
            policy_loss = compute_policy_loss(ratio, batch_advantages, current_entropy, 
                                            self.epsilon, self.entropy_weight)
            value_loss = compute_value_loss(current_values, batch_returns)
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item()
        }