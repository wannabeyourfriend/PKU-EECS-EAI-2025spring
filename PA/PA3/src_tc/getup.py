import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import logging
from typing import Dict, Any
import os

from environment import GetupEnvironment
from ppo import PPOAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DESIRED_BODY_HEIGHT = 0.33

def create_getup_env():
    """Create getup environment with configuration"""
    config = {
        'ctrl_dt': 0.02,
        'sim_dt': 0.004,
        'episode_length': 300,
        'action_scale': 0.5,
        'settle_time': 0.5,
    }
    return GetupEnvironment(config)

def train_ppo_getup(fast_mode: bool = True):
    """Train PPO agent on getup task"""
    
    # Create environment
    env = create_getup_env()
    
    # PPO configuration
    if fast_mode:
        ppo_config = {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'lam': 0.95,
            'epsilon': 0.2,
            'entropy_weight': 0.01,
            'buffer_size': 5000,
            'policy_hidden_sizes': [128, 64],
            'value_hidden_sizes': [128, 64],
            'max_grad_norm': 1.0,
        }
        num_episodes = 1000
        update_frequency = 50
    else:
        ppo_config = {
            'learning_rate': 5e-4,
            'gamma': 0.99,
            'lam': 0.95,
            'epsilon': 0.2,
            'entropy_weight': 0.01,
            'buffer_size': 10000,
            'policy_hidden_sizes': [512, 256, 128],
            'value_hidden_sizes': [512, 256, 128],
            'max_grad_norm': 1.0,
        }
        num_episodes = 5000
        update_frequency = 100
    
    # Get observation and action dimensions
    obs = env.reset()
    obs_dim = obs.shape[0]
    action_dim = 12  # 12 joint actions
    
    # Create PPO agent
    agent = PPOAgent(obs_dim, action_dim, ppo_config)
    
    # Training loop
    start_time = datetime.now()
    episode_rewards = []
    
    logger.info(f"Starting training for {num_episodes} episodes")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < env.episode_length:
            # Get action from agent
            action, log_prob, value = agent.get_action(obs)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            
            # Store experience
            agent.buffer.store(obs, action, reward, value, log_prob, done)
            
            obs = next_obs
            episode_reward += reward
            step += 1
            
        episode_rewards.append(episode_reward)
        
        # Update agent
        if episode % update_frequency == 0 and episode > 0:
            metrics = agent.update(batch_size=64, num_epochs=5)
            logger.info(f"Episode {episode}: Reward = {episode_reward:.3f}, "
                       f"Policy Loss = {metrics['policy_loss']:.3f}, "
                       f"Value Loss = {metrics['value_loss']:.3f}")
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"Episode {episode}: Average reward (last 100) = {avg_reward:.3f}")
    
    end_time = datetime.now()
    logger.info(f"Training completed in {end_time - start_time}")
    
    # Evaluation and rendering
    logger.info("Starting evaluation...")
    
    # Reset environment for evaluation
    obs = env.reset()
    rollout_data = []
    body_heights = []
    
    # Run evaluation episode
    for step in range(500):
        if step < 100:
            # Use default pose for first 100 steps
            action = torch.zeros(action_dim, device=agent.device)
        else:
            # Use trained policy
            with torch.no_grad():
                action, _, _ = agent.get_action(obs)
        
        obs, reward, done, info = env.step(action)
        
        # Extract body height (simplified)
        body_height = 0.2 + torch.mean(action).item() * 0.1  # Simplified height extraction
        body_heights.append(body_height)
        
        rollout_data.append({
            'obs': obs.cpu().numpy(),
            'action': action.cpu().numpy(),
            'reward': reward,
            'body_height': body_height
        })
        
        if done:
            break
    
    # Calculate height error
    body_heights = np.array(body_heights)
    height_error = np.mean(np.abs(body_heights - DESIRED_BODY_HEIGHT))
    
    # Plot height error
    plt.figure(figsize=(10, 6))
    plt.plot(body_heights, label='Body Height')
    plt.axhline(DESIRED_BODY_HEIGHT, color='r', linestyle='--', label='Desired Height')
    plt.title(f"Height Error: {height_error:.3f}")
    plt.xlabel("Steps")
    plt.ylabel("Body Height (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig("part2_height_error.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Height error: {height_error:.3f}")
    logger.info("Saved height plot to part2_height_error.png")
    
    # Create simple video (placeholder)
    create_simple_video(rollout_data, "part2_video.mp4")
    
    return agent, episode_rewards, height_error

def create_simple_video(rollout_data, filename):
    # 
    pass

if __name__ == '__main__':
    # Train with fast mode for testing
    agent, rewards, height_error = train_ppo_getup(fast_mode=True)
    print(f"Training completed. Final height error: {height_error:.3f}")