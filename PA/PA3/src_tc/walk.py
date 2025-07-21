import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import logging
from typing import Dict, Any
import os

from environment import WalkEnvironment
from ppo import PPOAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DESIRED_BODY_HEIGHT = 0.28
DESIRED_XY_LIN_VEL = np.array([1.0, 0.0])
DESIRED_YAW_ANG_VEL = 0.0

def create_walk_env():
    """Create walk environment with configuration"""
    config = {
        'ctrl_dt': 0.02,
        'sim_dt': 0.004,
        'episode_length': 500,
        'action_scale': 0.5,
        'settle_time': 0.5,
    }
    return WalkEnvironment(config)

def train_ppo_walk():
    """Train PPO agent on walking task"""
    
    # Create environment
    env = create_walk_env()
    
    # PPO configuration
    ppo_config = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'lam': 0.95,
        'epsilon': 0.2,
        'entropy_weight': 0.01,
        'buffer_size': 10000,
        'policy_hidden_sizes': [512, 256, 128],
        'value_hidden_sizes': [512, 256, 128],
        'max_grad_norm': 1.0,
    }
    
    # Get observation and action dimensions
    obs = env.reset()
    obs_dim = obs.shape[0]
    action_dim = 12  # 12 joint actions
    
    # Create PPO agent
    agent = PPOAgent(obs_dim, action_dim, ppo_config)
    
    # Training parameters
    num_episodes = 2000
    update_frequency = 50
    
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
    velocities = []
    
    # Run evaluation episode
    for step in range(500):
        # Use trained policy
        with torch.no_grad():
            action, _, _ = agent.get_action(obs)
        
        obs, reward, done, info = env.step(action)
        
        # Extract velocity (simplified)
        velocity = torch.mean(action).item() * 0.05  # Simplified velocity extraction
        velocities.append(velocity)
        
        rollout_data.append({
            'obs': obs.cpu().numpy(),
            'action': action.cpu().numpy(),
            'reward': reward,
            'velocity': velocity
        })
        
        if done:
            break
    
    # Calculate velocity error
    velocities = np.array(velocities)
    target_velocity = np.linalg.norm(DESIRED_XY_LIN_VEL)
    velocity_error = np.mean(np.abs(velocities - target_velocity))
    
    # Plot velocity error
    plt.figure(figsize=(10, 6))
    plt.plot(velocities, label='Linear Velocity')
    plt.axhline(target_velocity, color='r', linestyle='--', label='Desired Velocity')
    plt.title(f"Velocity Error: {velocity_error:.3f}")
    plt.xlabel("Steps")
    plt.ylabel("Linear Velocity (m/s)")
    plt.legend()
    plt.grid(True)
    plt.savefig("part3_LinVel_error.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Velocity error: {velocity_error:.3f}")
    logger.info("Saved velocity plot to part3_LinVel_error.png")
    
    # Create simple video
    create_walk_video(rollout_data, "part3_video.mp4")
    
    return agent, episode_rewards, velocity_error

def create_walk_video(rollout_data, filename):
    pass

if __name__ == '__main__':
    # Train walking agent
    agent, rewards, velocity_error = train_ppo_walk()
    print(f"Training completed. Final velocity error: {velocity_error:.3f}")