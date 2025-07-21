import numpy as np
import torch
import torch.nn as nn
import mujoco
import cv2
from typing import Dict, Any, Tuple, Optional
import logging

class BaseEnvironment:
    """Base environment class for PyTorch-based RL environments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dt = config.get('ctrl_dt', 0.02)
        self.episode_length = config.get('episode_length', 300)
        self.action_scale = config.get('action_scale', 0.5)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MuJoCo model (placeholder - would need actual model file)
        self.model = None
        self.data = None
        self._setup_model()
        
    def _setup_model(self):
        """Setup MuJoCo model - to be implemented by subclasses"""
        pass
        
    def reset(self) -> torch.Tensor:
        """Reset environment and return initial observation"""
        raise NotImplementedError
        
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Take a step in the environment"""
        raise NotImplementedError
        
    def render(self, mode='rgb_array', width=640, height=480):
        """Render the environment"""
        if self.model is None or self.data is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
            
        # Placeholder rendering
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

class State:
    """State container for environment data"""
    
    def __init__(self, obs: torch.Tensor, reward: float = 0.0, done: bool = False, 
                 info: Dict = None, metrics: Dict = None):
        self.obs = obs
        self.reward = reward
        self.done = done
        self.info = info or {}
        self.metrics = metrics or {}
        
    def replace(self, **kwargs):
        """Create a new state with updated values"""
        new_state = State(self.obs, self.reward, self.done, self.info.copy(), self.metrics.copy())
        for key, value in kwargs.items():
            setattr(new_state, key, value)
        return new_state

class GetupEnvironment(BaseEnvironment):
    """PyTorch implementation of the Getup environment"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.desired_body_height = 0.33
        self.n_substeps = int(config.get('sim_dt', 0.004) / config.get('ctrl_dt', 0.02))
        self.settle_steps = int(config.get('settle_time', 0.5) / config.get('ctrl_dt', 0.02))
        
        # Initialize default pose and other parameters
        self.default_pose = torch.zeros(12, device=self.device)  # 12 joint angles
        self.init_q = torch.zeros(19, device=self.device)  # 7 (base) + 12 (joints)
        
        # State tracking
        self.current_step = 0
        
    def reset(self) -> torch.Tensor:
        """Reset the environment"""
        self.current_step = 0
        
        # Initialize info dictionary first
        self.info = {
            "last_act": torch.zeros(12, device=self.device),
            "last_last_act": torch.zeros(12, device=self.device),
        }
        
        # Initialize position and velocity
        qpos = self.init_q.clone()
        qvel = torch.zeros(18, device=self.device)  # 6 (base) + 12 (joints)
        
        # Create initial observation
        obs = self._get_obs(qpos, qvel)
        
        return obs
        
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Take a step in the environment"""
        action = action.to(self.device)
        
        # Apply action scaling
        motor_targets = self.default_pose + action * self.action_scale
        
        # Simulate physics (simplified)
        qpos, qvel = self._simulate_step(motor_targets)
        
        # Get observation
        obs = self._get_obs(qpos, qvel)
        
        # Calculate reward
        reward = self._calculate_reward(qpos, qvel, action)
        
        # Check termination
        done = self._check_termination(qpos, qvel) or self.current_step >= self.episode_length
        
        # Update info
        self.info["last_last_act"] = self.info["last_act"].clone()
        self.info["last_act"] = action.clone()
        
        self.current_step += 1
        
        info = {"step": self.current_step}
        
        return obs, reward, done, info
        
    def _simulate_step(self, motor_targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate one physics step (simplified)"""
        # This is a simplified simulation - in practice would use MuJoCo
        # For now, just return some dummy values that change based on actions
        
        # Simulate some basic physics
        qpos = self.init_q.clone()
        qvel = torch.zeros(18, device=self.device)
        
        # Simple height simulation based on joint targets
        height_effect = torch.mean(motor_targets) * 0.1
        qpos[2] = 0.2 + height_effect  # z-position
        
        return qpos, qvel
        
    def _get_obs(self, qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
        """Get observation from state"""
        # Extract relevant state information
        body_pos = qpos[:3]  # x, y, z position
        body_quat = qpos[3:7]  # quaternion orientation
        joint_pos = qpos[7:]  # joint positions
        joint_vel = qvel[6:]  # joint velocities (excluding base)
        
        # Compute derived quantities
        gravity_vector = self._compute_gravity_vector(body_quat)
        body_lin_vel = qvel[:3]  # simplified
        body_ang_vel = qvel[3:6]  # simplified
        
        # Concatenate observation
        obs = torch.cat([
            body_pos,
            gravity_vector,
            joint_pos - self.default_pose,
            joint_vel,
            body_lin_vel,
            body_ang_vel,
            self.info.get("last_act", torch.zeros(12, device=self.device))
        ])
        
        return obs
        
    def _compute_gravity_vector(self, quat: torch.Tensor) -> torch.Tensor:
        """Compute gravity vector in body frame"""
        # Simplified gravity vector computation
        # In practice, would use proper quaternion rotation
        return torch.tensor([0.0, 0.0, -1.0], device=self.device)
        
    def _calculate_reward(self, qpos: torch.Tensor, qvel: torch.Tensor, action: torch.Tensor) -> float:
        """Calculate reward for getup task"""
        # Extract state information
        body_pos = qpos[:3]
        body_quat = qpos[3:7]
        joint_pos = qpos[7:]
        body_ang_vel = qvel[3:6]
        
        # 1. Body height reward
        body_height = body_pos[2]
        height_error = torch.abs(body_height - self.desired_body_height)
        rew_height = torch.exp(-5.0 * height_error)
        
        # 2. Body orientation reward
        gravity_vector = self._compute_gravity_vector(body_quat)
        gravity_norm = torch.norm(gravity_vector)
        gravity_normalized = gravity_vector / (gravity_norm + 1e-8)
        upright_alignment = -gravity_normalized[2]
        rew_orientation = torch.clamp(upright_alignment, min=0.0)
        
        # 3. Joint position reward
        joint_pos_error = torch.norm(joint_pos - self.default_pose)
        rew_joint_pos = torch.exp(-0.5 * joint_pos_error)
        
        # 4. Angular velocity reward
        ang_vel_magnitude = torch.norm(body_ang_vel)
        rew_ang_vel = torch.exp(-0.1 * ang_vel_magnitude)
        
        # Combine rewards
        reward = (2.0 * rew_height + 
                 1.5 * rew_orientation + 
                 0.5 * rew_joint_pos + 
                 0.3 * rew_ang_vel)
        
        return reward.item()
        
    def _check_termination(self, qpos: torch.Tensor, qvel: torch.Tensor) -> bool:
        """Check if episode should terminate"""
        # Simple termination conditions
        body_height = qpos[2]
        if body_height < 0.1 or body_height > 1.0:
            return True
        return False

class WalkEnvironment(BaseEnvironment):
    """PyTorch implementation of the Walk environment"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.desired_body_height = 0.28
        self.desired_xy_lin_vel = torch.tensor([1.0, 0.0], device=self.device)
        self.desired_yaw_ang_vel = 0.0
        
        # Initialize default pose and other parameters
        self.default_pose = torch.zeros(12, device=self.device)
        self.init_q = torch.zeros(19, device=self.device)
        
        self.current_step = 0
        
    def reset(self) -> torch.Tensor:
        """Reset the environment"""
        self.current_step = 0
        
        # Initialize with random perturbations
        qpos = self.init_q.clone()
        qvel = torch.zeros(18, device=self.device)
        
        # Add random initial position
        dxy = torch.uniform(-0.5, 0.5, (2,), device=self.device)
        qpos[:2] += dxy
        
        # Add random initial velocity
        qvel[:6] = torch.uniform(-0.5, 0.5, (6,), device=self.device)
        
        obs = self._get_obs(qpos, qvel)
        
        return obs
        
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Take a step in the environment"""
        action = action.to(self.device)
        
        # Apply action
        motor_targets = self.default_pose + action * self.action_scale
        
        # Simulate physics
        qpos, qvel = self._simulate_step(motor_targets)
        
        # Get observation
        obs = self._get_obs(qpos, qvel)
        
        # Calculate reward
        reward = self._calculate_reward(qpos, qvel, action)
        
        # Check termination
        done = self._check_termination(qpos, qvel) or self.current_step >= self.episode_length
        
        # Update info
        self.info["last_last_act"] = self.info["last_act"].clone()
        self.info["last_act"] = action.clone()
        
        self.current_step += 1
        
        info = {"step": self.current_step}
        
        return obs, reward, done, info
        
    def _simulate_step(self, motor_targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate one physics step"""
        # Simplified simulation
        qpos = self.init_q.clone()
        qvel = torch.zeros(18, device=self.device)
        
        # Simulate forward motion based on actions
        forward_effect = torch.mean(motor_targets) * 0.05
        qpos[0] += forward_effect  # x-position
        qvel[0] = forward_effect / self.dt  # x-velocity
        
        return qpos, qvel
        
    def _get_obs(self, qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
        """Get observation from state"""
        body_pos = qpos[:3]
        joint_pos = qpos[7:]
        joint_vel = qvel[6:]
        body_lin_vel = qvel[:3]
        body_ang_vel = qvel[3:6]
        
        gravity_vector = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        
        obs = torch.cat([
            body_lin_vel,
            body_ang_vel,
            gravity_vector,
            joint_pos - self.default_pose,
            joint_vel,
            self.info.get("last_act", torch.zeros(12, device=self.device))
        ])
        
        return obs
        
    def _calculate_reward(self, qpos: torch.Tensor, qvel: torch.Tensor, action: torch.Tensor) -> float:
        """Calculate reward for walking task"""
        body_pos = qpos[:3]
        body_lin_vel = qvel[:3]
        body_ang_vel = qvel[3:6]
        
        # Velocity tracking rewards
        lin_vel_error = torch.norm(body_lin_vel[:2] - self.desired_xy_lin_vel)
        tracking_lin_vel = torch.exp(-2.0 * lin_vel_error)
        
        ang_vel_error = torch.abs(body_ang_vel[2] - self.desired_yaw_ang_vel)
        tracking_ang_vel = torch.exp(-1.0 * ang_vel_error)
        
        # Height reward
        height_error = torch.abs(body_pos[2] - self.desired_body_height)
        reward_height = torch.exp(-3.0 * height_error)
        
        # Orientation reward (simplified)
        reward_orientation = 0.0  # Placeholder
        
        # Combine rewards
        reward = (tracking_lin_vel + 
                 0.5 * tracking_ang_vel + 
                 0.2 * reward_height + 
                 -5.0 * reward_orientation)
        
        return torch.clamp(reward * self.dt, 0.0, 10000.0).item()
        
    def _check_termination(self, qpos: torch.Tensor, qvel: torch.Tensor) -> bool:
        """Check termination conditions"""
        body_height = qpos[2]
        if body_height < 0.1 or body_height > 1.0:
            return True
        return False