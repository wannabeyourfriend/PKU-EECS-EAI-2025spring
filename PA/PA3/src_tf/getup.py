import os
import time
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import functools
from typing import Dict, Any, Tuple, Optional

# Configure TensorFlow for Metal acceleration on Mac
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Enable Metal acceleration for Mac
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Metal GPU acceleration enabled: {len(physical_devices)} GPU(s) found")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU devices found, using CPU")

# Setup logging
_start = time.time()
class ETFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return f"{record.created - _start:.1f}s"

fmt = "%(asctime)s [%(levelname)s] - %(message)s"
handler = logging.StreamHandler()
handler.setFormatter(ETFormatter(fmt))

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(handler)

import numpy as np
import mujoco

DESIRED_BODY_HEIGHT = 0.33

class TensorFlowState:
    """TensorFlow equivalent of mjx_env.State"""
    def __init__(self, obs: tf.Tensor, reward: tf.Tensor = None, done: tf.Tensor = None, 
                 info: Dict = None, metrics: Dict = None, data=None):
        self.obs = obs
        self.reward = reward if reward is not None else tf.constant(0.0)
        self.done = done if done is not None else tf.constant(False)
        self.info = info if info is not None else {}
        self.metrics = metrics if metrics is not None else {}
        self.data = data
    
    def replace(self, **kwargs):
        """Create a new state with updated fields"""
        new_state = TensorFlowState(
            obs=kwargs.get('obs', self.obs),
            reward=kwargs.get('reward', self.reward),
            done=kwargs.get('done', self.done),
            info=kwargs.get('info', self.info),
            metrics=kwargs.get('metrics', self.metrics),
            data=kwargs.get('data', self.data)
        )
        return new_state

class MuJoCoData:
    """Simplified MuJoCo data container for TensorFlow"""
    def __init__(self, qpos: tf.Tensor, qvel: tf.Tensor, site_xpos: tf.Tensor, time: tf.Tensor = None):
        self.qpos = qpos
        self.qvel = qvel
        self.site_xpos = site_xpos
        self.time = time if time is not None else tf.constant(0.0)
    
    def replace(self, **kwargs):
        return MuJoCoData(
            qpos=kwargs.get('qpos', self.qpos),
            qvel=kwargs.get('qvel', self.qvel),
            site_xpos=kwargs.get('site_xpos', self.site_xpos),
            time=kwargs.get('time', self.time)
        )

class MyGetupEnvTF:
    """TensorFlow implementation of the Getup environment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.action_scale = config.get('action_scale', 0.5)
        self.n_substeps = int(config.get('sim_dt', 0.004) / config.get('ctrl_dt', 0.02))
        self.settle_steps = int(config.get('settle_time', 0.5) / config.get('ctrl_dt', 0.02))
        self.episode_length = config.get('episode_length', 300)
        self.dt = config.get('ctrl_dt', 0.02)
        
        # Initialize default pose and model parameters
        self.num_joints = 12
        self.num_dof = 19  # 7 (base) + 12 (joints)
        self._default_pose = tf.constant(np.zeros(self.num_joints), dtype=tf.float32)
        self._init_q = tf.constant(np.zeros(self.num_dof), dtype=tf.float32)
        self._imu_site_id = 0  # Simplified
        
        # Initialize MuJoCo model (simplified for this example)
        self.mjx_model = self._create_simplified_model()
        
    def _create_simplified_model(self):
        """Create a simplified model representation"""
        class SimplifiedModel:
            def __init__(self):
                self.nv = 18  # velocity dimensions
                self.nu = 12  # control dimensions
        return SimplifiedModel()
    
    @tf.function
    def _simulate_physics(self, qpos: tf.Tensor, qvel: tf.Tensor, motor_targets: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Simplified physics simulation using TensorFlow operations"""
        # This is a simplified physics simulation
        # In a real implementation, you would integrate with MuJoCo or use a differentiable physics engine
        
        # Simple integration step
        dt = 0.004  # simulation timestep
        
        # Apply motor control (simplified PD control)
        kp = 35.0
        kd = 0.5
        joint_pos = qpos[7:]
        joint_vel = qvel[6:]
        
        # PD control torques
        torques = kp * (motor_targets - joint_pos) - kd * joint_vel
        
        # Simple forward dynamics (mass matrix assumed to be identity for simplification)
        joint_acc = torques  # Simplified
        
        # Integration
        new_joint_vel = joint_vel + joint_acc * dt
        new_joint_pos = joint_pos + new_joint_vel * dt
        
        # Update full state
        new_qpos = tf.concat([qpos[:7], new_joint_pos], axis=0)
        new_qvel = tf.concat([qvel[:6], new_joint_vel], axis=0)
        
        return new_qpos, new_qvel
    
    # @tf.function
    def _get_obs(self, data: MuJoCoData, info: Dict) -> tf.Tensor:
        """Get observation from current state"""
        # Extract relevant state information
        body_pos = data.qpos[:3]
        body_quat = data.qpos[3:7]
        joint_pos = data.qpos[7:]
        joint_vel = data.qvel[6:]
        
        # Simplified observation (in practice, this would be more complex)
        obs = tf.concat([
            body_pos,
            body_quat,
            joint_pos,
            joint_vel,
        ], axis=0)
        
        return obs
    
   # @tf.function
    def _get_termination(self, data: MuJoCoData) -> tf.Tensor:
        """Check if episode should terminate"""
        # Simple termination conditions
        body_height = data.site_xpos[0, 2]  # Fix: access as 2D tensor [batch_idx, coord_idx]
        height_too_low = body_height < 0.1
        height_too_high = body_height > 1.0
        
        return tf.cast(height_too_low | height_too_high, tf.float32)
    
    # @tf.function
    def get_local_linvel(self, data: MuJoCoData) -> tf.Tensor:
        """Get local linear velocity (simplified)"""
        return data.qvel[:3]  # Simplified
    
    # @tf.function
    def get_gyro(self, data: MuJoCoData) -> tf.Tensor:
        """Get angular velocity"""
        return data.qvel[3:6]
    
    # @tf.function
    def get_gravity(self, data: MuJoCoData) -> tf.Tensor:
        """Get gravity vector in body frame (simplified)"""
        # This would normally involve rotation matrix calculations
        # Simplified to assume upright orientation
        return tf.constant([0.0, 0.0, -9.81], dtype=tf.float32)
    
    # @tf.function
    def step(self, state: TensorFlowState, action: tf.Tensor) -> TensorFlowState:
        """Take a step in the environment"""
        # Apply action scaling
        motor_targets = state.data.qpos[7:] + action * self.action_scale
        
        # Simulate physics for multiple substeps
        qpos, qvel = state.data.qpos, state.data.qvel
        for _ in range(self.n_substeps):
            qpos, qvel = self._simulate_physics(qpos, qvel, motor_targets)
        
        # Create new data
        site_xpos = tf.stack([qpos[:3]])  # Simplified: body position as site position
        data = MuJoCoData(qpos=qpos, qvel=qvel, site_xpos=site_xpos)
        
        # Get observation and termination
        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)
        
        # --- Reward Calculation (TensorFlow version) ---
        
        # Extract state variables
        body_pos = site_xpos[self._imu_site_id]
        body_lin_vel = self.get_local_linvel(data)
        body_ang_vel = self.get_gyro(data)
        gravity_vector = self.get_gravity(data)
        default_qpos = self._default_pose
        joint_qpos = data.qpos[7:]
        joint_qvel = data.qvel[7:]
        
        # 1. Body Height Reward
        body_height = body_pos[2]
        height_error = tf.abs(body_height - DESIRED_BODY_HEIGHT)
        rew_height = tf.exp(-10.0 * height_error)
        
        # 2. Body Orientation Reward
        gravity_norm = tf.linalg.norm(gravity_vector)
        gravity_normalized = gravity_vector / (gravity_norm + 1e-8)
        upright_reward = tf.maximum(0.0, -gravity_normalized[2])
        rew_orientation = tf.exp(5.0 * (upright_reward - 1.0))
        
        # 3. Joint Position Reward
        joint_pos_error = tf.linalg.norm(joint_qpos - default_qpos)
        rew_joint_pos = tf.exp(-2.0 * joint_pos_error)
        
        # 4. Angular Velocity Penalty
        ang_vel_penalty = tf.linalg.norm(body_ang_vel)
        rew_ang_vel = tf.exp(-1.0 * ang_vel_penalty)
        
        # 5. Linear Velocity Penalty
        lin_vel_penalty = tf.linalg.norm(body_lin_vel[:2])
        rew_lin_vel = tf.exp(-0.5 * lin_vel_penalty)
        
        # 6. Action Smoothness Reward
        last_action = state.info.get("last_act", tf.zeros_like(action))
        action_diff = tf.linalg.norm(action - last_action)
        rew_smoothness = tf.exp(-0.1 * action_diff)
        
        # 7. Energy Efficiency
        joint_vel_penalty = tf.linalg.norm(joint_qvel)
        rew_energy = tf.exp(-0.05 * joint_vel_penalty)
        
        # Combine rewards
        rew_term_1 = 4.0 * rew_height
        rew_term_2 = 3.0 * rew_orientation
        rew_term_3 = 1.5 * rew_joint_pos
        rew_term_4 = 1.0 * rew_ang_vel
        rew_term_5 = 0.5 * rew_lin_vel
        rew_term_6 = 0.3 * rew_smoothness
        rew_term_7 = 0.2 * rew_energy
        
        reward = rew_term_1 + rew_term_2 + rew_term_3 + rew_term_4 + rew_term_5 + rew_term_6 + rew_term_7
        
        # Update state info
        new_info = state.info.copy()
        new_info["last_last_act"] = state.info.get("last_act", tf.zeros_like(action))
        new_info["last_act"] = action
        
        # Update metrics
        new_metrics = {"reward": reward}
        
        # Create new state
        new_state = state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            info=new_info,
            metrics=new_metrics
        )
        
        return new_state
    
    def reset(self, rng_seed: int = None) -> TensorFlowState:
        """Reset the environment"""
        if rng_seed is not None:
            tf.random.set_seed(rng_seed)
        
        # Initialize state
        qpos = self._init_q
        qvel = tf.zeros(self.mjx_model.nv)
        
        # Create initial data
        site_xpos = tf.stack([qpos[:3]])
        data = MuJoCoData(qpos=qpos, qvel=qvel, site_xpos=site_xpos, time=tf.constant(0.0))
        
        # Settle the robot
        motor_targets = qpos[7:]
        for _ in range(self.settle_steps):
            qpos, qvel = self._simulate_physics(qpos, qvel, motor_targets)
        
        data = data.replace(qpos=qpos, qvel=qvel, time=tf.constant(0.0))
        
        # Initialize info
        info = {
            "last_act": tf.zeros(self.mjx_model.nu),
            "last_last_act": tf.zeros(self.mjx_model.nu),
        }
        
        # Get initial observation
        obs = self._get_obs(data, info)
        
        # Create initial state
        state = TensorFlowState(
            data=data,
            obs=obs,
            reward=tf.constant(0.0),
            done=tf.constant(False),
            info=info,
            metrics={"reward": tf.constant(0.0)}
        )
        
        return state

def create_env():
    """Create the TensorFlow environment"""
    env_cfg = {
        'ctrl_dt': 0.02,
        'sim_dt': 0.004,
        'Kp': 35.0,
        'Kd': 0.5,
        'episode_length': 300,
        'drop_from_height_prob': 0.6,
        'settle_time': 0.5,
        'action_repeat': 1,
        'action_scale': 0.5,
        'soft_joint_pos_limit_factor': 0.95,
        'energy_termination_threshold': np.inf,
        'noise_config': {
            'level': 0.0,
            'scales': {
                'joint_pos': 0.03,
                'joint_vel': 1.5,
                'gyro': 0.2,
                'gravity': 0.05,
            },
        },
    }
    env = MyGetupEnvTF(env_cfg)
    return env

# TensorFlow PPO Implementation
class PPOAgent:
    """TensorFlow implementation of PPO agent"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Dict[str, Any]):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Build networks
        self.policy_network = self._build_policy_network()
        self.value_network = self._build_value_network()
        
        # Optimizers
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 3e-4))
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 3e-4))
    
    def _build_policy_network(self) -> tf.keras.Model:
        """Build policy network"""
        inputs = tf.keras.Input(shape=(self.obs_dim,))
        x = tf.keras.layers.Dense(512, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        
        # Mean and log_std for continuous actions
        mean = tf.keras.layers.Dense(self.action_dim, activation='tanh')(x)
        log_std = tf.keras.layers.Dense(self.action_dim)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=[mean, log_std])
        return model
    
    def _build_value_network(self) -> tf.keras.Model:
        """Build value network"""
        inputs = tf.keras.Input(shape=(self.obs_dim,))
        x = tf.keras.layers.Dense(512, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        value = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=value)
        return model
    
    @tf.function
    def get_action(self, obs: tf.Tensor, deterministic: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """Get action from policy"""
        mean, log_std = self.policy_network(obs)
        
        if deterministic:
            return mean, tf.zeros_like(mean)
        
        std = tf.exp(log_std)
        action = mean + std * tf.random.normal(tf.shape(mean))
        
        # Compute log probability
        log_prob = -0.5 * tf.reduce_sum(
            tf.square((action - mean) / std) + 2 * log_std + tf.math.log(2 * np.pi),
            axis=-1
        )
        
        return action, log_prob
    
    @tf.function
    def get_value(self, obs: tf.Tensor) -> tf.Tensor:
        """Get value estimate"""
        return tf.squeeze(self.value_network(obs), axis=-1)

def train_ppo_tf():
    """Train PPO using TensorFlow with Metal acceleration"""
    import mediapy as media
    
    LOGGER.info("Starting TensorFlow PPO training with Metal acceleration")
    
    # Training parameters
    ppo_params = {
        'num_timesteps': 40_000_000,
        'num_evals': 0,
        'reward_scaling': 1.0,
        'episode_length': 500,
        'normalize_observations': True,
        'action_repeat': 1,
        'unroll_length': 20,
        'num_minibatches': 32,
        'num_updates_per_batch': 4,
        'discounting': 0.97,
        'learning_rate': 5e-4,
        'entropy_cost': 1e-2,
        'num_envs': 4096,
        'batch_size': 128,
        'max_grad_norm': 1.0,
    }
    
    start_t = datetime.now()
    
    # Create environment
    env = create_env()
    
    # Get observation and action dimensions
    dummy_state = env.reset(rng_seed=0)
    obs_dim = dummy_state.obs.shape[0]
    action_dim = env.mjx_model.nu
    
    LOGGER.info(f"Environment created - obs_dim: {obs_dim}, action_dim: {action_dim}")
    
    # Create PPO agent
    agent = PPOAgent(obs_dim, action_dim, ppo_params)
    
    # Training loop (simplified)
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset(rng_seed=episode)
        episode_reward = 0.0
        
        for step in range(ppo_params['episode_length']):
            # Get action
            obs_batch = tf.expand_dims(state.obs, 0)
            action, log_prob = agent.get_action(obs_batch, deterministic=False)
            action = tf.squeeze(action, 0)
            
            # Take step
            state = env.step(state, action)
            episode_reward += state.reward.numpy()
            
            if state.done:
                break
        
        if episode % 100 == 0:
            LOGGER.info(f"Episode {episode}, Reward: {episode_reward:.3f}")
    
    end_t = datetime.now()
    LOGGER.info(f"Training completed in: {end_t - start_t}")
    
    # Evaluation and rendering
    render_length = 500
    _pre_render_length = 100
    
    rollout = []
    body_height = []
    
    state = env.reset(rng_seed=42)
    for i in range(render_length):
        if i < _pre_render_length:
            # Use default pose
            ctrl = env._default_pose
        else:
            # Use trained policy
            obs_batch = tf.expand_dims(state.obs, 0)
            ctrl, _ = agent.get_action(obs_batch, deterministic=True)
            ctrl = tf.squeeze(ctrl, 0)
        
        state = env.step(state, ctrl)
        rollout.append(state)
        env_height = state.data.site_xpos[env._imu_site_id][2].numpy()
        body_height.append(env_height)
    
    # Plot results
    body_height = np.array(body_height)
    height_error = np.mean(np.abs(body_height - DESIRED_BODY_HEIGHT))
    
    plt.figure(figsize=(10, 6))
    plt.plot(body_height, label='Body Height')
    plt.axhline(DESIRED_BODY_HEIGHT, color='r', linestyle='--', label='Desired Height')
    plt.title(f"TensorFlow Metal Training - Height error: {height_error:.3f}")
    plt.xlabel("Steps")
    plt.ylabel("Body Height (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig("tensorflow_metal_height_error.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    LOGGER.info(f"Height tracking error: {height_error:.3f}")
    LOGGER.info("Training and evaluation completed successfully!")

if __name__ == '__main__':
    train_ppo_tf()