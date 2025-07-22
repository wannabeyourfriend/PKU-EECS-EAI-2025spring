import os
import time
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import functools
from typing import Dict, Any, Tuple, Optional
import numpy as np
import psutil
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Import wandb for monitoring
import wandb

# Constants
DESIRED_BODY_HEIGHT = 0.33

# Configure TensorFlow for Metal acceleration on Mac
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# GPU monitoring utilities
class GPUMonitor:
    """Monitor GPU usage and system metrics"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start GPU monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitor loop that runs in separate thread"""
        while self.monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Get TensorFlow GPU memory info
                gpu_memory_info = self._get_tf_gpu_memory()
                
                # Log to wandb
                wandb.log({
                    "system/cpu_percent": cpu_percent,
                    "system/memory_percent": memory.percent,
                    "system/memory_used_gb": memory.used / (1024**3),
                    "system/memory_available_gb": memory.available / (1024**3),
                    **gpu_memory_info
                })
                
                time.sleep(5)  # Monitor every 5 seconds
            except Exception as e:
                LOGGER.warning(f"GPU monitoring error: {e}")
                time.sleep(5)
    
    def _get_tf_gpu_memory(self):
        """Get TensorFlow GPU memory usage"""
        try:
            gpu_devices = tf.config.list_physical_devices('GPU')
            if not gpu_devices:
                return {"gpu/memory_used_mb": 0, "gpu/memory_total_mb": 0}
            
            # Get memory info from TensorFlow
            details = tf.config.experimental.get_memory_info('GPU:0')
            current_mb = details['current'] / (1024 * 1024)
            peak_mb = details['peak'] / (1024 * 1024)
            
            return {
                "gpu/memory_current_mb": current_mb,
                "gpu/memory_peak_mb": peak_mb,
                "gpu/device_count": len(gpu_devices)
            }
        except Exception as e:
            LOGGER.warning(f"Failed to get GPU memory info: {e}")
            return {"gpu/memory_used_mb": 0, "gpu/memory_total_mb": 0}

# Enable Metal acceleration for Mac
def setup_metal_gpu():
    """Setup Metal GPU acceleration for Apple Silicon"""
    print("Setting up TensorFlow Metal GPU acceleration...")
    
    # Check if tensorflow-metal is available
    try:
        import tensorflow_metal
        print("✓ tensorflow-metal plugin detected")
    except ImportError:
        print("⚠ tensorflow-metal not found. Install with: pip install tensorflow-metal")
        return False
    
    # Configure GPU devices
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        try:
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"✓ Metal GPU acceleration enabled: {len(gpu_devices)} GPU(s) found")
            
            # Test GPU computation
            with tf.device('/GPU:0'):
                test_tensor = tf.random.normal([100, 100])
                result = tf.matmul(test_tensor, test_tensor)
                print("✓ GPU computation test passed")
            return True
        except Exception as e:
            print(f"✗ GPU setup error: {e}")
            return False
    else:
        print("✗ No GPU devices found")
        return False

gpu_available = setup_metal_gpu()

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

class TensorFlowState:
    """TensorFlow state container for environment data"""
    
    def __init__(self, data: MuJoCoData, obs: tf.Tensor, reward: tf.Tensor, 
                 done: tf.Tensor, info: Dict, metrics: Dict = None):
        self.data = data
        self.obs = obs
        self.reward = reward
        self.done = done
        self.info = info
        self.metrics = metrics or {}
        
    def replace(self, **kwargs):
        """Create a new state with updated values"""
        return TensorFlowState(
            data=kwargs.get('data', self.data),
            obs=kwargs.get('obs', self.obs),
            reward=kwargs.get('reward', self.reward),
            done=kwargs.get('done', self.done),
            info=kwargs.get('info', self.info),
            metrics=kwargs.get('metrics', self.metrics)
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
        rew_action_rate = tf.exp(-0.1 * action_diff)
        
        # 7. Energy Efficiency
        joint_vel_penalty = tf.linalg.norm(joint_qvel)
        rew_energy = tf.exp(-0.05 * joint_vel_penalty)
        
        # 8. Termination Penalty
        # Add a penalty for being close to termination conditions
        body_height = data.site_xpos[0, 2]
        height_too_low = tf.maximum(0.0, 0.2 - body_height) * 5.0
        height_too_high = tf.maximum(0.0, body_height - 0.9) * 5.0
        rew_termination = tf.exp(-(height_too_low + height_too_high))
        
        # 9. Torque Penalty (simplified)
        torque_penalty = tf.linalg.norm(action)
        rew_torques = tf.exp(-0.05 * torque_penalty)
        
        # Hierarchical reward combination (similar to walk.py)
        # First level: Primary stability rewards
        primary_reward = 0.5 * rew_height + 0.3 * rew_orientation + 0.2 * rew_termination
        
        # Second level: Posture and motion quality
        posture_reward = 0.4 * rew_joint_pos + 0.3 * rew_ang_vel + 0.3 * rew_lin_vel
        
        # Third level: Efficiency and smoothness
        efficiency_reward = 0.4 * rew_action_rate + 0.3 * rew_energy + 0.3 * rew_torques
        
        # Combine all levels with appropriate weights
        reward = 0.6 * primary_reward + 0.3 * posture_reward + 0.1 * efficiency_reward
        
        # Scale reward by time step for consistency
        reward = reward * self.dt
        
        # Clip reward to reasonable range
        reward = tf.clip_by_value(reward, -10.0, 10.0)
        
        # Update state info
        new_info = state.info.copy()
        new_info["last_last_act"] = state.info.get("last_act", tf.zeros_like(action))
        new_info["last_act"] = action
        
        # Store detailed reward components for wandb logging
        reward_components = {
            "rew_height": rew_height.numpy(),
            "rew_orientation": rew_orientation.numpy(),
            "rew_joint_pos": rew_joint_pos.numpy(),
            "rew_ang_vel": rew_ang_vel.numpy(),
            "rew_lin_vel": rew_lin_vel.numpy(),
            "rew_action_rate": rew_action_rate.numpy(),
            "rew_energy": rew_energy.numpy(),
            "rew_termination": rew_termination.numpy(),
            "rew_torques": rew_torques.numpy(),
            "primary_reward": primary_reward.numpy(),
            "posture_reward": posture_reward.numpy(),
            "efficiency_reward": efficiency_reward.numpy(),
            "height_error": height_error.numpy(),
            "body_height": body_height.numpy(),
        }
        
        # Update metrics
        new_metrics = {
            "reward": reward,
            "reward_components": reward_components
        }
        
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
        
        # Build networks on GPU if available
        device = '/GPU:0' if gpu_available else '/CPU:0'
        with tf.device(device):
            self.policy_network = self._build_policy_network()
            self.value_network = self._build_value_network()
        
        # Optimizers
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 3e-4))
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 3e-4))
        
        # Training metrics
        self.policy_loss_metric = tf.keras.metrics.Mean()
        self.value_loss_metric = tf.keras.metrics.Mean()
        self.entropy_metric = tf.keras.metrics.Mean()
        self.kl_divergence_metric = tf.keras.metrics.Mean()
    
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
    
    @tf.function
    def compute_policy_loss(self, obs, actions, advantages, old_log_probs):
        """Compute PPO policy loss"""
        mean, log_std = self.policy_network(obs)
        std = tf.exp(log_std)
        
        # Compute new log probabilities
        new_log_probs = -0.5 * tf.reduce_sum(
            tf.square((actions - mean) / std) + 2 * log_std + tf.math.log(2 * np.pi),
            axis=-1
        )
        
        # Compute ratio
        ratio = tf.exp(new_log_probs - old_log_probs)
        
        # Compute clipped surrogate loss
        clip_ratio = self.config.get('clip_ratio', 0.2)
        clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
        
        policy_loss = -tf.reduce_mean(tf.minimum(
            ratio * advantages,
            clipped_ratio * advantages
        ))
        
        # Compute entropy bonus
        entropy = tf.reduce_mean(tf.reduce_sum(log_std + 0.5 * tf.math.log(2 * np.pi * np.e), axis=-1))
        entropy_cost = self.config.get('entropy_cost', 1e-2)
        
        # Compute KL divergence for monitoring
        kl_div = tf.reduce_mean(old_log_probs - new_log_probs)
        
        total_loss = policy_loss - entropy_cost * entropy
        
        return total_loss, policy_loss, entropy, kl_div
    
    @tf.function
    def compute_value_loss(self, obs, returns):
        """Compute value function loss"""
        values = tf.squeeze(self.value_network(obs), axis=-1)
        value_loss = tf.reduce_mean(tf.square(returns - values))
        return value_loss
    
    def update_networks(self, batch_data):
        """Update policy and value networks"""
        obs, actions, advantages, returns, old_log_probs = batch_data
        
        # Update policy network
        with tf.GradientTape() as tape:
            total_loss, policy_loss, entropy, kl_div = self.compute_policy_loss(
                obs, actions, advantages, old_log_probs
            )
        
        policy_grads = tape.gradient(total_loss, self.policy_network.trainable_variables)
        
        # Clip gradients
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        policy_grads, _ = tf.clip_by_global_norm(policy_grads, max_grad_norm)
        
        self.policy_optimizer.apply_gradients(
            zip(policy_grads, self.policy_network.trainable_variables)
        )
        
        # Update value network
        with tf.GradientTape() as tape:
            value_loss = self.compute_value_loss(obs, returns)
        
        value_grads = tape.gradient(value_loss, self.value_network.trainable_variables)
        value_grads, _ = tf.clip_by_global_norm(value_grads, max_grad_norm)
        
        self.value_optimizer.apply_gradients(
            zip(value_grads, self.value_network.trainable_variables)
        )
        
        # Update metrics
        self.policy_loss_metric.update_state(policy_loss)
        self.value_loss_metric.update_state(value_loss)
        self.entropy_metric.update_state(entropy)
        self.kl_divergence_metric.update_state(kl_div)
        
        return {
            'policy_loss': policy_loss.numpy(),
            'value_loss': value_loss.numpy(),
            'entropy': entropy.numpy(),
            'kl_divergence': kl_div.numpy(),
            'total_loss': total_loss.numpy()
        }
    
    def get_training_metrics(self):
        """Get current training metrics"""
        metrics = {
            'actor/policy_loss': self.policy_loss_metric.result().numpy(),
            'critic/value_loss': self.value_loss_metric.result().numpy(),
            'actor/entropy': self.entropy_metric.result().numpy(),
            'actor/kl_divergence': self.kl_divergence_metric.result().numpy(),
        }
        
        # Reset metrics
        self.policy_loss_metric.reset_states()
        self.value_loss_metric.reset_states()
        self.entropy_metric.reset_states()
        self.kl_divergence_metric.reset_states()
        
        return metrics

def train_ppo_tf():
    """Train PPO using TensorFlow with Metal acceleration and wandb monitoring"""
    
    # Initialize wandb
    wandb.init(
        project="getup-ppo-tensorflow",
        name=f"tf-metal-ppo-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "framework": "tensorflow",
            "algorithm": "PPO",
            "device": "Metal GPU" if gpu_available else "CPU",
            "environment": "GetupEnv",
        }
    )
    
    LOGGER.info("Starting TensorFlow PPO training with Metal acceleration and wandb monitoring")
    
    # Training parameters
    ppo_params = {
        'num_timesteps': 10_000_000,
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
        'clip_ratio': 0.2,
    }
    
    # Log hyperparameters to wandb
    wandb.config.update(ppo_params)
    
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor()
    gpu_monitor.start_monitoring()
    
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
    
    # Training loop with wandb logging
    num_episodes = 1000
    global_step = 0
    
    for episode in range(num_episodes):
        state = env.reset(rng_seed=episode)
        episode_reward = 0.0
        episode_length = 0
        episode_reward_components = {}
        
        # Collect episode data
        observations = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        
        for step in range(ppo_params['episode_length']):
            # Get action and value
            obs_batch = tf.expand_dims(state.obs, 0)
            action, log_prob = agent.get_action(obs_batch, deterministic=False)
            value = agent.get_value(obs_batch)
            
            action = tf.squeeze(action, 0)
            log_prob = tf.squeeze(log_prob, 0)
            value = tf.squeeze(value, 0)
            
            # Store data
            observations.append(state.obs.numpy())
            actions.append(action.numpy())
            log_probs.append(log_prob.numpy())
            values.append(value.numpy())
            
            # Take step
            state = env.step(state, action)
            rewards.append(state.reward.numpy())
            episode_reward += state.reward.numpy()
            episode_length += 1
            global_step += 1
            
            # Accumulate reward components
            if 'reward_components' in state.metrics:
                for key, value in state.metrics['reward_components'].items():
                    if key not in episode_reward_components:
                        episode_reward_components[key] = []
                    episode_reward_components[key].append(value)
            
            if state.done:
                break
        
        # Compute advantages and returns (simplified GAE)
        returns = []
        advantages = []
        gae = 0
        next_value = 0
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + ppo_params['discounting'] * next_value - values[i]
            gae = delta + ppo_params['discounting'] * 0.95 * gae  # GAE lambda = 0.95
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            next_value = values[i]
        
        # Convert to tensors
        observations = tf.constant(observations)
        actions = tf.constant(actions)
        advantages = tf.constant(advantages)
        returns = tf.constant(returns)
        old_log_probs = tf.constant(log_probs)
        
        # Normalize advantages
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        
        # Update networks
        batch_data = (observations, actions, advantages, returns, old_log_probs)
        training_metrics = agent.update_networks(batch_data)
        
        # Log episode metrics to wandb
        episode_metrics = {
            'episode/reward': episode_reward,
            'episode/length': episode_length,
            'episode/episode_number': episode,
            'training/global_step': global_step,
        }
        
        # Add reward components
        for key, values in episode_reward_components.items():
            episode_metrics[f'rewards/{key}'] = np.mean(values)
        
        # Add training metrics
        if episode % 10 == 0:  # Update training metrics every 10 episodes
            training_metrics_wandb = agent.get_training_metrics()
            episode_metrics.update(training_metrics_wandb)
        
        # Log to wandb
        wandb.log(episode_metrics, step=global_step)
        
        if episode % 100 == 0:
            LOGGER.info(f"Episode {episode}, Reward: {episode_reward:.3f}, Length: {episode_length}")
    
    # Stop GPU monitoring
    gpu_monitor.stop_monitoring()
    
    end_t = datetime.now()
    LOGGER.info(f"Training completed in: {end_t - start_t}")
    
    # Evaluation and rendering
    render_length = 500
    _pre_render_length = 100
    
    rollout = []
    body_height = []
    eval_rewards = []
    
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
        eval_rewards.append(state.reward.numpy())
    
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
    
    # Log final evaluation metrics
    wandb.log({
        'evaluation/height_error': height_error,
        'evaluation/mean_reward': np.mean(eval_rewards),
        'evaluation/final_height': body_height[-1],
    })
    
    # Log the plot to wandb
    wandb.log({"evaluation/height_plot": wandb.Image("tensorflow_metal_height_error.png")})
    
    plt.show()
    
    LOGGER.info(f"Height tracking error: {height_error:.3f}")
    LOGGER.info("Training and evaluation completed successfully!")
    
    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    train_ppo_tf()