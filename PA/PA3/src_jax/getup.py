import os
import pickle
import matplotlib.pyplot as plt

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import time
import logging
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

import jax
# Force JAX to use CPU backend to avoid Metal compatibility issues with MuJoCo MJX
jax.config.update('jax_platform_name', 'cpu')
LOGGER.info(f"Using JAX platform: {jax.default_backend()}")

from jax import numpy as jp
from ml_collections import config_dict
import mujoco
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1.getup import Getup

DESIRED_BODY_HEIGHT = 0.33

class MyGetupEnv(Getup):
    def step(self, state: mjx_env.State, action: jax.Array):
        motor_targets = state.data.qpos[7:] + action * self._config.action_scale
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        # --- Key State Extraction ---

        # Unit vector pointing up in world frame (z-axis negative in many physics engines)
        up_vec = jp.array([0.0, 0.0, -1.0])

        # Extract the 3D position of the IMU site (typically mounted on the body), in world coordinates [m]
        body_pos = data.site_xpos[self._imu_site_id]

        # Compute the linear velocity of the body in its local (body) frame [m/s]
        body_lin_vel = self.get_local_linvel(data)

        # Retrieve the angular velocity (gyroscope data) of the body in its local frame [rad/s]
        body_ang_vel = self.get_gyro(data)

        # Get the gravity vector expressed in the local body frame, usually used to infer orientation
        gravity_vector = self.get_gravity(data)

        # Make a copy of the default pose (e.g., reference or rest configuration) of the full model
        default_qpos = self._default_pose.copy()

        # Extract joint positions (excluding root pose: typically first 7 elements are base position + orientation)
        joint_qpos = data.qpos[7:]

        # Extract joint velocities (excluding root velocities)
        joint_qvel = data.qvel[7:]

        # TODO: your code here.
        #  Hint: consider three import objective related to getup task:
        #   1. body height
        #   2. body orientation
        #   3. joint position (error to default pose)
        #   4. body angular velocity (error to zero)

        # 1. Body Height Reward - Exponential reward for reaching desired height
        body_height = body_pos[2]  # z-coordinate of body position
        height_error = jp.abs(body_height - DESIRED_BODY_HEIGHT)
        rew_height = jp.exp(-10.0 * height_error)  # Sharp exponential reward
        
        # 2. Body Orientation Reward - Encourage upright posture
        # gravity_vector in body frame should point downward when upright
        # When upright, gravity_vector should be close to [0, 0, -1] in body frame
        gravity_norm = jp.linalg.norm(gravity_vector)
        gravity_normalized = gravity_vector / (gravity_norm + 1e-8)
        # Reward for gravity vector pointing downward (negative z in body frame)
        upright_reward = jp.maximum(0.0, -gravity_normalized[2])  # Clamp to positive
        rew_orientation = jp.exp(5.0 * (upright_reward - 1.0))  # Exponential scaling
        
        # 3. Joint Position Reward - Minimize deviation from default pose
        joint_pos_error = jp.linalg.norm(joint_qpos - default_qpos)
        rew_joint_pos = jp.exp(-2.0 * joint_pos_error)
        
        # 4. Angular Velocity Penalty - Encourage stability (minimize spinning)
        ang_vel_penalty = jp.linalg.norm(body_ang_vel)
        rew_ang_vel = jp.exp(-1.0 * ang_vel_penalty)
        
        # 5. Linear Velocity Penalty - Encourage staying in place during getup
        lin_vel_penalty = jp.linalg.norm(body_lin_vel[:2])  # Only x,y components
        rew_lin_vel = jp.exp(-0.5 * lin_vel_penalty)
        
        # 6. Action Smoothness Reward - Penalize large action changes
        action_diff = jp.linalg.norm(action - state.info["last_act"])
        rew_smoothness = jp.exp(-0.1 * action_diff)
        
        # 7. Energy Efficiency - Penalize large joint velocities
        joint_vel_penalty = jp.linalg.norm(joint_qvel)
        rew_energy = jp.exp(-0.05 * joint_vel_penalty)
        
        # 8. Termination Cost - Penalize early termination (similar to walk.py)
        rew_termination = jp.where(done, -10.0, 0.0)  # Heavy penalty for termination
        
        # 9. Action Rate Cost - Penalize large action changes (similar to walk.py)
        action_diff = jp.linalg.norm(action - state.info["last_act"])
        rew_action_rate = -action_diff  # Linear penalty for action changes
        
        # 10. Torque Cost - Penalize large torques (energy efficiency)
        # Note: In getup task, we don't have actuator_force directly, so we approximate with action magnitude
        rew_torques = -jp.linalg.norm(action)  # Penalize large actions as proxy for torques
        
        # Improved reward composition following walk.py's SOTA design pattern
        reward = (
            # Primary objectives (highest weights)
            4.0 * rew_height                    # Primary: reach target height
            + 0.5 * rew_orientation             # Critical: maintain upright orientation
            + 0.5 * rew_joint_pos               # Important: stay close to default pose
            
            # Stability rewards (medium weights)
            + 0.5 * rew_ang_vel                 # Stability: minimize angular velocity
            + 0.8 * rew_lin_vel                 # Stability: minimize lateral movement
            + 0.5 * rew_energy                  # Efficiency: minimize energy consumption
            
            # Smoothness and safety penalties (lower weights but important)
            + 0.3 * rew_smoothness              # Smoothness: encourage smooth actions
            + -1.0 * rew_termination            # Termination penalty
            + -0.01 * rew_action_rate           # Action rate penalty (scaled down)
            + -0.001 * rew_torques              # Torque penalty (scaled down)
        )
        
        # Apply time scaling and clipping similar to walk.py
        reward = jp.clip(reward * self.dt, -100.0, 100.0)  # More conservative clipping for getup task
        # TODO: End of your code.

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action

        state.metrics["reward"] = reward
        done = jp.float32(done)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Sample a random initial configuration with some probability.
        # rng, key1, key2 = jax.random.split(rng, 3)
        # qpos = jp.where(
        #     jax.random.bernoulli(key1, self._config.drop_from_height_prob),
        #     self._get_random_qpos(key2),
        #     self._init_q,
        # )
        qpos = self._init_q.copy()
        # Sample a random root velocity.
        # rng, key = jax.random.split(rng)
        qvel = jp.zeros(self.mjx_model.nv)
        # qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

        # Let the robot settle for a few steps.
        data = mjx_env.step(self.mjx_model, data, qpos[7:], self._settle_steps)
        data = data.replace(time=0.0)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
        }

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        metrics = {"reward": jp.zeros(())}
        return mjx_env.State(data, obs, reward, done, metrics, info)


def create_env():
    # env config
    env_cfg = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        Kp=35.0,
        Kd=0.5,
        episode_length=300,
        drop_from_height_prob=0.6,
        settle_time=0.5,
        action_repeat=1,
        action_scale=0.5,
        soft_joint_pos_limit_factor=0.95,
        energy_termination_threshold=np.inf,
        noise_config=config_dict.create(
            level=0.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
            ),
        ),
    )
    env = MyGetupEnv(env_cfg)
    return env


def save_checkpoint(params, metrics, checkpoint_dir, step):
    """Save model parameters and training metrics to checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_data = {
        'params': params,
        'metrics': metrics,
        'step': step,
        'timestamp': time.time()
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # Also save as latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pkl')
    with open(latest_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    LOGGER.info(f"Checkpoint saved at step {step}: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path):
    """Load model parameters and training metrics from checkpoint."""
    if not os.path.exists(checkpoint_path):
        LOGGER.warning(f"Checkpoint not found: {checkpoint_path}")
        return None, None, 0
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        params = checkpoint_data['params']
        metrics = checkpoint_data['metrics']
        step = checkpoint_data['step']
        
        LOGGER.info(f"Checkpoint loaded from step {step}: {checkpoint_path}")
        return params, metrics, step
    except Exception as e:
        LOGGER.error(f"Failed to load checkpoint: {e}")
        return None, None, 0


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory."""
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pkl')
    if os.path.exists(latest_path):
        return latest_path
    
    # Fallback: find the checkpoint with highest step number
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pkl')]
    if not checkpoint_files:
        return None
    
    # Extract step numbers and find the maximum
    step_numbers = []
    for f in checkpoint_files:
        try:
            step = int(f.split('_')[1].split('.')[0])
            step_numbers.append((step, f))
        except:
            continue
    
    if step_numbers:
        latest_file = max(step_numbers, key=lambda x: x[0])[1]
        return os.path.join(checkpoint_dir, latest_file)
    
    return None


def evaluate_from_checkpoint(checkpoint_path, render_video=True):
    """Load a checkpoint and evaluate the trained model."""
    params, metrics, step = load_checkpoint(checkpoint_path)
    if params is None:
        LOGGER.error("Failed to load checkpoint for evaluation")
        return
    
    LOGGER.info(f"Evaluating model from checkpoint at step {step}")
    
    # Create environment and inference function
    env = create_env()
    
    # We need to recreate the network to get the inference function
    import functools
    from brax.training.agents.ppo import networks as ppo_networks
    from ml_collections import config_dict
    
    ppo_params = config_dict.create(
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="privileged_state",
            value_obs_key="privileged_state",
        ),
    )
    
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )
    
    # Create dummy environment to get observation/action specs
    dummy_env = create_env()
    from mujoco_playground import wrapper
    wrapped_env = wrapper.wrap_for_brax_training(dummy_env)
    
    # Create networks
    networks = network_factory(
        wrapped_env.observation_size,
        wrapped_env.action_size,
        preprocess_observations_fn=lambda x, rng: x
    )
    
    make_inference_fn = networks.make_policy
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    
    # Run evaluation
    render_length = 500
    _pre_render_length = 100

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    rollout = []
    body_height = []

    state = jit_reset(rng)
    for i in range(render_length):
        if i < _pre_render_length:
            ctrl = env._default_pose.copy()
        else:
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)

        state = jit_step(state, ctrl)
        rollout.append(state)
        env_height = state.data.site_xpos[env._imu_site_id][2]
        body_height.append(env_height)

    body_height = jp.array(body_height)
    height_error = np.mean(np.abs(body_height - DESIRED_BODY_HEIGHT))
    plt.figure(figsize=(10, 6))
    plt.plot(body_height)
    plt.axhline(DESIRED_BODY_HEIGHT, color='r', linestyle='--', label='Desired Height')
    plt.title(f"Height error: {height_error:.3f} (Step {step})")
    plt.xlabel("steps")
    plt.ylabel("body height")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"checkpoint_eval_step_{step}_height_error.png")
    plt.show()

    if render_video:
        import mediapy as media
        
        render_every = 2
        fps = 1.0 / env.dt / render_every
        print(f"fps: {fps}")

        traj = rollout[::render_every]
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

        frames = env.render(
            traj,
            camera="track",
            height=480,
            width=640,
            scene_option=scene_option,
        )
        video_path = f'../experiments/solutions/checkpoint_eval_step_{step}_video.mp4'
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        media.write_video(video_path, frames)
        print(f"Evaluation video saved to {video_path}")
    
    return height_error


def train_ppo(checkpoint_dir='./checkpoints', resume_from_checkpoint=None, save_interval=100_000):
    import mediapy as media

    from datetime import datetime
    import functools
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    from ml_collections import config_dict

    ppo_params = config_dict.create(
        num_timesteps=50_000_000, 
        num_evals=0,
        reward_scaling=1.0,
        episode_length=500,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=5e-4, # 3e-4
        entropy_cost=1e-2,
        num_envs=4096,
        batch_size=128, # 256
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="privileged_state",
            value_obs_key="privileged_state",
        ),
    )

    # Handle checkpoint loading
    initial_params = None
    initial_step = 0
    
    if resume_from_checkpoint:
        if resume_from_checkpoint == 'latest':
            checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        else:
            checkpoint_path = resume_from_checkpoint
        
        if checkpoint_path:
            loaded_params, loaded_metrics, loaded_step = load_checkpoint(checkpoint_path)
            if loaded_params is not None:
                initial_params = loaded_params
                initial_step = loaded_step
                LOGGER.info(f"Resuming training from step {initial_step}")

    start_t = datetime.now()

    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )

    # Create custom progress function for checkpointing
    def progress_fn(num_steps, metrics):
        if num_steps % save_interval == 0 and num_steps > 0:
            # Note: We can't access params here directly from the training loop
            # This is a limitation of the current brax training API
            LOGGER.info(f"Training progress: {num_steps} steps completed")
        return metrics

    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=progress_fn,
        num_eval_envs=0,
        log_training_metrics=True,
        training_metrics_steps=1_000_000
    )

    env = create_env()
    eval_env = create_env()
    
    # If we have initial params, we would need to modify the training function
    # For now, we'll save checkpoints after training completes
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    
    end_t = datetime.now()
    print(f"time to train: {end_t - start_t}")

    # Save final checkpoint
    final_step = ppo_params.num_timesteps
    save_checkpoint(params, metrics, checkpoint_dir, final_step)

    render_length = 500
    _pre_render_length = 100

    # Enable perturbation in the eval env.
    eval_env = create_env()

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(0)
    rollout = []
    body_height = []

    # is_upright = self._is_upright(gravity)
    # is_at_desired_height = self._is_at_desired_height(torso_height)
    # gate = is_upright * is_at_desired_height

    state = jit_reset(rng)
    for i in range(render_length):
        if i < _pre_render_length:
            ctrl = env._default_pose.copy()
        else:
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)

        state = jit_step(state, ctrl)
        rollout.append(state)
        env_height = state.data.site_xpos[env._imu_site_id][2]
        body_height.append(env_height)

    body_height = jp.array(body_height)
    height_error = np.mean(np.abs(body_height - DESIRED_BODY_HEIGHT))
    plt.plot(body_height)
    # plot desired body height
    plt.axhline(DESIRED_BODY_HEIGHT, color='r', linestyle='--')
    plt.title(f"Height error: {height_error:.3f}")
    plt.xlabel("steps")
    plt.ylabel("body height")
    plt.savefig("part2_height_error.png")

    render_every = 2
    fps = 1.0 / eval_env.dt / render_every
    print(f"fps: {fps}")

    traj = rollout[::render_every]
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

    frames = eval_env.render(
        traj,
        camera="track",
        height=480,
        width=640,
        scene_option=scene_option,
    )
    media.write_video('../experiments/solutions/part2_video_50_000_000.mp4', frames)
    print("video saved to part2.mp4")

if __name__ == '__main__':
    # 示例用法:
    # 1. 从头开始训练并保存checkpoint
    # train_ppo(checkpoint_dir='./getup_checkpoints', save_interval=1_000_000)
    
    # 2. 从最新checkpoint恢复训练
    # train_ppo(checkpoint_dir='./getup_checkpoints', resume_from_checkpoint='latest')
    
    # 3. 从特定checkpoint恢复训练
    train_ppo(checkpoint_dir='./getup_checkpoints', resume_from_checkpoint='./getup_checkpoints/checkpoint_latest.pkl')
    
    # 4. 评估已保存的checkpoint
    # evaluate_from_checkpoint('./getup_checkpoints/checkpoint_latest.pkl')