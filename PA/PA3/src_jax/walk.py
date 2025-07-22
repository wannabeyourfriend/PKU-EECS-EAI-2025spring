import os
import pickle
import matplotlib.pyplot as plt

os.environ['JAX_PLATFORMS'] = 'cpu'

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

from typing import Any, Dict

import jax
from jax import numpy as jp
from ml_collections import config_dict
import mujoco
import numpy as np
from mujoco import mjx

from mujoco.mjx._src import math
from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1.joystick import Joystick


DESIRED_BODY_HEIGHT = 0.28
DESIRED_XY_LIN_VEL = np.array([1.0, 0.0])
DESIRED_YAW_ANG_VEL = 0.0

class MyWalkEnv(Joystick):
    def step(self, state: mjx_env.State, action: jax.Array):
        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        contact = jp.array(
            [
                collision.geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

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

        # TODO: your code here. hint: use DESIRED_XY_LIN_VEL and DESIRED_YAW_ANG_VEL as goal
        
        # === SOTA Reward Design for Quadruped Walking ===
        
        # 1. Linear Velocity Tracking (Exponential reward with tolerance)
        # Based on ANYmal and other SOTA quadruped controllers
        lin_vel_error = jp.linalg.norm(body_lin_vel[:2] - DESIRED_XY_LIN_VEL)
        tracking_lin_vel = jp.exp(-2.0 * lin_vel_error**2)  # Gaussian reward with sigma=0.5
        
        # 2. Angular Velocity Tracking (Yaw rate tracking)
        # Focus on yaw angular velocity (z-axis rotation)
        yaw_vel_error = jp.abs(body_ang_vel[2] - DESIRED_YAW_ANG_VEL)
        tracking_ang_vel = jp.exp(-1.5 * yaw_vel_error**2)  # Gaussian reward with sigma=0.58
        
        # === Additional SOTA Reward Components ===
        
        # 3. Forward Progress Reward (encourages forward motion)
        forward_progress = jp.maximum(0.0, body_lin_vel[0])  # Only reward forward motion
        
        # 4. Lateral Stability (penalize excessive lateral motion)
        lateral_stability = jp.exp(-5.0 * body_lin_vel[1]**2)
        
        # 5. Smooth Velocity Changes (penalize abrupt velocity changes)
        if "last_body_vel" in state.info:
            vel_smoothness = jp.exp(-0.5 * jp.linalg.norm(body_lin_vel - state.info["last_body_vel"])**2)
        else:
            vel_smoothness = 1.0
        
        # 6. Energy Efficiency (reward efficient gaits)
        # Penalize high joint velocities when not necessary
        joint_vel_penalty = jp.exp(-0.01 * jp.sum(joint_qvel**2))
        
        # 7. Gait Regularity (encourage periodic foot contacts)
        # This promotes natural quadruped gaits
        contact_symmetry = 1.0 - 0.5 * jp.abs(jp.sum(contact[:2]) - jp.sum(contact[2:]))
        
        # 8. Base Stability (minimize roll and pitch)
        # Extract roll and pitch from gravity vector
        roll_pitch_error = jp.linalg.norm(gravity_vector[:2])
        base_stability = jp.exp(-3.0 * roll_pitch_error**2)

        # TODO: End of your code.
        info = state.info
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        contact = jp.array(
            [
                collision.geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        reward_orientation = jp.sum(jp.square(self.get_upvector(data)[:2]))
        reward_height = jp.exp(-3 * (DESIRED_BODY_HEIGHT - body_pos[2]) ** 2)
        rew_termination= self._cost_termination(done)
        rew_pose =  self._reward_pose(data.qpos[7:])
        rew_torques = self._cost_torques(data.actuator_force)
        rew_action_rate =  self._cost_action_rate(action, info["last_act"], info["last_last_act"])

        rew_energy = self._cost_energy(data.qvel[6:], data.actuator_force)
        rew_feet_slip = self._cost_feet_slip(data, contact, info)
        rew_feet_clearance = self._cost_feet_clearance(data)
        rew_feet_height = self._cost_feet_height(
            state.info["swing_peak"], first_contact, info
        )
        rew_feet_air_time = self._reward_feet_air_time(
            state.info["feet_air_time"], first_contact, jp.hstack([DESIRED_XY_LIN_VEL, DESIRED_YAW_ANG_VEL])
        )
        rew_dof = self._cost_joint_pos_limits(data.qpos[7:])

        # Bookkeeping - SOTA Reward Composition
        reward = (
            # Primary tracking objectives (highest weights)
            2.0 * tracking_lin_vel          # Linear velocity tracking
            + 1.0 * tracking_ang_vel        # Angular velocity tracking
            
            # Locomotion quality rewards
            + 0.5 * forward_progress         # Encourage forward motion
            + 0.3 * lateral_stability        # Lateral stability
            + 0.2 * vel_smoothness          # Smooth velocity changes
            + 0.15 * joint_vel_penalty      # Energy efficiency
            + 0.1 * contact_symmetry        # Gait regularity
            + 0.4 * base_stability          # Base stability
            
            # Original stability and safety rewards (adjusted weights)
            + 0.15 * reward_height          # Body height maintenance
            + -3.0 * reward_orientation     # Orientation penalty (reduced)
            + -1.0 * rew_termination        # Termination penalty
            + 0.3 * rew_pose               # Pose reward (reduced)
            + -0.0001 * rew_torques        # Torque penalty (reduced)
            + -0.005 * rew_action_rate     # Action rate penalty (reduced)
            + -0.0005 * rew_energy         # Energy penalty (reduced)
            + -0.05 * rew_feet_slip        # Feet slip penalty (reduced)
            + -1.0 * rew_feet_clearance    # Feet clearance penalty (reduced)
            + -0.1 * rew_feet_height       # Feet height penalty (reduced)
            + 0.05 * rew_feet_air_time     # Feet air time reward (reduced)
            + -0.5 * rew_dof               # DOF penalty (reduced)
        )
        reward = jp.clip(reward * self.dt, 0., 10000.0)
        
        # Store current velocity for next step
        state.info["last_body_vel"] = body_lin_vel
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action

        done = jp.float32(done)
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["steps_until_next_cmd"] -= 1
        state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)

        state.metrics["reward/sum"] = reward
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # d(xyzrpy)=U(-0.5, 0.5)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])


        rng, key1, key2 = jax.random.split(rng, 3)
        time_until_next_cmd = jax.random.exponential(key1) * 5.0
        steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(jp.int32)
        cmd = jax.random.uniform(
            key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
        )

        info = {
            "rng": rng,
            "command": cmd,
            "steps_until_next_cmd": steps_until_next_cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": jp.zeros(3),
            "last_body_vel": jp.zeros(3),  # Initialize last_body_vel to maintain pytree structure
        }

        metrics = {}
        metrics["reward/sum"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
        gyro = self.get_gyro(data)

        gravity = self.get_gravity(data)

        joint_angles = data.qpos[7:]

        joint_vel = data.qvel[6:]

        linvel = self.get_local_linvel(data)

        state = jp.hstack(
            [
                linvel,  # 3
                gyro,  # 3
                gravity,  # 3
                joint_angles - self._default_pose,  # 12
                joint_vel,  # 12
                info["last_act"],  # 12
            ]
        )

        accelerometer = self.get_accelerometer(data)
        angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

        privileged_state = jp.hstack(
            [
                state,
                gyro,  # 3
                accelerometer,  # 3
                gravity,  # 3
                linvel,  # 3
                angvel,  # 3
                joint_angles - self._default_pose,  # 12
                joint_vel,  # 12
                data.actuator_force,  # 12
                info["last_contact"],  # 4
                feet_vel,  # 4*3
                info["feet_air_time"],  # 4
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

def create_env():
    # env config
    env_cfg = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        Kp=35.0,
        Kd=0.5,
        episode_length=500,
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
        reward_config=config_dict.create(
            tracking_sigma=0.25,
            max_foot_height=0.1,
        ),
        command_config=config_dict.create(
            # Uniform distribution for command amplitude.
            a=[1.5, 0.8, 1.2],
            # Probability of not zeroing out new command.
            b=[0.9, 0.25, 0.5],
        ),
    )
    env = MyWalkEnv(config=env_cfg)
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

def train_ppo(checkpoint_dir='./checkpoints', resume_from_checkpoint=None, save_interval=10_000_000):
    import mediapy as media

    from datetime import datetime
    import functools
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    from ml_collections import config_dict

    ppo_params = config_dict.create(
        num_timesteps=6_000_000, #200_000_000,
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

    render_length = 1000

    # Enable perturbation in the eval env.
    eval_env = create_env()

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(0)
    rollout = []
    body_lin_vel = []

    state = jit_reset(rng)
    for i in range(render_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)

        state = jit_step(state, ctrl)
        rollout.append(state)

        linvel = env.get_local_linvel(state.data)
        body_lin_vel.append(linvel[0])

    body_lin_vel = jp.array(body_lin_vel)
    lin_vel_error = np.mean(np.abs(body_lin_vel - DESIRED_XY_LIN_VEL[0]))
    plt.plot(body_lin_vel)
    # plot desired body height
    plt.axhline(DESIRED_XY_LIN_VEL[0], color='r', linestyle='--')
    plt.title(f"LinVel error: {lin_vel_error:.3f}")
    plt.xlabel("steps")
    plt.ylabel("body linear velocity")
    plt.savefig("part3_LinVel_error.png")

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
    media.write_video('../experiments/solutions/part3_video_100_000_000.mp4', frames)
    print("video saved to part3.mp4")
    
    return make_inference_fn, params, metrics

def evaluate_from_checkpoint(checkpoint_path, render_video=True):
    """Load a checkpoint and evaluate the trained model."""
    params, metrics, step = load_checkpoint(checkpoint_path)
    if params is None:
        LOGGER.error("Failed to load checkpoint for evaluation")
        return
    
    LOGGER.info(f"Evaluating model from step {step}")
    
    # Create environment and inference function
    env = create_env()
    
    # We need to recreate the network to get the inference function
    from brax.training.agents.ppo import networks as ppo_networks
    import functools
    
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="privileged_state",
        value_obs_key="privileged_state",
    )
    
    # Create dummy environment to get observation spec
    dummy_env = create_env()
    obs_size = dummy_env.observation_size
    action_size = dummy_env.action_size
    
    # Create networks
    networks = network_factory(obs_size, action_size, preprocess_observations_fn=lambda x, y: x)
    make_inference_fn = ppo_networks.make_inference_fn(networks)
    
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(0)
    rollout = []
    body_lin_vel = []

    state = jit_reset(rng)
    render_length = 1000
    
    for i in range(render_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)

        state = jit_step(state, ctrl)
        rollout.append(state)

        linvel = env.get_local_linvel(state.data)
        body_lin_vel.append(linvel[0])

    body_lin_vel = jp.array(body_lin_vel)
    lin_vel_error = np.mean(np.abs(body_lin_vel - DESIRED_XY_LIN_VEL[0]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(body_lin_vel)
    plt.axhline(DESIRED_XY_LIN_VEL[0], color='r', linestyle='--')
    plt.title(f"LinVel error: {lin_vel_error:.3f} (Step {step})")
    plt.xlabel("steps")
    plt.ylabel("body linear velocity")
    plt.savefig(f"evaluation_step_{step}_LinVel_error.png")
    plt.show()
    
    if render_video:
        import mediapy as media
        
        render_every = 2
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
        media.write_video(f'evaluation_step_{step}_video.mp4', frames)
        LOGGER.info(f"Evaluation video saved: evaluation_step_{step}_video.mp4")
    
    return lin_vel_error

if __name__ == '__main__':
    # Example usage:
    
    # Train from scratch
    train_ppo(checkpoint_dir='./checkpoints', save_interval=1_000_000)
    
    # Resume from latest checkpoint
    # train_ppo(checkpoint_dir='./checkpoints', resume_from_checkpoint='latest')
    
    # Resume from specific checkpoint
    # train_ppo(checkpoint_dir='./checkpoints', resume_from_checkpoint='./checkpoints/checkpoint_50000000.pkl')
    
    # Evaluate from checkpoint
    # evaluate_from_checkpoint('./checkpoints/checkpoint_latest.pkl')