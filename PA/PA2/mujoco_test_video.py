import argparse
import os
import platform
import time
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
from tqdm import trange, tqdm  

from src.type import Grasp
from src.sim.grasp_env import Obs, GraspEnvConfig, GraspEnv, get_grasps
from src.sim.cfg import MjRenderConfig

if platform.system() == "Darwin":
    os.environ['MUJOCO_GL'] = 'osmesa'
    print("macOS detected: Using osmesa for off-screen rendering")
else:
    os.environ['MUJOCO_GL'] = 'osmesa'
    print("Using osmesa for off-screen rendering")


class VideoRecorder:
    
    def __init__(self, output_path: str, fps: int = 30, resolution: tuple = (640, 480)):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.frames: List[np.ndarray] = []
        self.writer = None
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
    def add_frame(self, frame: np.ndarray):
        if frame.shape[:2] != self.resolution[::-1]:
            frame = cv2.resize(frame, self.resolution)
        
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self.frames.append(frame.copy())
    
    def save_video(self):
        if not self.frames:
            print("No frames to save")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.fps, 
            self.resolution
        )
        
        print(f"Saving video with {len(self.frames)} frames to {self.output_path}")
        
        for frame in tqdm(self.frames, desc="Writing video"):
            self.writer.write(frame)
            
        self.writer.release()
        print(f"Video saved successfully: {self.output_path}")
    
    def clear(self):
        self.frames.clear()
        if self.writer:
            self.writer.release()
            self.writer = None


class GraspEnvWithVideo(GraspEnv):
    
    def __init__(self, config: GraspEnvConfig, video_recorder: Optional[VideoRecorder] = None, 
                 use_global_camera: bool = False, camera_config: dict = None):
        super().__init__(config)
        self.video_recorder = video_recorder
        self.frame_count = 0
        self.use_global_camera = use_global_camera
        self.camera_config = camera_config or {}
        
    def step_with_recording(self, action):
        self.sim.step(action)
        
        if self.video_recorder:
            self._record_frame()
            
    def _record_frame(self):
        try:
            if self.use_global_camera:
                # 使用全局视角相机
                obs = self.get_global_obs()
            else:
                # 使用原始的手腕相机
                obs = self.get_obs()
            
            if obs.rgb is not None:
                self.video_recorder.add_frame(obs.rgb)
                self.frame_count += 1
        except Exception as e:
            print(f"Error recording frame: {e}")
    
    def get_global_obs(self) -> Obs:
        global_render_cfg = MjRenderConfig(
            height=self.camera_config.get('height', 480),
            width=self.camera_config.get('width', 640),
            lookat=np.array(self.camera_config.get('lookat', [0.5, 0.0, 0.5])),  # 看向桌子中心
            distance=self.camera_config.get('distance', 2.0),  # 相机距离
            azimuth=np.deg2rad(self.camera_config.get('azimuth', 45)),  # 方位角
            elevation=np.deg2rad(self.camera_config.get('elevation', -30)),  # 仰角
            fovy=self.camera_config.get('fovy', 45)  # 视野角度
        )
        
        x = self.sim.render(global_render_cfg)
        
        obj_ids = self.sim.get_seg_id_list(self.obj.name)
        seg = x["seg"][..., 0]
        obj_seg = np.zeros_like(seg).astype(np.uint8)
        for i in range(len(obj_ids)):
            obj_seg[seg == obj_ids[i]] = 255
        
        from src.utils import to_pose
        obj_pose = to_pose(*self.sim.get_body_pose(self.obj.geom_id))
        
        camera_pose = np.eye(4)
        
        obs = Obs(
            rgb=x["rgb"],
            depth=x["depth"],
            seg=obj_seg,
            camera_pose=camera_pose,
            object_pose=obj_pose,
        )
        return obs
    
    def execute_plan_with_recording(self, traj: np.ndarray) -> bool:
        obj_init_z = self.sim.get_body_pose(self.obj.geom_id)[0][2]
        self.sim.reset(traj[0])
        
        if self.video_recorder:
            self._record_frame()
        
        for _ in range(3):
            self.sim.step(traj[0][:8])
            if self.video_recorder:
                self._record_frame()
        
        for i, qpos in enumerate(traj):
            self.sim.step(qpos[:8])
            if self.video_recorder:
                self._record_frame()
            
            time.sleep(0.01)
        
        for _ in range(10):
            self.sim.step(traj[-1][:8])
            if self.video_recorder:
                self._record_frame()
        
        obj_final_z = self.sim.get_body_pose(self.obj.geom_id)[0][2]
        return obj_final_z - obj_init_z > self.config.succ_height_thresh


def main():
    parser = argparse.ArgumentParser(description="Trajectory Evaluation with Video Recording")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.1)
    parser.add_argument("--headless", type=int, default=1)  # 默认使用headless
    parser.add_argument("--wait_steps", type=int, default=25)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--grasp", type=int, default=0)
    
    parser.add_argument("--record_video", type=int, default=1, help="是否录制视频 (0/1)")
    parser.add_argument("--video_fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--video_resolution", type=str, default="640x480", help="视频分辨率 (WxH)")
    parser.add_argument("--output_dir", type=str, default="videos", help="视频输出目录")
    
    parser.add_argument("--use_global_camera", type=int, default=1, help="使用全局视角相机 (0/1)")
    parser.add_argument("--camera_distance", type=float, default=2.0, help="相机距离")
    parser.add_argument("--camera_azimuth", type=float, default=45, help="相机方位角 (度)")
    parser.add_argument("--camera_elevation", type=float, default=-30, help="相机仰角 (度)")
    parser.add_argument("--camera_lookat", type=str, default="0.5,0.0,0.5", help="相机看向的点 (x,y,z)")
    parser.add_argument("--camera_fovy", type=float, default=45, help="相机视野角度")
    
    args = parser.parse_args()
    
    args.headless = 1
    
    if 'x' in args.video_resolution:
        width, height = map(int, args.video_resolution.split('x'))
        video_resolution = (width, height)
    else:
        video_resolution = (640, 480)
    
    # 解析相机看向的点
    lookat = list(map(float, args.camera_lookat.split(',')))
    
    # 相机配置
    camera_config = {
        'height': video_resolution[1],
        'width': video_resolution[0],
        'distance': args.camera_distance,
        'azimuth': args.camera_azimuth,
        'elevation': args.camera_elevation,
        'lookat': lookat,
        'fovy': args.camera_fovy
    }
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for run_idx in trange(args.num):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        video_recorder = None
        if args.record_video:
            camera_type = "global" if args.use_global_camera else "wrist"
            video_filename = f"grasp_{args.robot}_{args.obj}_{camera_type}_{timestamp}_run{run_idx}.mp4"
            video_path = output_dir / video_filename
            video_recorder = VideoRecorder(
                str(video_path), 
                fps=args.video_fps, 
                resolution=video_resolution
            )
            print(f"Will record video to: {video_path}")
            print(f"Camera config: distance={args.camera_distance}, azimuth={args.camera_azimuth}°, elevation={args.camera_elevation}°")
        
        env_config = GraspEnvConfig(
            robot=args.robot,
            obj_name=args.obj,
            headless=bool(args.headless),
            ctrl_dt=args.ctrl_dt,
            wait_steps=args.wait_steps,
        )
        
        env = GraspEnvWithVideo(
            env_config, 
            video_recorder, 
            use_global_camera=bool(args.use_global_camera),
            camera_config=camera_config
        )
        env.launch()
        env.reset()
        
        obs = env.get_obs()
        env.save_obs(obs)
        if video_recorder:
            env._record_frame()
        
        if not args.grasp:
            if video_recorder:
                for _ in range(30): 
                    env._record_frame()
                    time.sleep(0.033)
            
            env.close()
            if video_recorder:
                video_recorder.save_video()
            continue
        
        grasps = get_grasps(args.obj)
        plan = None
        
        for obj_frame_grasp in grasps:
            robot_frame_grasp = Grasp(
                trans=obs.object_pose[:3, :3] @ obj_frame_grasp.trans
                + obs.object_pose[:3, 3],
                rot=obs.object_pose[:3, :3] @ obj_frame_grasp.rot,
                width=obj_frame_grasp.width,
            )
            plan = env.plan_grasp(robot_frame_grasp, obs.robot_frame_pc)
            if plan is not None:
                break
        
        if plan is not None:
            print("Executing grasp plan...")
            succ = env.execute_plan_with_recording(plan)
            print(f"Execution {'succeeded' if succ else 'failed'}")
        else:
            print("No plan found")
            if video_recorder:
                for _ in range(60):
                    env._record_frame()
                    time.sleep(0.033)
        
        env.close()
        
        if video_recorder:
            video_recorder.save_video()
            print(f"Recorded {env.frame_count} frames")


if __name__ == "__main__":
    main()