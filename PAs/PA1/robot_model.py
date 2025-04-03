import os
from typing import List
import numpy as np
import xml.etree.ElementTree as ET

from urdf_types import Link, Joint, FixedJoint, RevoluteJoint
from config import RobotConfig
from rotation import rpy_to_mat
from utils import str_to_np
from vis import Vis


class RobotModel:
    robot_cfg: RobotConfig
    links: List[Link]
    joints: List[Joint]

    def __init__(self, robot_cfg: RobotConfig):
        """
        Initialize the RobotModel with the given RobotConfig

        Parameters
        ----------
        robot_cfg : RobotConfig
            The configuration of the robot
        """
        self.robot_cfg = robot_cfg
        self.load_urdf(robot_cfg)

    def fk(self, qpos: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for all of the links.

        Here we assume the i-th joint has i-th link as parent
        and i+1-th link as child.

        In practice multiple joints can have a shared parent link.
        But dealing with those special cases is not required in the homework.

        The result is in robot frame, which means that the first link
        has (0, 0, 0) as translation and I as rotation matrix

        Here we assume each link's frame, except the first one,
        is as same as its parent joint's frame.

        See urdf_types.py for the definition of Link and Joint.

        Parameters
        ----------
        qpos: np.ndarray
            The current joint angles with shape (J,)
            (which means its length is the number of revolute joints)

        Returns
        -------
        np.ndarray
            The poses of links with shape (L, 4, 4)
            (which means its length is the number of links)

        Note
        ----
        The 4*4 pose matrix combines translation and rotation.

        Its format is:
            R  t
            0  1
        where R is rotation matrix and t is translation vector
        """
        # 初始化所有连杆的位姿矩阵
        num_links = len(self.links)
        poses = np.zeros((num_links, 4, 4))
        
        # 设置第一个连杆的位姿为单位矩阵（位于原点，无旋转）
        poses[0, :, :] = np.eye(4)
        
        # 旋转关节计数器（用于索引qpos数组）
        revolute_joint_idx = 0
        
        # 计算每个连杆的位姿
        for i in range(1, num_links):
            # 获取当前连杆的父关节（即第i-1个关节）
            joint = self.joints[i-1]
            
            # 获取父连杆的位姿
            parent_pose = poses[i-1]
            
            # 创建关节的变换矩阵
            joint_transform = np.eye(4)
            
            # 设置关节的旋转和平移
            joint_transform[:3, :3] = joint.rot
            joint_transform[:3, 3] = joint.trans
            
            # 如果是旋转关节，需要应用关节角度
            if isinstance(joint, RevoluteJoint):
                # 获取当前关节角度
                angle = qpos[revolute_joint_idx]
                revolute_joint_idx += 1
                
                # 计算旋转轴的旋转矩阵
                axis = joint.axis
                # 创建旋转矩阵（罗德里格斯公式）
                cos_theta = np.cos(angle)
                sin_theta = np.sin(angle)
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)
                
                # 应用关节角度旋转
                rot_matrix = np.eye(4)
                rot_matrix[:3, :3] = R
                joint_transform = joint_transform @ rot_matrix
            
            # 计算当前连杆的位姿 = 父连杆位姿 * 关节变换
            poses[i] = parent_pose @ joint_transform
            
        return poses

    def load_urdf(self, robot_cfg: RobotConfig):
        """
        Load the URDF into this RobotModel

        Theoretically one can write a general code that load
        everything only from the URDF, but it will make the
        code too complex. Thus we read the joints' name and
        links' name from the RobotConfig instead.

        Parameters
        ----------
        robot_cfg : RobotConfig
            The configuration of the robot
        """
        self.links = [None for _ in robot_cfg.link_names]
        self.joints = [None for _ in robot_cfg.joint_names]
        tree = ET.parse(robot_cfg.urdf_path)
        root = tree.getroot()
        for child in root:
            if child.tag == "link":
                idx = robot_cfg.link_names.index(child.attrib["name"])
                self.links[idx] = Link(
                    name=child.attrib["name"],
                    visual_meshes=[
                        os.path.join(
                            os.path.dirname(robot_cfg.urdf_path), m.attrib["filename"]
                        )
                        for m in child.findall("./visual/geometry/mesh")
                    ],
                )
            elif child.tag == "joint":
                idx = robot_cfg.joint_names.index(child.attrib["name"])
                joint_type = child.attrib["type"]
                kwargs = dict(
                    name=child.attrib["name"],
                    trans=str_to_np(child.find("origin").attrib["xyz"]),
                    rot=rpy_to_mat(str_to_np(child.find("origin").attrib["rpy"])),
                )
                if joint_type == "fixed":
                    self.joints[idx] = FixedJoint(**kwargs)
                elif joint_type == "revolute":
                    self.joints[idx] = RevoluteJoint(
                        axis=str_to_np(child.find("axis").attrib["xyz"]),
                        lower_limit=float(child.find("limit").attrib["lower"]),
                        upper_limit=float(child.find("limit").attrib["upper"]),
                        **kwargs
                    )

    def vis(self, poses: np.ndarray, color: str) -> list:
        """
        A helper function to visualize the fk result with plotly.

        You can modify it for debugging

        Parameters
        ----------
        poses: np.ndarray
            The poses of each link with shape (L, 4, 4)

        color: str (or any other format supported by plotly)
            The color of the meshes shown in visualization

        Returns
        -------
        A list of plotly objects that can be shown in Vis.show
        """
        vis_list = []
        for l, p in zip(self.links, poses):
            vis_list += Vis.pose(p[:3, 3], p[:3, :3])
            for m in l.visual_meshes:
                vis_list += Vis.mesh(path=m, trans=p[:3, 3], rot=p[:3, :3], color=color)
        return vis_list


if __name__ == "__main__":
    # a simple test to check if the code is working
    # you can modify it to test your code
    from config import get_robot_config

    cfg = get_robot_config("galbot")
    robot_model = RobotModel(cfg)

    gt = np.load(os.path.join("data", "fk.npz"))
    idx = 0
    q = gt["q"][idx]
    gt_poses = gt["poses"][idx]

    # the correct answer is the green one
    gt_vis = robot_model.vis(gt_poses, color="lightgreen")

    my_poses = robot_model.fk(q * 0)
    # your answer is the brown one
    my_vis = robot_model.vis(my_poses, color="brown")

    # it will be shown in the browser
    # if not, you can input a html path to this function and open it manually
    # if a new page is shown but nothing is displayed, you can try refreshing the page
    Vis.show(gt_vis + my_vis, path=None)
