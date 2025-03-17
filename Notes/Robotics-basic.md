# Embodied Intelligence Introduction

> Notes taking: 王子轩
>
> email: `wang-zx23@mails.tsinghua.edu.cn`
>
> Lecture Intructor: `He Wang`

[TOC]

## Kinematics 

### **Rigid Transformation ** 

>Describing the motion of bodies (positionand velocity). Kinematics does not consider how to achieve motion via force.

DoF: degree of freedom自由度

>使用$(R_{s \rightarrow b}, \mathbf{t}_{s \rightarrow b})$二元组来描述**Rigid Transformation**

![](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250224155811798.png)

如上图所示，我们使用$\mathcal{F}_{s}$来表示坐标，如图有
$$
o_b^s = o_s^s + \mathbf{t}_{s \rightarrow b }^s\\
[x_b^s,\cdots] = R_{s \rightarrow b}^s[x_s^s, \cdots]
$$
那么得到在$s$和$b$两个坐标系中的关系：$p^s = R^s_{s \rightarrow b}p^b + \mathbf{t}^s_{s \rightarrow b}$

变换是非线性的：
$$
p_2^s = R_{s\rightarrow b}^s p_2^b + t_{s\rightarrow b}^s\\
p_2^s = R_{s\rightarrow b}^s p_2^b + t_{s\rightarrow b}^s\\
p_1^s + p_2^s \neq R_{s\rightarrow b}^s(p_1^b + p_2^b) + t_{s\rightarrow b}^s \quad \text{when} \quad  t_{s\rightarrow b}^s \neq \mathbf{0}\\
$$
但是，考虑到并行计算，我们希望能够把坐标变换写成矩阵的形式；

**Homogeneous coordinate**
$$
\hat{x} = [x, 1]^T \in \mathbb{R}^4\\
$$
**Homogeneous transformation matrix**
$$
T^s_{s \rightarrow b} = \begin{bmatrix} 
R^s_{s \rightarrow b} & \mathbf{t}_{s \rightarrow b }^s \\
1 &  0\\
\end{bmatrix}
$$
Coordinate transformation under **linear form:**
$$
\hat{x}^s = T^s_{s \rightarrow b} \hat{x}^b
$$
作为一种通用的记法，我们可以将他们写成：
$$
\hat{x}^1 = T^1_{1 \rightarrow 2} \hat{x}^2
$$
两坐标系之间的变化是矩阵求逆的关系：
$$
T^2_{2 \rightarrow 1} = (T^1_{1 \rightarrow 2})^{-1}
$$

### **多关节刚体几何学 Multi-Link Rigid-Body Geometry **

> Terms: 
>
> **Links** are the rigid-body connected in sequence
>
> **Joints** are the connectors between links. They determine the DoF of motion between adjacent links
>
> | base                                                         | link1                                                        | link2                                                        | end_effectror                                                |
> | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
> | ![Image 1](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250224163254971.png) | ![Image 2](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250224163156899.png) | ![Image 3](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250224163202993.png) | ![image-20250224163208029](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250224163208029.png) |

$$

$$







![image-20250224163348606](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250224163348606.png)

### **Kinematics: 2 end2effector methods**

- **Joint Space (Configuration Space)**

  > The space in which each coordinate is a vector of joint poses (angles around joint axis).

- **Cartesian Space (Task Space)**

  >Cartesian space: The space of the rigid transformations of the end-effector by , where is the end-effector frame.

| Aspect           | Joint Space                                                  | Cartesian Space                                              |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Definition       | Describes robot configuration in terms of joint variables.   | Describes the position and orientation of the end-effector in the physical space. |
| Representation   | A vector of joint angles or displacements (e.g., ([q1, q2, q_3])). | A vector for position and possibly a matrix for orientation (e.g., ([x, y, z]) and a rotation matrix). |
| Space Type       | Abstract, does not directly represent physical space.        | Represents the physical space in which the robot operates.   |
| Control          | Used to control individual joints.                           | Used to control the movement of the end-effector.            |
| Motion           | Joint motion (e.g., move each joint to a certain angle).     | End-effector motion (e.g., move the tool to a specific position and orientation). |
| Complexity       | Can be simpler for computation and planning in certain cases. | Can be more complex, as it involves transforming between joint and Cartesian spaces. |
| Transformations: |                                                              |                                                              |

- **前向运动学 Forward Kinematics** 

> Map the joint space coordinate$\theta \in \mathbb{R}^n$ to a transformation matrix $T$:
> $$
> T_{s \rightarrow e} = f(\theta)
> $$

求解方法：直接Calculated by composing transformations along the kinematic chain.



- **逆向动力学** 

IK问题的解：

- 少数情况下有解析解
- 多数情况采用数值方法，利用梯度下降对energy function进行优化，例如$\text{argmin} |T(\theta) - T_{target}|$



现有的机械臂自由度通常为6或者7。



Given the forward kinematics and the target pose , find solutions that satisfy Ts→e(θ) T_target = 𝕊𝔼(3) 的θ
满足Ts→e(θ) = T_target

**无解和奇点问题**：其中一个解决的思路是加一个关节,cuRobo





使用VLA的**representation**：预测 $\Delta \theta $    v.s. $\Delta R \quad \text{and} \quad \Delta t$



## Rotation

欧拉角



衡量两个旋转矩阵之间的距离
$$
(R_2 R_1^T)R_1 = R_2, \text{distance} = \text{tr}(R_2 R_1^T) - 1
$$




### Robot 2

$$
\mathbb{SO}(n) = \{R \in R^{n \times n}:\text{det}(R) = 1, RR^{T} = 1 \}
$$



旋转物体不改变物体的手性（Chirality），所以旋转矩阵$\text{det} = +1$、

- **Cloud-Edge-Marginal**: Cloud-edge-end computing power layout, balancing latency and energy consumption.

- **The Issues of Rotation Matrix as Rotation Representation**: Computational efficiency vs. numerical stability

  - Rotation -> 3 DoF (Degrees of Freedom), but the matrix has 9 values

  - Normalization: How to normalize a rotation matrix -> Schmidt orthogonalization

欧拉角：

**Roll-Yaw-Pitch** representation 旋转不具备交换性，顺序影响最终结果；万向界死锁现象：

![image-20250303153930366](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250303153930366.png)

- 另外一种方向的表征

  ![image-20250303154125132](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250303154125132.png)

存储一个$[w, \theta]$

对于旋转矩阵进行正交化后，根据不同的正交化方法，可以得到不同的矩阵：如何找到最优的矩阵？通过投影，将非旋转矩阵投影，计算L2-norm，需要做SVD分解，再计算$\theta(R_1^TR_2)$

Quaternion在计算上具有简便性，ROS\PhysX\PyBullet,用的是(x, y, z, w)的表征





### Interpolation

路径规划问题中，需要在控制周期内找到$R(t)$,不能直接进行线性插值

- 思路1：R1->R2, 可以使用axis angle进行间接的旋转插值
- 思路2：对于quaternion直接在球面上进行插值，弧上的插值。

SLERP技术：
$$
v(t) = \frac{\text{sin} \beta}{\text{sin} (\alpha + \beta)} v_0 + \frac{\text{sin} \alpha}{\text{sin} (\alpha + \beta)}v_1
$$
Sampling 

球面均匀采样：不可以对$\theta, \phi$均匀采样，两级会比较集中，球面的均匀采样需要计算分布函数，并且取逆

对Euler的均匀采样会导致biased dataset

creating dataset：using axis angle/qu

不同的rotation representation对于neural network的train的影响：

微小的ground truth的旋转，但是Eular angle、angle-axis以及quaternion会有不连续的问题，但神经网络对应的是连续函数族，因此rotation matrix对应的是最好的表征，考虑一个损失函数$||R' - R_{3 \times 3}||$

On the continuaty of Rotation representation

## Motion Planning











eg：机械臂移动刚体，有搬运的task：$(R, T) \rightarrow (R', T')$

传统机器人学对于motion planning的formulation

![motion-planning](assets/clipboard-image-1742196580.png)





![image-20250303175146398](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250303175146398.png)

![](E:\project\EAI\Notes\assets\image-20250303175154472-1742198215327-2.png)

`qpos`represent：$(\theta_1, \theta_2, \cdots)$

高维的寻路问题->维度灾难

example：collision modeling of robot

如何表征link的物理表面：

- 1. 使用mesh跑从q的模拟
- 2. 采用球真包裹表面，考虑球与球之间的碰撞检测![image-20250317155642977](E:\project\EAI\Notes\assets\image-20250317155642977.png)

vision-mesh和collision-mesh的区分，（凸分解）















### Grasping

#### Definition: 

> Grasping is the process of restraining an object’s motion in a desired way by applying forces and torques at a set of contacts

nonprehensile-prehensile 可否抓握

Grasp Synthesis is a **high-dimensional search or optimization problem** to find gripper poses or joint configurations.

![](assets/Robot_Vision_and_Learning_Lecture05_03_17.jpeg)



**Terminology**

- Grasp pose：the position and orientation of a hand

  - 4-DoF grasp: a 3D position and 1D hand orientation aligned with the direction of gravity, a.k.a. “top-down grasping
  - 6-DoF grasp: 3D position and 3D orientation
  - Finger DoFs:  TheParallel gripper can provide a degree of freedom independently.
  - Dex Hand: up to 21 

  - 7 DoF grasp

**Open-Loop Grasping**



![NaVid](https://pku-epic.github.io/D3RoMa/static/images/method.png)

- Work: 

抓一次看一次，没有形成闭环 Tabletop Grasping

GraspVLA

![动图](https://pic4.zhimg.com/v2-5ad8043b58536f1b1f71e0d9bf956d39_b.webp)



**Two Paths for Open-Loop Grasping**

- Known objects:
  - Pose Estimation
- unknown and general objects
  - directly predict grasping pose



6D Object  Pose:

When the camera's intrinsic parameters are known, in the absence of symmetry, a single RGB can fully and uniquely predict the 6D pose of an object.

![image-20250317155609453](E:\project\EAI\Notes\assets\image-20250317155609453.png)







#### Rotation Regression

