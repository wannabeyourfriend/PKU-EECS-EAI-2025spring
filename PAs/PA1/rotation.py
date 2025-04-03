import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize the quaternion.

    Parameters
    ----------
    q: np.ndarray
        Unnormalized quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        Normalized quaternion with shape (4,)
    """
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])  # 返回单位四元数(相当于没有旋转)
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Return the conjugate of the quaternion.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The conjugate of the quaternion with shape (4,)
    """
    # 四元数共轭:保持实部不变,虚数向量部分取反
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply the two quaternions.
    实现四元数的乘法

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)

    Returns
    -------
    np.ndarray
        The multiplication result with shape (4,)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Use quaternion to rotate a 3D vector.
    最核心的功能,将向量v按照四元数q的旋转进行旋转
    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)
    v: np.ndarray
        Vector with shape (3,)

    Returns
    -------
    np.ndarray
        The rotated vector with shape (3,)
    """
    # 确保四元数是单位四元数
    q = quat_normalize(q)
    
    # 将向量转换为纯四元数 (0, v)
    v_quat = np.array([0, v[0], v[1], v[2]])
    
    # 执行 q * v * q^(-1) 旋转,其中q^(-1)表示q的共轭
    q_conj = quat_conjugate(q)
    rotated = quat_multiply(quat_multiply(q, v_quat), q_conj)
    
    # 返回旋转后的向量
    return rotated[1:]


def quat_relative_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute the relative rotation angle between the two quaternions.
    计算两个四元数之间的相对旋转角度，用来表征两个旋转之间的差异
    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)

    Returns
    -------
    float
        The relative rotation angle in radians, greater than or equal to 0.
    """
    # 确保输入是单位四元数
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    
    # 计算相对四元数 q_rel = q2 * q1^(-1)
    q_rel = quat_multiply(q2, quat_conjugate(q1))
    
    # 从相对四元数中提取角度
    cos_theta = np.clip(q_rel[0], -1.0, 1.0)
    angle = 2 * np.arccos(cos_theta)
    
    # 确保角度在 [0, π] 范围内
    if angle > np.pi:
        angle = 2 * np.pi - angle
    
    return angle


def interpolate_quat(q1: np.ndarray, q2: np.ndarray, ratio: float) -> np.ndarray:
    """
    Interpolate between two quaternions with given ratio.

    When the ratio is 0, return q1; when the ratio is 1, return q2.

    The interpolation should be done in the shortest minor arc connecting the quaternions on the unit sphere.

    If there are multiple correct answers, you can output any of them.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)
    ratio: float
        The ratio of interpolation, should be in [0, 1]

    Returns
    -------
    np.ndarray
        The interpolated quaternion with shape (4,)

    Note
    ----
    What should be done if the inner product of the quaternions is negative?
    """
    # 确保输入是单位四元数
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    
    # 计算四元数的点积
    dot = np.sum(q1 * q2)
    
    # 如果点积为负,取其中一个四元数的负值
    # 这确保我们沿着最短路径进行插值
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # 如果四元数几乎相同,直接线性插值并归一化
    if dot > 0.9995:
        result = q1 + ratio * (q2 - q1)
        return quat_normalize(result)
    
    # 执行球面线性插值 (SLERP)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    theta = theta_0 * ratio
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """
    Convert the quaternion to rotation matrix.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    # 确保输入是单位四元数
    q = quat_normalize(q)
    w, x, y, z = q
    
    # 构建旋转矩阵
    R = np.zeros((3, 3))
    
    # 填充旋转矩阵的元素
    R[0, 0] = 1 - 2 * (y**2 + z**2)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)
    
    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x**2 + z**2)
    R[1, 2] = 2 * (y * z - w * x)
    
    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x**2 + y**2)
    
    return R


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """
    Convert the rotation matrix to quaternion.

    Parameters
    ----------
    mat: np.ndarray
        The rotation matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        The quaternion with shape (4,)
    """
    # 使用Shepperd的方法从旋转矩阵提取四元数
    trace = np.trace(mat)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (mat[2, 1] - mat[1, 2]) * s
        y = (mat[0, 2] - mat[2, 0]) * s
        z = (mat[1, 0] - mat[0, 1]) * s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s
    
    return quat_normalize(np.array([w, x, y, z]))


def mat_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    """
    Convert the rotation matrix to axis-angle representation.

    The length of the axis-angle vector should be less or equal to pi.

    Parameters
    ----------
    mat: np.ndarray
        The rotation matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        The axis-angle representation with shape (3,)
    """
    # 首先将旋转矩阵转换为四元数
    q = mat_to_quat(mat)
    
    # 然后将四元数转换为轴角表示
    return quat_to_axis_angle(q)


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """
    Convert the quaternion to axis-angle representation.

    The length of the axis-angle vector should be less or equal to pi.

    If there are multiple answers, you can output any.

    Parameters
    ----------
    q: np.ndarray
        The quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The axis-angle representation with shape (3,)
    """
    # 确保输入是单位四元数
    q = quat_normalize(q)
    w, x, y, z = q
    
    # 计算旋转角度
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    
    # 如果角度接近0,返回零向量
    if np.abs(angle) < 1e-10:
        return np.zeros(3)
    
    # 计算旋转轴
    sin_half_angle = np.sqrt(1.0 - w * w)
    if np.abs(sin_half_angle) < 1e-10:
        axis = np.array([1.0, 0.0, 0.0])  # 任意轴，无旋转
    else:
        axis = np.array([x, y, z]) / sin_half_angle
    
    # 确保角度不超过π
    if angle > np.pi:
        angle = 2 * np.pi - angle
        axis = -axis
    
    # 返回轴角表示
    return axis * angle


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """
    Convert the axis-angle representation to quaternion.

    The length of the axis-angle vector should be le    ss or equal to pi

    Parameters
    ----------
    aa: np.ndarray
        The axis-angle representation with shape (3,)

    Returns
    -------
    np.ndarray
        The quaternion with shape (4,)
    """
    # 计算旋转角度(轴角向量的长度)
    angle = np.linalg.norm(aa)
    
    # 如果角度接近0,返回单位四元数
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    # 计算旋转轴
    axis = aa / angle
    
    # 计算四元数分量
    half_angle = angle / 2.0
    sin_half_angle = np.sin(half_angle)
    
    w = np.cos(half_angle)
    x = axis[0] * sin_half_angle
    y = axis[1] * sin_half_angle
    z = axis[2] * sin_half_angle
    
    return np.array([w, x, y, z])


def axis_angle_to_mat(aa: np.ndarray) -> np.ndarray:
    """
    Convert the axis-angle representation to rotation matrix.

    Parameters
    ----------
    aa: np.ndarray
        The axis-angle representation with shape (3,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    # 首先将轴角表示转换为四元数
    q = axis_angle_to_quat(aa)
    
    # 然后将四元数转换为旋转矩阵
    return quat_to_mat(q)


def uniform_random_quat() -> np.ndarray:
    """
    Generate a random quaternion with uniform distribution.

    Returns
    -------
    np.ndarray
        The random quaternion with shape (4,)
    """
    # 使用Marsaglia方法生成均匀分布的随机四元数
    u1, u2, u3 = np.random.random(3)
    
    # 计算四元数分量
    sqrt_u1 = np.sqrt(u1)
    sqrt_1_u1 = np.sqrt(1 - u1)
    
    theta1 = 2 * np.pi * u2
    theta2 = 2 * np.pi * u3
    
    w = sqrt_1_u1 * np.sin(theta1)
    x = sqrt_1_u1 * np.cos(theta1)
    y = sqrt_u1 * np.sin(theta2)
    z = sqrt_u1 * np.cos(theta2)
    
    return np.array([w, x, y, z])


def rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    """
    Convert roll-pitch-yaw euler angles into rotation matrix.

    This is required since URDF use this as rotation representation.

    Parameters
    ----------
    rpy: np.ndarray
        The euler angles with shape (3,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    roll, pitch, yaw = rpy

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = R_z @ R_y @ R_x  # Matrix multiplication in ZYX order
    return R
