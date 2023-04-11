import numpy as np

from typing import Union
from pyminisim.util import wrap_angle


def unnormalize(action: np.ndarray,
                lb: np.ndarray,
                ub: np.ndarray) -> np.ndarray:
    deviation = (ub - lb) / 2.
    shift = (ub + lb) / 2.
    action = (action * deviation) + shift
    return action


def normalize_asymmetric(value: Union[float, np.ndarray],
                         lb: Union[float, np.ndarray],
                         ub: Union[float, np.ndarray]):
    return (value - lb) / (ub - lb)


def normalize_symmetric(value: Union[float, np.ndarray],
                        lb: Union[float, np.ndarray],
                        ub: Union[float, np.ndarray]):
    ratio = 2. / (ub - lb)
    shift = (ub + lb) / 2.
    return (value - shift) * ratio


def subgoal_to_global(subgoal_polar: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
    x_rel_rot = subgoal_polar[0] * np.cos(subgoal_polar[1])
    y_rel_rot = subgoal_polar[0] * np.sin(subgoal_polar[1])
    theta = robot_pose[2]
    x_rel = x_rel_rot * np.cos(theta) - y_rel_rot * np.sin(theta)
    y_rel = x_rel_rot * np.sin(theta) + y_rel_rot * np.cos(theta)
    x_abs = x_rel + robot_pose[0]
    y_abs = y_rel + robot_pose[1]
    return np.array([x_abs, y_abs])


def transition_to_subgoal_polar(pose_initial: np.ndarray, pose_new: np.ndarray) -> np.ndarray:
    # TODO: Check
    linear = np.linalg.norm(pose_initial[:2] - pose_new[:2])
    angular = np.arctan2(pose_new[1] - pose_initial[1], pose_new[0] - pose_initial[0])
    angular = wrap_angle(angular - pose_initial[2])
    return np.array([linear, angular])
