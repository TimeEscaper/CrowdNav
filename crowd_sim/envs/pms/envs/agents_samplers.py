import random
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Union
from nip import nip
from scipy.spatial.distance import cdist
from pyminisim.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS
from pyminisim.util import wrap_angle

from crowd_sim.envs.pms.utils.sampling import get_or_sample_bool, get_or_sample_choice, get_or_sample_uniform, \
    random_angle, get_or_sample_int


@dataclass
class AgentsSample:
    n_peds: int
    world_size: Tuple[int, int]
    robot_initial_pose: np.ndarray
    robot_goal: np.ndarray
    ped_initial_poses: Optional[np.ndarray]
    ped_linear_vels: Optional[np.ndarray]
    ped_goals: Optional[np.ndarray]


class AbstractAgentsSampler(ABC):

    def __init__(self, max_peds: int):
        self._max_peds = max_peds

    @property
    def max_peds(self) -> int:
        return self._max_peds

    @abstractmethod
    def sample(self) -> AgentsSample:
        raise NotImplementedError()


@nip
class CompositeAgentsSampler(AbstractAgentsSampler):

    def __init__(self, samplers: List[AbstractAgentsSampler], weights: List[float] = None):
        super(CompositeAgentsSampler, self).__init__(
            max_peds=max([sampler.max_peds for sampler in samplers])
        )
        assert len(samplers) > 0, f"At least one sampler must be specified"
        if weights is not None:
            assert len(weights) == len(samplers), f"Number of weight must be equal to the number of samplers"
            assert np.allclose(sum(weights), 1.), f"Weight must sum up to one"
        self._samplers = tuple(samplers)
        self._indices = np.array(sorted(range(len(samplers))))
        self._weights = np.array(weights) if weights is not None else None

    def sample(self) -> AgentsSample:
        idx = np.random.choice(self._indices, p=self._weights)
        sampler = self._samplers[idx]
        return sampler.sample()


@nip
class RandomAgentsSampler(AbstractAgentsSampler):

    def __init__(self,
                 n_peds: Union[int, Tuple[int, int]],
                 sampling_square: Tuple[int, int] = (20, 20),
                 ped_linear_vels: Union[float, Tuple[float, float]] = 1.5,
                 min_robot_goal_distance: float = 5.,
                 max_sample_trials: int = 1000):
        super(RandomAgentsSampler, self).__init__(
            max_peds=n_peds[1] if isinstance(n_peds, tuple) else n_peds
        )
        assert (isinstance(n_peds, tuple) and len(n_peds) == 2 and n_peds[1] > n_peds[0] >= 0) or (
                    isinstance(n_peds, int) and n_peds >= 0), \
            f"n_peds must be int or tuple (min, max) where max > min >= 0, {n_peds} given"
        self._n_peds = n_peds
        self._sampling_square = sampling_square
        self._ped_linear_vels = ped_linear_vels
        self._min_robot_goal_distance = min_robot_goal_distance
        self._max_sample_trials = max_sample_trials

    def sample(self) -> AgentsSample:
        n_peds = get_or_sample_int(self._n_peds)
        if n_peds > 0:
            return self._sample_jointly(n_peds)
        return self._sample_robot_only()

    def _sample_jointly(self, n_peds: int) -> AgentsSample:
        low = np.array([-self._sampling_square[0] / 2, -self._sampling_square[1] / 2])
        high = np.array([self._sampling_square[0] / 2, self._sampling_square[1] / 2])
        sampled = False
        positions = None
        for _ in range(self._max_sample_trials):
            positions = np.random.uniform(low, high, (n_peds + 1, 2))
            dists = cdist(positions, positions, "euclidean")
            dists = dists[np.triu_indices(dists.shape[0], 1)]
            sampled = (dists > PEDESTRIAN_RADIUS + ROBOT_RADIUS).all()
            if sampled:
                break
        if not sampled:
            raise RuntimeError("Failed to sample positions")
        ped_poses = positions[:-1, :]
        robot_pose = positions[-1, :]

        sampled = False
        goal = None
        for _ in range(self._max_sample_trials):
            goal = np.random.uniform(low, high)
            sampled = np.linalg.norm(robot_pose - goal) >= self._min_robot_goal_distance
            if sampled:
                break
        if not sampled:
            raise RuntimeError("Failed to sample goal")

        ped_poses = np.concatenate((ped_poses, random_angle((n_peds, 1))), axis=-1)
        robot_pose = np.concatenate((robot_pose, random_angle((1,))), axis=-1)

        ped_vels = get_or_sample_uniform(self._ped_linear_vels, n_peds)

        return AgentsSample(n_peds=n_peds,
                            robot_initial_pose=robot_pose,
                            robot_goal=goal,
                            world_size=self._sampling_square,
                            ped_initial_poses=ped_poses,
                            ped_linear_vels=ped_vels,
                            ped_goals=None)

    def _sample_robot_only(self) -> AgentsSample:
        low = np.array([-self._sampling_square[0] / 2, -self._sampling_square[1] / 2])
        high = np.array([self._sampling_square[0] / 2, self._sampling_square[1] / 2])

        robot_pose = np.random.uniform(low, high, (2,))

        sampled = False
        goal = None
        for _ in range(self._max_sample_trials):
            goal = np.random.uniform(low, high)
            sampled = np.linalg.norm(robot_pose - goal) >= self._min_robot_goal_distance
            if sampled:
                break
        if not sampled:
            raise RuntimeError("Failed to sample goal")

        robot_pose = np.array([robot_pose[0], robot_pose[1], np.random.uniform(-np.pi, np.pi)])

        return AgentsSample(n_peds=0,
                            robot_initial_pose=robot_pose,
                            robot_goal=goal,
                            world_size=self._sampling_square,
                            ped_initial_poses=None,
                            ped_linear_vels=None,
                            ped_goals=None)


@nip
class RobotOnlySampler(RandomAgentsSampler):

    def __init__(self,
                 sampling_square: Tuple[int, int] = (20, 20),
                 min_robot_goal_distance: float = 5.,
                 max_sample_trials: int = 1000):
        super(RobotOnlySampler, self).__init__(n_peds=0,
                                               sampling_square=sampling_square,
                                               min_robot_goal_distance=min_robot_goal_distance,
                                               max_sample_trials=max_sample_trials)


@nip
class FixedRobotOnlySampler(AbstractAgentsSampler):

    def __init__(self,
                 robot_pose: Tuple[float, float, float],
                 goal_position: Tuple[float, float]):
        assert len(robot_pose) == 3 and len(goal_position) == 2
        super(FixedRobotOnlySampler, self).__init__(max_peds=0)
        self._robot_pose = np.array(robot_pose)
        self._goal_position = np.array(goal_position)

    def sample(self) -> AgentsSample:
        return AgentsSample(n_peds=0,
                            robot_initial_pose=self._robot_pose.copy(),
                            robot_goal=self._goal_position.copy(),
                            world_size=(20, 20),  # TODO: Remove this parameter in sampling system
                            ped_initial_poses=None,
                            ped_linear_vels=None,
                            ped_goals=None)


@nip
class CircularRobotCentralSampler(AbstractAgentsSampler):

    def __init__(self,
                 n_peds: Union[int, Tuple[int, int]],
                 ped_circle_radius: Union[int, Tuple[float, float]],
                 ped_linear_vels: Union[float, Tuple[float, float]] = 1.5):
        super(CircularRobotCentralSampler, self).__init__(
            max_peds=n_peds[1] if isinstance(n_peds, tuple) else n_peds
        )
        if isinstance(n_peds, tuple):
            assert len(n_peds) == 2 and n_peds[1] > n_peds[0] > 0
        self._n_peds = n_peds
        self._ped_circle_radius = ped_circle_radius
        self._ped_linear_vels = ped_linear_vels

    def sample(self) -> AgentsSample:
        n_peds = get_or_sample_int(self._n_peds)

        angle_step = 2 * np.pi / n_peds
        ped_angles = wrap_angle(np.array([angle_step * i for i in range(n_peds)]) + random_angle())
        ped_goals_angles = wrap_angle(ped_angles + np.pi)
        ped_circle_radius = np.random.uniform(self._ped_circle_radius[0],
                                              self._ped_circle_radius[1]) \
            if isinstance(self._ped_circle_radius, tuple) else self._ped_circle_radius

        ped_poses = np.stack((ped_circle_radius * np.cos(ped_angles),
                              ped_circle_radius * np.sin(ped_angles),
                              random_angle(n_peds)), axis=1)
        ped_goals = np.stack((ped_circle_radius * np.cos(ped_goals_angles),
                              ped_circle_radius * np.sin(ped_goals_angles)), axis=1)
        ped_goals = ped_goals[:, np.newaxis, :]

        robot_pose = np.array([0., 0., random_angle()])
        robot_goal_angle = random_angle()
        robot_goal_radius = np.random.uniform(self._ped_circle_radius[0],
                                              self._ped_circle_radius[1]) \
            if isinstance(self._ped_circle_radius, tuple) else self._ped_circle_radius
        robot_goal = np.array([robot_goal_radius * np.cos(robot_goal_angle),
                               robot_goal_radius * np.sin(robot_goal_angle)])

        ped_vels = get_or_sample_uniform(self._ped_linear_vels, n_peds)

        return AgentsSample(n_peds=n_peds,
                            robot_initial_pose=robot_pose,
                            robot_goal=robot_goal,
                            world_size=(ped_circle_radius, ped_circle_radius),
                            ped_initial_poses=ped_poses,
                            ped_linear_vels=ped_vels,
                            ped_goals=ped_goals)


@nip
class ProxyFixedAgentsSampler(AbstractAgentsSampler):

    def __init__(self, sampler: AbstractAgentsSampler, n_samples: int):
        super(ProxyFixedAgentsSampler, self).__init__(sampler.max_peds)
        self._samples = [sampler.sample() for _ in range(n_samples)]
        self._current_idx = 0

    def sample(self) -> AgentsSample:
        sample = self._samples[self._current_idx]
        self._current_idx = self._current_idx + 1 if self._current_idx < len(self._samples) - 1 else 0
        return sample
