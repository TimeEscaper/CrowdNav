from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Optional, Tuple

import gym
import numpy as np
from nip import nip

from crowd_sim.envs.pms.utils.math import unnormalize
from crowd_sim.envs.pms.utils.sampling import get_or_sample_uniform, get_or_sample_bool, get_or_sample_choice


class AbstractActionSpaceConfig(ABC):
    TYPE_END2END = "end2end"
    TYPE_SUBGOAL = "subgoal"

    def __init__(self, action_space: gym.spaces.Space, action_type: str):
        action_types = [AbstractActionSpaceConfig.TYPE_END2END, AbstractActionSpaceConfig.TYPE_SUBGOAL]
        assert action_type in action_types, f"action_type must be in {action_types}, {action_type} is given"
        self._action_space = action_space
        self._action_type = action_type

    @abstractmethod
    def get_action(self, policy_action: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def action_type(self) -> str:
        return self._action_type


class ContinuousActionSpace(AbstractActionSpaceConfig):

    def __init__(self,
                 lb: np.ndarray,
                 ub: np.ndarray,
                 normalize: bool,
                 action_type: str):
        assert lb.shape == ub.shape and len(lb.shape) == 1, f"ub and lb must be equal shape 1D arrays, " \
                                                            f"got {lb.shape} and {ub.shape}"
        shape = (lb.shape[0],)
        if normalize:
            action_space = gym.spaces.Box(low=-np.ones_like(lb),
                                          high=np.ones_like(ub),
                                          shape=shape,
                                          dtype=np.float32)
        else:
            action_space = gym.spaces.Box(low=lb.copy(),
                                          high=ub.copy(),
                                          shape=shape,
                                          dtype=np.float32)
        super(ContinuousActionSpace, self).__init__(action_space, action_type)
        self._lb = np.array(lb)
        self._ub = np.array(ub)
        self._normalize = normalize

    def get_action(self, policy_action: np.ndarray) -> np.ndarray:
        action = np.clip(policy_action, self.action_space.low, self.action_space.high)
        if self._normalize:
            action = unnormalize(action, self._lb, self._ub)
        return action


class MultiDiscreteActionSpace(AbstractActionSpaceConfig):

    def __init__(self,
                 lb: np.ndarray,
                 ub: np.ndarray,
                 grid: np.ndarray,
                 action_type: str):
        assert lb.shape == ub.shape and len(lb.shape) == 1, f"ub and lb must be equal shape 1D arrays, " \
                                                            f"got {lb.shape} and {ub.shape}"
        assert grid.shape == lb.shape, f"Shape of the grid must be equal to the shape of " \
                                       f"bounds {lb.shape}, {grid.shape} is given"
        action_space = gym.spaces.MultiDiscrete(grid.tolist())
        super(MultiDiscreteActionSpace, self).__init__(action_space, action_type)
        self._grid = [np.linspace(lb[i], ub[i], grid[i]) for i in range(grid.shape[0])]

    def get_action(self, policy_action: np.ndarray) -> np.ndarray:
        return np.array([self._grid[i][policy_action[i]] for i in range(policy_action.shape[0])])


@nip
class ContinuousUnicycleActionSpace(ContinuousActionSpace):

    def __init__(self,
                 lb: Tuple[float, float],
                 ub: Tuple[float, float],
                 normalize: bool):
        assert len(lb) == 2 and len(ub) == 2, f"Size of the lb and ub must be 2, {len(lb)} and {len(ub)} are given"
        super(ContinuousUnicycleActionSpace, self).__init__(lb=np.array(lb),
                                                            ub=np.array(ub),
                                                            normalize=normalize,
                                                            action_type=AbstractActionSpaceConfig.TYPE_END2END)


@nip
class MultiDiscreteUnicycleActionSpace(MultiDiscreteActionSpace):

    def __init__(self,
                 lb: Tuple[float, float],
                 ub: Tuple[float, float],
                 n_linear: int,
                 n_angular: int):
        assert len(lb) == 2 and len(ub) == 2, f"Size of the lb and ub must be 2, {len(lb)} and {len(ub)} are given"
        assert n_linear >= 2 and n_angular >= 2 and isinstance(n_linear, int) and isinstance(n_angular, int), \
            f"n_linear and n_angular must be integers and >= 2"
        super(MultiDiscreteUnicycleActionSpace, self).__init__(lb=np.array(lb),
                                                               ub=np.array(ub),
                                                               grid=np.array([n_linear, n_angular]),
                                                               action_type=AbstractActionSpaceConfig.TYPE_END2END)


@nip
class ContinuousPolarSubgoalActionSpace(ContinuousActionSpace):

    def __init__(self,
                 lb: Tuple[float, float],
                 ub: Tuple[float, float],
                 normalize: bool):
        assert len(lb) == 2 and len(ub) == 2, f"Size of the lb and ub must be 2, {len(lb)} and {len(ub)} are given"
        super(ContinuousPolarSubgoalActionSpace, self).__init__(lb=np.array(lb),
                                                                ub=np.array(ub),
                                                                normalize=normalize,
                                                                action_type=AbstractActionSpaceConfig.TYPE_SUBGOAL)


@nip
class MultiPolarSubgoalActionSpace(MultiDiscreteActionSpace):

    def __init__(self,
                 lb: Tuple[float, float],
                 ub: Tuple[float, float],
                 n_radial: int,
                 n_angular: int):
        assert len(lb) == 2 and len(ub) == 2, f"Size of the lb and ub must be 2, {len(lb)} and {len(ub)} are given"
        assert n_radial >= 2 and n_angular >= 2 and isinstance(n_radial, int) and isinstance(n_angular, int), \
            f"n_radial and n_angular must be integers and >= 2"
        super(MultiPolarSubgoalActionSpace, self).__init__(lb=np.array(lb),
                                                           ub=np.array(ub),
                                                           grid=np.array([n_radial, n_angular]),
                                                           action_type=AbstractActionSpaceConfig.TYPE_SUBGOAL)


@dataclass
@nip
class ProblemConfig:
    ped_model: str
    robot_visible: bool
    detector_range: float
    detector_fov: float
    goal_reach_threshold: float
    max_steps: int
    subgoal_reach_threshold: Optional[float] = None
    max_subgoal_steps: Optional[int] = None


@dataclass
@nip
class SimConfig:
    sim_dt: float = 0.01
    policy_dt: float = 0.1
    rt_factor: Optional[float] = None
    render: bool = False


class AbstractProblemConfigSampler(ABC):

    @abstractmethod
    def sample(self) -> ProblemConfig:
        raise NotImplementedError()


@nip
class RandomProblemSampler(AbstractProblemConfigSampler):

    def __init__(self,
                 ped_model: Union[str, Tuple[str, ...]],
                 robot_visible: Union[bool, str] = False,
                 detector_range: Union[float, Tuple[float, float]] = 5,
                 detector_fov: Union[float, Tuple[float, float]] = 360.,
                 goal_reach_threshold: float = 0.1,
                 max_steps: int = 300,
                 subgoal_reach_threshold: Optional[float] = 0.1,
                 max_subgoal_steps: Optional[float] = None):
        if isinstance(robot_visible, str):
            assert robot_visible == "random", f"Only 'random' string is allowed, {robot_visible} is given"
        super(RandomProblemSampler, self).__init__()
        self._ped_model = ped_model
        self._robot_visible = robot_visible
        self._detector_range = detector_range
        self._detector_fov = detector_fov
        self._goal_reach_threshold = goal_reach_threshold
        self._max_steps = max_steps
        self._subgoal_reach_threshold = subgoal_reach_threshold
        self._max_subgoal_steps = max_subgoal_steps

    def sample(self) -> ProblemConfig:
        return ProblemConfig(
            ped_model=get_or_sample_choice(self._ped_model),
            robot_visible=get_or_sample_bool(self._robot_visible),
            detector_range=get_or_sample_uniform(self._detector_range),
            detector_fov=get_or_sample_uniform(self._detector_fov),
            goal_reach_threshold=self._goal_reach_threshold,
            max_steps=self._max_steps,
            subgoal_reach_threshold=self._subgoal_reach_threshold,
            max_subgoal_steps=self._max_subgoal_steps
        )


@nip
class ProxyFixedProblemSampler(AbstractProblemConfigSampler):

    def __init__(self, sampler: AbstractProblemConfigSampler, n_samples: int):
        super(ProxyFixedProblemSampler, self).__init__()
        self._configs = [sampler.sample() for _ in range(n_samples)]
        self._current_idx = 0

    def sample(self) -> ProblemConfig:
        config = self._configs[self._current_idx]
        self._current_idx = self._current_idx + 1 if self._current_idx < len(self._configs) - 1 else 0
        return config
