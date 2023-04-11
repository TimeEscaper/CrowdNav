import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, List
from nip import nip
from pyminisim.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS


class RewardContext:

    def __init__(self, attributes: Optional[Dict[str, Any]] = None):
        self._attributes = attributes or {}

    def has_attribute(self, attribute: str) -> bool:
        return attribute in self._attributes

    def get(self, attribute: str) -> Optional[Any]:
        if attribute in self._attributes:
            return self._attributes[attribute]
        return None

    def set(self, attribute: str, value: Any) -> None:
        self._attributes[attribute] = value


class AbstractReward(ABC):

    @abstractmethod
    def __call__(self, context: RewardContext) -> Tuple[float, Dict[str, float]]:
        raise NotImplementedError()


@nip
class CompositeReward(AbstractReward):

    def __init__(self, rewards: List[AbstractReward], weights: Optional[Tuple[float]] = None):
        super(CompositeReward, self).__init__()
        if weights is not None:
            assert len(rewards) == len(weights), "Number of weights must be same as number of rewards"
        else:
            weights = [1. for _ in range(len(rewards))]
        self._rewards = rewards
        self._weights = weights

    def __call__(self, context: RewardContext) -> Tuple[float, Dict[str, float]]:
        total_reward = 0.
        total_info = {}
        for reward_fn, weight in zip(self._rewards, self._weights):
            reward, info = reward_fn(context)
            total_reward += weight * reward
            total_info.update(info)
        return total_reward, total_info


@nip
class BranchReward(AbstractReward):

    def __init__(self,
                 step_reward: Union[AbstractReward, float],
                 success_reward: Union[AbstractReward, float],
                 fail_reward: Union[AbstractReward, float],
                 truncated_is_fail: bool = False):
        self._step_reward = step_reward
        self._success_reward = success_reward
        self._fail_reward = fail_reward
        self._truncated_is_fail = truncated_is_fail

    def __call__(self, context: RewardContext) -> Tuple[float, Dict[str, float]]:
        collision = context.get("collision") or False
        truncated = context.get("truncated") or False
        if collision or (truncated and self._truncated_is_fail):
            return BranchReward._return_reward(self._fail_reward, context)
        success = context.get("success") or False
        if success:
            return BranchReward._return_reward(self._success_reward, context)
        return BranchReward._return_reward(self._step_reward, context)

    @staticmethod
    def _return_reward(reward: Union[AbstractReward, float],
                       context: RewardContext) -> Tuple[float, Dict[str, float]]:
        if isinstance(reward, AbstractReward):
            reward, info = reward(context)
            info["branch"] = reward
            return reward, info
        return reward, {"branch": reward}


@nip
class PotentialGoalReward(AbstractReward):

    def __init__(self, coefficient: float = 2.):
        self._coefficient = coefficient

    def __call__(self, context: RewardContext) -> Tuple[float, Dict[str, float]]:
        pose = context.get("robot_pose")[:2]
        prev_pose = context.get("previous_robot_pose")[:2]
        goal = context.get("goal")[:2]
        d_t = np.linalg.norm(pose - goal)
        d_t_prev = np.linalg.norm(prev_pose - goal)
        reward = self._coefficient * (-d_t + d_t_prev)
        return reward, {"goal_potential": reward}


@nip
class AngularVelocityPenalty(AbstractReward):

    def __init__(self, coefficient: float = 0.005):
        assert coefficient >= 0., "Coefficient must be positive (minus sign will be added automatically in the class)"
        self._coefficient = -coefficient

    def __call__(self, context: RewardContext) -> Tuple[float, Dict[str, float]]:
        reward = self._coefficient * abs(context.get("robot_velocity")[2])
        return reward, {"angular_velocity_penalty": reward}


@nip
class BasicPredictionPenalty(AbstractReward):

    def __init__(self, factor: float = 20., mode: str = "previous"):
        assert factor >= 0., "Factor must be positive (minus sign will be added automatically in the class)"
        assert mode in ["previous", "next"], f"'previous' and 'next' modes available, {mode} is given"
        self._factor = -factor
        self._context_key = "previous_ped_predictions" if mode == "previous" else "next_ped_predictions"

    def __call__(self, context: RewardContext) -> Tuple[float, Dict[str, float]]:
        robot_position = context.get("robot_pose")[:2]
        predictions = context.get(self._context_key)

        reward = 0.
        if len(predictions) != 0:
            min_reward = np.inf
            has_collision = False
            for ped_predictions in predictions.values():
                distances = np.linalg.norm(robot_position - ped_predictions[0], axis=1)
                collision_mask = distances <= ROBOT_RADIUS + PEDESTRIAN_RADIUS
                if collision_mask.any():
                    has_collision = True
                    collision_idx = np.argmax(collision_mask)
                    penalty = self._factor / (2. ** (collision_idx + 1))
                    if penalty < min_reward:
                        min_reward = penalty

            if has_collision:
                reward = min_reward

        return reward, {"prediction_penalty": reward}


@nip
class DiscomfortPenalty(AbstractReward):

    def __init__(self, coefficient: float = 1., discomfort_dist: float = 0.3):
        assert coefficient >= 0., "Coefficient must be positive"
        assert discomfort_dist > 0., f"Discomfort distance must be > 0., {discomfort_dist} is given"
        self._coefficient = coefficient
        self._discomfort_dist = discomfort_dist

    def __call__(self, context: RewardContext) -> Tuple[float, Dict[str, float]]:
        separation = context.get("separation")
        if not np.isinf(separation) and separation < self._discomfort_dist:
            reward = self._coefficient * (-self._discomfort_dist / 2. + separation / 2.)
        else:
            reward = 0.
        return reward, {"discomfort_penalty": reward}


@nip
class StepPenalty(AbstractReward):

    def __init__(self, penalty_magnitude: float = 0.01):
        assert penalty_magnitude >= 0., "Penalty magnitude must be non-negative, minus sign will be added automatically"
        self._penalty = -penalty_magnitude

    def __call__(self, context: RewardContext) -> Tuple[float, Dict[str, float]]:
        reward = self._penalty
        return reward, {"step_penalty": reward}
