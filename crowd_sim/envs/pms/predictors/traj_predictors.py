import pkg_resources
import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Tuple, Optional
from nip import nip
from crowd_sim.envs.pms.predictors.covariance_net.model import CovarianceNet
from crowd_sim.envs.pms.predictors.covariance_net.cons_vel_model import batched_constant_velocity_model
from crowd_sim.envs.pms.predictors.covariance_net.model_utils  import get_rotation_translation_matrix


class AbstractTrajectoryPredictor(ABC):

    def __init__(self,
                 dt: float,
                 horizon: int,
                 history_length: int) -> None:
        assert horizon > 0, f"Prediction horizon must be positive, {horizon} is given"
        assert history_length > 0, f"History length must be positive, {history_length} is given"
        self._dt = dt
        self._horizon = horizon
        self._history_length = history_length

    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @property
    def history_length(self) -> int:
        return self._history_length

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def dt(self) -> float:
        return self._dt

@nip
class CovarianceNetPredictor(AbstractTrajectoryPredictor):
    _MODEL_HORIZON = 25
    _DT = 0.1
    _HISTORY_LENGTH = 8

    def __init__(self, horizon: int, device: str = "cpu"):
        assert horizon <= CovarianceNetPredictor._MODEL_HORIZON, \
            f"Horizon can not be larger than horizon that the model was trained with " \
            f"({CovarianceNetPredictor._MODEL_HORIZON} steps)"
        super(CovarianceNetPredictor, self).__init__(dt=CovarianceNetPredictor._DT,
                                                     horizon=horizon,
                                                     history_length=CovarianceNetPredictor._HISTORY_LENGTH)
        self._model = CovarianceNet(input_size=2,
                                    hidden_size=64,
                                    prediction_steps=CovarianceNetPredictor._MODEL_HORIZON)
        self._model.load_state_dict(
            torch.load(pkg_resources.resource_filename("crowd_sim.envs.pms.predictors.covariance_net",
                                                       "weights/covariance_net_dt_01_horizon_25.pth"),
                       map_location=device))
        _ = self._model.to(device)
        self._model = self._model.eval()
        self._device = device
        self._horizon = horizon

    def predict(self, joint_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # joint_history: (history, n_neighbours, state_dim)
        n_agents = joint_history.shape[1]
        if n_agents == 1:
            ego_agent = joint_history[:, 0, :2]
            ego_vel = joint_history[-1, 0, 2:]
            neighbours_stub = np.ones((joint_history.shape[0], 1, 2)) * 1000.
            neighbours_stub = neighbours_stub.transpose((1, 0, 2))
            pred, cov = self._predict_ego_agent(ego_agent, ego_vel, neighbours_stub)
            return pred[:self._horizon, np.newaxis, :], cov[:self._horizon, np.newaxis, :, :]

        preds = []
        covs = []
        for i in range(n_agents):
            ego_agent = joint_history[:, i, :2]
            ego_vel = joint_history[-1, i, 2:]
            neighbours = joint_history[:, [j for j in range(n_agents) if j != i], :2].transpose((1, 0, 2))
            pred, cov = self._predict_ego_agent(ego_agent, ego_vel, neighbours)
            preds.append(pred)
            covs.append(cov)
        preds = np.array(preds).transpose((1, 0, 2))
        covs = np.array(covs).transpose((1, 0, 2, 3))
        return preds[:self._horizon], covs[:self._horizon]

    def _predict_ego_agent(self,
                           ego_history: np.ndarray,
                           ego_vel: np.ndarray,
                           neighbours_history: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # ego_history: (history, pose_dim)
        # ego_vel: (velocity_dim,)
        # neighbours_history: (history, n_neighbours, pose_dim)
        rt_matrix = get_rotation_translation_matrix(ego_history[7, 0], ego_history[7, 1],
                                                    ego_vel[0], ego_vel[1])

        ego_history = np.matmul(ego_history + rt_matrix[:2, 2], rt_matrix[:2, :2].T)
        neighbours_history = np.matmul(neighbours_history + rt_matrix[:2, 2], rt_matrix[:2, :2].T)

        with torch.no_grad():
            pred, cov = self._model(torch.Tensor(ego_history).unsqueeze(0).to(self._device),
                                    ego_vel[np.newaxis],
                                    torch.Tensor(neighbours_history).unsqueeze(0).to(self._device))
            pred = pred.clone().detach().cpu().numpy()[0]
            cov = cov.clone().detach().cpu().numpy()[0]

        inv_rotation = np.linalg.inv(rt_matrix[:2, :2].T)
        pred = np.matmul(pred[:, :2], inv_rotation) - rt_matrix[:2, 2]
        cov = np.matmul(inv_rotation, np.matmul(cov, inv_rotation.T))

        return pred, cov


@nip
class ConstantVelocityPredictor(AbstractTrajectoryPredictor):

    def __init__(self, horizon: int, history_length: int):
        super(ConstantVelocityPredictor, self).__init__(dt=0.1,
                                                        horizon=horizon,
                                                        history_length=history_length)

    def predict(self, joint_history: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # joint_history: (history, n_neighbours, state_dim)
        predictions = batched_constant_velocity_model(positions=joint_history.transpose((1, 0, 2))[:, :, :2],
                                                      current_velocity=None,
                                                      num_future_positions=self.horizon).transpose((1, 0, 2))
        dummy_cov = np.tile(np.eye(2) * 0.001, (self.horizon, joint_history.shape[1], 1, 1))
        return predictions, dummy_cov
