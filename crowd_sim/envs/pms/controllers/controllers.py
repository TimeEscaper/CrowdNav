import numpy as np
import do_mpc
import casadi as cs

from abc import ABC, abstractmethod
from typing import Set, List, Dict, Optional, Tuple, Any, Union
from nip import nip
from pyminisim.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS


class AbstractController(ABC):

    @abstractmethod
    def step(self, robot_state: np.ndarray, ped_predictions: Dict[str, np.ndarray]) -> \
            Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def goal(self) -> Optional[np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def set_goal(self, state: np.ndarray, goal: np.ndarray):
        raise NotImplementedError()


class AbstractControllerFactory(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


@nip
class DoMPCController(AbstractController):

    def __init__(self,
                 horizon: int,
                 dt: float,
                 total_peds: int,
                 Q: float,
                 R: np.ndarray,
                 W: float,
                 lb: Union[Tuple[float, float], np.ndarray],
                 ub: Union[Tuple[float, float], np.ndarray],
                 r_robot: float,
                 r_ped: float,
                 dummy_ped: Tuple[float, float],
                 solver: str,
                 cost_type: str,
                 constraint: Optional[Tuple[str, float]]) -> None:
        super(DoMPCController, self).__init__()

        # Architecture requires at least one dummy pedestrian in the system
        if total_peds == 0:
            total_peds = 1
        self._total_peds = total_peds
        self._horizon = horizon
        self._dummy_ped = np.array(dummy_ped)

        # System model
        self._model = do_mpc.model.Model("discrete")

        # State variables
        x = self._model.set_variable("_x", "x")
        y = self._model.set_variable("_x", "y")
        phi = self._model.set_variable("_x", "phi")
        v = self._model.set_variable("_u", "v")
        w = self._model.set_variable("_u", "w")

        # Discrete equations of motion
        self._model.set_rhs("x", x + v * cs.cos(phi) * dt)
        self._model.set_rhs("y", y + v * cs.sin(phi) * dt)
        self._model.set_rhs("phi", phi + w * dt)

        # Pedestrians
        self._state_peds = self._model.set_variable("_tvp", "p_peds", shape=(2, total_peds))
        # Pedestrians' covariances
        # Each covariance matrix is represented a row [a1, a2, b1, b2]
        self._covariances_peds = self._model.set_variable("_tvp", "cov_peds", shape=(4, total_peds))
        # Inverse of covariances
        # Each covariance matrix is represented a row [a1, a2, b1, b2]
        self._inverse_covariances_peds = self._model.set_variable("_tvp", "inv_cov_peds", shape=(4, total_peds))
        # Goal
        current_goal = self._model.set_variable("_tvp", "current_goal", shape=3)
        # Current adjustable margin for constraints
        current_margin = self._model.set_variable("_tvp", "current_margin", shape=1)
        # Initial position on the prediction step
        p_rob_0 = self._model.set_variable("_tvp", "p_rob_0", shape=3)

        self._model.setup()

        # MPC controller
        self._mpc = do_mpc.controller.MPC(self._model)
        # Controller setup
        setup_mpc = {
            "n_horizon": horizon - 1,
            "t_step": dt,
            "store_full_solution": True,
            "nlpsol_opts": {"ipopt.print_level": 0,
                            "ipopt.sb": "yes",
                            "print_time": 0,
                            'ipopt.linear_solver': solver}
        }
        self._mpc.set_param(**setup_mpc)

        # Horizontally concatenated array of robot current position for all the pedestrians
        p_robot_hcat = cs.hcat([self._model._x.cat[:2] for _ in range(total_peds)])
        p_peds = self._state_peds[:2, :]
        u = self._model._u.cat

        # Stage cost
        if cost_type == "mahalanobis":
            S = 0
            delta = (p_robot_hcat - p_peds)
            for ped_ind in range(total_peds):
                S += 1 / (delta[:, ped_ind].T @ cs.reshape(
                    self._inverse_covariances_peds[:, ped_ind], 2, 2) @ delta[:, ped_ind])
            stage_cost = u.T @ R @ u + W * S

        elif cost_type == "euclidean":
            S = 0
            delta = (p_robot_hcat - p_peds)
            for ped_ind in range(self._total_peds):
                S += 1 / (delta[:, ped_ind].T @ delta[:, ped_ind])
            stage_cost = u.T @ R @ u + W * S

        elif cost_type == "simple":
            stage_cost = u.T @ R @ u
        else:
            raise ValueError(f"Unknown cost type {cost_type}")

        # Goal cost
        p_rob_N = self._model._x.cat[:3]
        delta_p = cs.norm_2(p_rob_N[:2] - current_goal[:2]) / cs.norm_2(p_rob_0[:2] - current_goal[:2])
        goal_cost = Q * delta_p ** 2  # GO-MPC Cost-function

        # Set cost
        self._mpc.set_objective(lterm=stage_cost + goal_cost,
                                mterm=goal_cost)

        self._mpc.set_rterm(v=1e-2, w=1e-2)

        # Bounds
        self._mpc.bounds['lower', '_u', 'v'] = lb[0]
        self._mpc.bounds['lower', '_u', 'w'] = lb[1]
        self._mpc.bounds['upper', '_u', 'v'] = ub[0]
        self._mpc.bounds['upper', '_u', 'w'] = ub[1]

        if constraint is not None:
            constraint_type, constraint_value = constraint
            if constraint_type == "euclidean":
                lb_dists_square = np.array([(r_robot + r_ped + constraint_value) ** 2 for _ in range(total_peds)])
                dx_dy_square = (p_robot_hcat - p_peds) ** 2
                pedestrians_distances_squared = dx_dy_square[0, :] + dx_dy_square[1, :]
                final_constraint = pedestrians_distances_squared + current_margin ** 2
                self._mpc.set_nl_cons("euclidean_dist_to_peds", -final_constraint,
                                      ub=-lb_dists_square)

            elif constraint_type == "mahalanobis":
                delta = (p_robot_hcat - p_peds)
                array_mahalanobis_distances = cs.SX(1, total_peds)
                for ped_ind in range(self._total_peds):
                    array_mahalanobis_distances[ped_ind] = delta[:, ped_ind].T @ cs.reshape(
                        self._inverse_covariances_peds[:, ped_ind], 2, 2) @ delta[:, ped_ind]

                array_mahalanobis_bounds = cs.SX(1, total_peds)
                V_s = np.pi * (r_robot + r_ped + 1) ** 2
                for ped_ind in range(self._total_peds):
                    deter = 2 * np.pi * DoMPCController._determinant(
                        cs.reshape(self._covariances_peds[:, ped_ind], 2, 2))
                    deter = cs.sqrt(deter)
                    array_mahalanobis_bounds[ped_ind] = 2 * cs.log(deter * constraint_value / V_s)

                self._mpc.set_nl_cons("mahalanobis_dist_to_peds",
                                      -array_mahalanobis_distances - array_mahalanobis_bounds,
                                      ub=np.zeros(total_peds))

            else:
                raise ValueError(f"Unknown constraint type {constraint_type}")

        # Time-variable-parameter function for pedestrian
        self._mpc_tvp_fun = self._mpc.get_tvp_template()

        def mpc_tvp_fun(t_now):
            return self._mpc_tvp_fun

        self._mpc.set_tvp_fun(mpc_tvp_fun)

        # Setup
        self._mpc.setup()
        self._initial_guess_set = False
        # # set initial guess
        # self._mpc.x0 = np.array(init_state)
        # self._mpc.set_initial_guess()
        # # set basic goal
        # self.set_new_goal(init_state,
        #                   goal)

        self._goal = None

    def set_goal(self, state: np.ndarray, goal: np.ndarray):
        self._goal = np.array([
            goal[0],
            goal[1],
            np.arctan2(goal[1] - state[1], goal[0] - state[0])
        ])
        self._mpc_tvp_fun['_tvp', :, 'current_goal'] = self._goal

    def set_margin(self, margin: float):
        self._mpc_tvp_fun['_tvp', :, 'current_margin'] = margin

    @property
    def goal(self) -> Optional[np.ndarray]:
        if self._goal is not None:
            return self._goal[:2]
        return None

    def step(self, robot_state: np.ndarray, ped_predictions: Dict[int, np.ndarray]) -> \
            Tuple[np.ndarray, Dict[str, Any]]:
        assert self._goal is not None, f"Goal must be set before calling 'step' method"
        if not self._initial_guess_set:
            self._mpc.x0 = robot_state
            self._mpc.set_initial_guess()
            self._initial_guess_set = True

        self._update_ped_tvp(ped_predictions)

        control = self._mpc.make_step(robot_state).T[0]
        mpc_trajectory = self._get_mpc_trajectory()

        return control, {"mpc_traj": mpc_trajectory}

    def _update_ped_tvp(self, ped_predictions: Dict[int, np.ndarray]):
        predicted_trajectories = np.tile(self._dummy_ped, (self._horizon, self._total_peds, 1))
        predicted_covs = np.tile(np.array([[0.001, 0.0], [0., 0.001]]), (self._horizon, self._total_peds, 1, 1))

        for i, ped_prediction in enumerate(ped_predictions.values()):
            predicted_trajectories[:, i, :2] = ped_prediction[0]
            predicted_covs[:, i, :] = ped_prediction[1]

        predicted_covs_inv = np.linalg.inv(predicted_covs)
        predicted_covs_flatten = predicted_covs.reshape((self._horizon, self._total_peds, 4))
        predicted_covs_inv_flatten = predicted_covs_inv.reshape((self._horizon, self._total_peds, 4))

        for step in range(len(self._mpc_tvp_fun['_tvp', :, 'p_peds'])):
            self._mpc_tvp_fun['_tvp', step, 'p_peds'] = predicted_trajectories[step].T
            self._mpc_tvp_fun['_tvp', step, 'cov_peds'] = predicted_covs_flatten[step].T
            self._mpc_tvp_fun['_tvp', step, 'inv_cov_peds'] = predicted_covs_inv_flatten[step].T

    def _get_mpc_trajectory(self) -> np.ndarray:
        rob_x_pred = self._mpc.data.prediction(('_x', 'x'))[0]
        rob_y_pred = self._mpc.data.prediction(('_x', 'y'))[0]
        array_xy_pred = np.concatenate([rob_x_pred, rob_y_pred], axis=1)
        return array_xy_pred


@nip
class DefaultMPCFactory(AbstractControllerFactory):

    def __init__(self,
                 horizon: int,
                 total_peds: int,
                 lb: Tuple[float, float],
                 ub: Tuple[float, float],
                 dt: float = 0.1,
                 Q: float = 100.,
                 R: np.ndarray = np.diag([1., 1.]),
                 W: float = 1000.,
                 dummy_ped: Tuple[float, float] = (10000., 10000.),
                 solver: str = "MUMPS"):
        super(DefaultMPCFactory, self).__init__()
        self._horizon = horizon
        self._total_peds = total_peds
        self._lb = lb
        self._ub = ub
        self._dt = dt
        self._Q = Q
        self._R = R
        self._W = W
        self._dummy_ped = dummy_ped
        self._solver = solver

    def __call__(self) -> AbstractController:
        return DoMPCController(
            horizon=self._horizon,
            dt=self._dt,
            total_peds=self._total_peds,
            Q=self._Q,
            R=self._R,
            W=self._W,
            lb=self._lb,
            ub=self._ub,
            r_robot=ROBOT_RADIUS,
            r_ped=PEDESTRIAN_RADIUS,
            dummy_ped=self._dummy_ped,
            solver=self._solver,
            cost_type="mahalanobis",
            constraint=("euclidean", 0.3)
        )
