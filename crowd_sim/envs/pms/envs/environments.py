from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Tuple, Callable

import gym
import numpy as np
from nip import nip
from pyminisim.core import PEDESTRIAN_RADIUS, ROBOT_RADIUS
from pyminisim.core import Simulation, SimulationState
from pyminisim.pedestrians import HeadedSocialForceModelPolicy, OptimalReciprocalCollisionAvoidance, \
    RandomWaypointTracker, FixedWaypointTracker
from pyminisim.robot import UnicycleRobotModel
from pyminisim.sensors import PedestrianDetector, PedestrianDetectorConfig
from pyminisim.visual import Renderer, CircleDrawing, Covariance2dDrawing
from pyminisim.world_map import EmptyWorld

from crowd_sim.envs.pms.controllers.controllers import AbstractController, AbstractControllerFactory, DefaultMPCFactory
from crowd_sim.envs.pms.envs import CompositeAgentsSampler, RandomAgentsSampler, CircularRobotCentralSampler
from crowd_sim.envs.pms.envs.curriculum import AbstractCurriculum, DummyCurriculum
from crowd_sim.envs.pms.envs.rewards import AbstractReward, RewardContext
from crowd_sim.envs.pms.envs.sim_config_samplers import AbstractActionSpaceConfig, SimConfig, \
    ContinuousPolarSubgoalActionSpace, RandomProblemSampler
from crowd_sim.envs.pms.envs.wrappers import StackHistoryWrapper
from crowd_sim.envs.pms.predictors.tracker import PedestrianTracker, CovarianceNetTrackerFactory
from crowd_sim.envs.utils.action import ActionPoint
from crowd_sim.envs.utils.info import Collision, Timeout, ReachGoal, Danger, Nothing
from crowd_sim.envs.utils.state import ObservableState


class AbstractEnvFactory(ABC):

    @abstractmethod
    def __call__(self, is_eval: bool) -> gym.Env:
        raise NotImplementedError()


class PyMiniSimWrap:

    def __init__(self,
                 action_space_config: AbstractActionSpaceConfig,
                 sim_config: SimConfig,
                 curriculum: AbstractCurriculum,
                 ped_tracker: PedestrianTracker,
                 controller: Optional[AbstractController] = None):
        if controller is not None:
            assert action_space_config.action_type == AbstractActionSpaceConfig.TYPE_SUBGOAL, \
                f"Subgoal action space must be set if controller is specified"
        else:
            assert action_space_config.action_type == AbstractActionSpaceConfig.TYPE_END2END, \
                f"End2end action space must be set if controller is not specified"

        self._action_space_config = action_space_config
        self._sim_config = sim_config
        self._curriculum = curriculum
        self._render = sim_config.render

        self._ped_tracker = ped_tracker
        self._controller = controller

        self._step_cnt = 0

        self._sim: Simulation = None
        self._renderer: Renderer = None
        self._robot_goal: np.ndarray = None
        self._goal_reach_threshold: float = None
        self._subgoal_reach_threshold: float = None
        self._max_steps: int = None
        self._max_subgoal_steps: int = None

    def update_curriculum(self):
        self._curriculum.update_stage()

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space_config.action_space

    @property
    def goal(self) -> np.ndarray:
        return self._robot_goal

    @property
    def current_step_cnt(self) -> int:
        return self._step_cnt

    @property
    def render_enabled(self) -> bool:
        return self._render

    @property
    def sim_state(self) -> SimulationState:
        return self._sim.current_state

    @property
    def ped_tracker(self) -> PedestrianTracker:
        return self._ped_tracker

    def _step_end2end(self, action: np.ndarray) -> Tuple[bool, float]:
        hold_time = 0.
        has_collision = False
        min_separation_distance = np.inf
        while hold_time < self._sim_config.policy_dt:
            self._sim.step(action)
            hold_time += self._sim_config.sim_dt
            if self._renderer is not None:
                self._renderer.render()
            collisions = self._sim.current_state.world.robot_to_pedestrians_collisions
            has_collision = collisions is not None and len(collisions) > 0

            robot_position = self._sim.current_state.world.robot.pose[:2]
            if self._sim.current_state.world.pedestrians is not None:
                ped_positions = np.array([v[:2] for v in self._sim.current_state.world.pedestrians.poses.values()])
                if ped_positions.shape[0] > 0:
                    separation = np.min(
                        np.linalg.norm(robot_position - ped_positions, axis=1)) - ROBOT_RADIUS - PEDESTRIAN_RADIUS
                    if separation < min_separation_distance:
                        min_separation_distance = separation

            if has_collision:
                break

        self._ped_tracker.update(self._get_detections())
        self._draw_predictions()
        return has_collision, min_separation_distance

    def _step_subgoal(self, action: np.ndarray) -> Tuple[bool, bool, bool, float]:
        if action is not None:
            subgoal = self._subgoal_to_absolute(action)
            max_subgoal_steps = self._max_subgoal_steps
        else:
            # Controller as expert, set goal as subgoal and randomly truncate the subgoal steps
            subgoal = self._robot_goal.copy()
            choice = np.random.choice([False, True], p=[0.4, 0.6])
            if choice:
                max_subgoal_steps = self._max_subgoal_steps
            else:
                max_subgoal_steps = np.random.randint(8, self._max_subgoal_steps)
        robot_state = self._sim.current_state.world.robot.pose
        self._controller.set_goal(state=robot_state, goal=subgoal)

        if self._renderer is not None:
            self._renderer.draw(f"subgoal", CircleDrawing(subgoal, 0.05, (0, 0, 255)))

        step_cnt = 0
        has_collision = False
        subgoal_reached = False
        goal_reached = False
        min_separation_distance = np.inf

        while True:
            control, info = self._controller.step(robot_state, self._ped_tracker.get_predictions())
            if "mpc_traj" in info and self._renderer is not None:
                self._renderer.draw(f"mpc_traj", CircleDrawing(info["mpc_traj"], 0.04, (209, 133, 128)))
            has_collision, separation = self._step_end2end(control)
            if separation < min_separation_distance:
                min_separation_distance = separation

            if has_collision:
                return has_collision, subgoal_reached, goal_reached, min_separation_distance
            robot_state = self._sim.current_state.world.robot.pose
            subgoal_reached = np.linalg.norm(robot_state[:2] - subgoal) - ROBOT_RADIUS < self._subgoal_reach_threshold
            goal_reached = np.linalg.norm(
                robot_state[:2] - self._robot_goal) - ROBOT_RADIUS < self._goal_reach_threshold
            if goal_reached:
                return has_collision, subgoal_reached, goal_reached, min_separation_distance
            if subgoal_reached:
                return has_collision, subgoal_reached, goal_reached, min_separation_distance
            step_cnt += 1
            if step_cnt >= max_subgoal_steps:
                break

        return has_collision, subgoal_reached, goal_reached, min_separation_distance

    def step(self, action: Optional[np.ndarray]) -> Tuple[bool, bool, bool, float]:
        if action is not None:
            action = self._action_space_config.get_action(action)

        if self._controller is None:
            has_collision, min_separation_distance = self._step_end2end(action)
            goal_reached = np.linalg.norm(self._sim.current_state.world.robot.pose[:2] -
                                          self._robot_goal) - ROBOT_RADIUS < self._goal_reach_threshold
        else:
            has_collision, subgoal_reached, goal_reached, min_separation_distance = self._step_subgoal(action)

        self._step_cnt += 1
        truncated = (self._step_cnt >= self._max_steps) and not has_collision

        if has_collision or truncated:
            success = False
        else:
            success = goal_reached

        return has_collision, truncated, success, min_separation_distance

    def reset(self, phase='test', test_case=None):

        problem = self._curriculum.get_problem_sampler().sample() if phase == "train" \
            else self._curriculum.get_eval_problem_sampler().sample()
        self._goal_reach_threshold = problem.goal_reach_threshold
        self._max_steps = problem.max_steps
        if self._controller is not None:
            assert problem.subgoal_reach_threshold is not None, "Subgoal reach threshold must be set in subgoal mode"
        self._subgoal_reach_threshold = problem.subgoal_reach_threshold
        self._max_subgoal_steps = problem.max_subgoal_steps or np.inf

        agents_sample = self._curriculum.get_agents_sampler().sample() if phase == "train" \
            else self._curriculum.get_eval_agents_sampler().sample()
        self._robot_goal = agents_sample.robot_goal

        robot_model = UnicycleRobotModel(initial_pose=agents_sample.robot_initial_pose,
                                         initial_control=np.array([0.0, np.deg2rad(0.0)]))

        if problem.ped_model != "none" and agents_sample.n_peds > 0:
            if agents_sample.ped_goals is None:
                waypoint_tracker = RandomWaypointTracker(world_size=agents_sample.world_size)
            else:
                waypoint_tracker = FixedWaypointTracker(initial_positions=agents_sample.ped_initial_poses[:, :2],
                                                        waypoints=agents_sample.ped_goals,
                                                        loop=True)

            if problem.ped_model == "hsfm":
                ped_model = HeadedSocialForceModelPolicy(waypoint_tracker=waypoint_tracker,
                                                         n_pedestrians=agents_sample.n_peds,
                                                         initial_poses=agents_sample.ped_initial_poses,
                                                         robot_visible=problem.robot_visible,
                                                         pedestrian_linear_velocity_magnitude=agents_sample.ped_linear_vels)
            elif problem.ped_model == "orca":
                # TODO: Implement velocities in ORCA
                ped_model = OptimalReciprocalCollisionAvoidance(dt=self._sim_config.sim_dt,
                                                                waypoint_tracker=waypoint_tracker,
                                                                n_pedestrians=agents_sample.n_peds,
                                                                initial_poses=agents_sample.ped_initial_poses,
                                                                robot_visible=problem.robot_visible)
            else:
                raise ValueError()
        else:
            ped_model = None

        ped_detector = PedestrianDetector(
            config=PedestrianDetectorConfig(max_dist=problem.detector_range,
                                            fov=problem.detector_fov,
                                            return_type=PedestrianDetectorConfig.RETURN_ABSOLUTE))

        sim = Simulation(world_map=EmptyWorld(),
                         robot_model=robot_model,
                         pedestrians_model=ped_model,
                         sensors=[ped_detector],
                         sim_dt=self._sim_config.sim_dt,
                         rt_factor=self._sim_config.rt_factor)
        if self._render:
            renderer = Renderer(simulation=sim,
                                resolution=70.,
                                screen_size=(800, 800))
            renderer.draw("goal", CircleDrawing(center=self._robot_goal[:2],
                                                radius=0.05,
                                                color=(255, 0, 0)))
        else:
            renderer = None

        self._sim = sim
        self._renderer = renderer

        self._step_cnt = 0

        self._ped_tracker.reset()
        self._ped_tracker.update(self._get_detections())
        self._draw_predictions()

    def enable_render(self):
        self._render = True

    def _draw_predictions(self):
        if self._renderer is None:
            return

        predictions = self._ped_tracker.get_predictions()
        if len(predictions) > 0:
            pred_traj = np.concatenate([v[0].reshape((-1, 2)) for v in predictions.values()], axis=0)
            pred_covs = np.concatenate([v[1].reshape((-1, 2, 2)) for v in predictions.values()], axis=0)
            self._renderer.draw(f"pred_traj", CircleDrawing(pred_traj, 0.05, (173, 153, 121)))
            self._renderer.draw(f"pred_covs", Covariance2dDrawing(pred_traj, pred_covs, (173, 153, 121), 0.05,
                                                                  n_sigma=1))
        else:
            self._renderer.clear_drawings(["pred_traj", "pred_covs"])

    def _get_detections(self) -> Dict[int, np.ndarray]:
        # TODO: add velocities to tracker
        if self._sim.current_state.world.pedestrians is not None:
            vels = self._sim.current_state.world.pedestrians.velocities
        else:
            vels = None
        detections = {k: np.array([v[0], v[1], vels[k][0], vels[k][1]]
                                  if vels is not None else [v[0], v[1], 0., 0.])
                      for k, v in self._sim.current_state.sensors["pedestrian_detector"].reading.pedestrians.items()}
        return detections

    def _subgoal_to_absolute(self, subgoal_polar: np.ndarray) -> np.ndarray:
        x_rel_rot = subgoal_polar[0] * np.cos(subgoal_polar[1])
        y_rel_rot = subgoal_polar[0] * np.sin(subgoal_polar[1])
        robot_pose = self._sim.current_state.world.robot.pose
        theta = robot_pose[2]
        x_rel = x_rel_rot * np.cos(theta) - y_rel_rot * np.sin(theta)
        y_rel = x_rel_rot * np.sin(theta) + y_rel_rot * np.cos(theta)
        x_abs = x_rel + robot_pose[0]
        y_abs = y_rel + robot_pose[1]
        return np.array([x_abs, y_abs])


class SocialNavEnv(gym.Env):

    _PEDS_PADDING = 8

    def __init__(self, render: bool = False):

        self._sim_wrap = PyMiniSimWrap(
            action_space_config=ContinuousPolarSubgoalActionSpace(
                lb=(1., -np.deg2rad(110.)),
                ub=(3., np.deg2rad(110.)),
                normalize=False
            ),

            sim_config=SimConfig(
                render=render
            ),

            curriculum=DummyCurriculum(
                agents_sampler=CompositeAgentsSampler(
                    samplers=[
                        RandomAgentsSampler(
                            n_peds=(1, 4),
                            sampling_square=(6, 6),
                            min_robot_goal_distance=2.,
                            ped_linear_vels=(1., 1.8)
                        ),
                        RandomAgentsSampler(
                            n_peds=(4, 8),
                            sampling_square=(8, 8),
                            min_robot_goal_distance=5.,
                            ped_linear_vels=(1., 1.8)
                        ),
                        CircularRobotCentralSampler(
                            n_peds=(2, 4),
                            ped_circle_radius=(1.8, 3.5),
                            ped_linear_vels=(1., 1.8)
                        )
                    ]
                ),
                problem_sampler=RandomProblemSampler(
                    ped_model="hsfm",
                    robot_visible=False,
                    detector_range=5.,
                    detector_fov=360.,
                    goal_reach_threshold=0.0,
                    max_steps=20,
                    subgoal_reach_threshold=0.1,
                    max_subgoal_steps=0.1
                ),
                n_eval_episodes=10
            ),

            ped_tracker=CovarianceNetTrackerFactory(
                horizon=25,
                max_ghost_tracking_time=8,
                device="cuda"
            )(),

            controller=DefaultMPCFactory(
                horizon=25,
                total_peds=SocialNavEnv._PEDS_PADDING,
                lb=(0., -2.),
                ub=(2., 2.)
            )()
        )

        self.robot = None
        self.global_time = None
        self.time_step = 0.1

        self.success_reward: Optional[float] = None
        self.collision_penalty: Optional[float] = None
        self.discomfort_dist: Optional[float] = None
        self.discomfort_penalty_factor: Optional[float] = None

    def configure(self, config):
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')

    def set_robot(self, robot):
        self.robot = robot

    def reset(self, phase='test', test_case=None):
        self._sim_wrap.reset(phase, test_case)

        robot_pose = self._sim_wrap.sim_state.world.robot.pose
        robot_vel = self._sim_wrap.sim_state.world.robot.velocity
        robot_goal = self._sim_wrap.goal
        self.robot.set(robot_pose[0],
                       robot_pose[1],
                       robot_goal[0],
                       robot_goal[1],
                       robot_vel[0],
                       robot_vel[1],
                       robot_pose[2],
                       ROBOT_RADIUS)  # TODO: Check with v_pref
        self.robot.time_step = 0.1  # TODO: Should we use this value for subgoals?
        self.robot.policy.time_step = 0.1
        return self._get_observation()

    def step(self, action: ActionPoint, update=True):
        action = np.array([action.px, action.py])
        prev_pose = self._sim_wrap.sim_state.world.robot.pose
        collision, truncated, success, separation = self._sim_wrap.step(action)

        if collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif truncated:
            reward = 0.
            done = True
            info = Timeout()
        elif success:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif separation < self.discomfort_dist:
            reward = (separation - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(separation)
        else:
            reward = 0.
            done = False
            info = Nothing()

        new_pose = self._sim_wrap.sim_state.world.robot.pose
        subgoal_action = subgoa

        obs = self._get_observation()

        action = ActionPoint(s_)

        self.robot.step(action)
        self.global_time += self.time_step

        return obs, reward, done, info

    def render(self, mode: str = "human", output_file: Optional[str] = None):
        pass

    def _get_observation(self):
        obs = []
        current_poses = self._sim_wrap.ped_tracker.get_current_poses(return_velocities=True)
        for v in current_poses.values():
            obs.append(ObservableState(v[0], v[1], v[2], v[3], PEDESTRIAN_RADIUS))
        if len(obs) < SocialNavEnv._PEDS_PADDING:
            for _ in range(SocialNavEnv._PEDS_PADDING - len(obs)):
                obs.append(ObservableState(-10., -10., 0., 0., 0.01))
        return obs


# @nip
# class SocialNavGraphEnv(gym.Env):
#
#     def __init__(self,
#                  action_space_config: AbstractActionSpaceConfig,
#                  sim_config: SimConfig,
#                  curriculum: AbstractCurriculum,
#                  ped_tracker: PedestrianTracker,
#                  reward: AbstractReward,
#                  peds_padding: int,
#                  is_eval: bool,
#                  rl_tracker_horizon: int,
#                  controller: Optional[AbstractController] = None,
#                  obs_mode: str = "prediction"):
#         self._sim_wrap = PyMiniSimWrap(action_space_config,
#                                        sim_config,
#                                        curriculum,
#                                        ped_tracker,
#                                        is_eval,
#                                        controller)
#         assert obs_mode in ["prediction", "current", "v_learning"], \
#             f"Only 'prediction', 'current' and 'v_learning' modes available, {obs_mode} is given"
#         self._reward = reward
#         self._rl_tracker_horizon = rl_tracker_horizon
#         self._obs_mode = obs_mode
#
#         self._peds_padding = peds_padding
#
#         if obs_mode != "v_learning":
#             self.observation_space = gym.spaces.Dict({
#                 "peds_traj": gym.spaces.Box(
#                     low=-np.inf,
#                     high=np.inf,
#                     shape=(self._peds_padding, rl_tracker_horizon + 1, 2) if obs_mode == "prediction"
#                     else (self._peds_padding, 5),  # Current state + predictions = 1 + horizon for prediction mode
#                     dtype=np.float
#                 ),
#                 "peds_visibility": gym.spaces.Box(
#                     low=False,
#                     high=True,
#                     shape=(self._peds_padding,),
#                     dtype=np.bool
#                 ),
#                 "robot_state": gym.spaces.Box(
#                     low=np.array([-np.inf, -np.inf, -np.inf, -np.pi, -np.inf, -np.inf, -np.inf]),
#                     high=np.array([np.inf, np.inf, np.inf, np.pi, np.inf, np.inf, np.inf]),
#                     shape=(7,),
#                     dtype=np.float
#                 )
#             })
#
#         else:
#             self.observation_space = gym.spaces.Dict({
#                 "peds": gym.spaces.Box(
#                     # p_x, p_y, v_x, v_y, d (in robot frame)
#                     low=np.tile(np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0.]), (self._peds_padding, 1)),
#                     high=np.tile(np.array([np.inf, np.inf, np.inf, np.inf, np.inf]), (self._peds_padding, 1)),
#                     shape=(self._peds_padding, 5)
#                 ),
#                 "peds_visibility": gym.spaces.Box(
#                     low=False,
#                     high=True,
#                     shape=(self._peds_padding,),
#                     dtype=np.bool
#                 ),
#                 "peds_prediction": gym.spaces.Box(
#                     # p_x, p_y, v_x, v_y (in global frame)
#                     low=-np.inf,
#                     high=np.inf,
#                     shape=(self._peds_padding, ped_tracker.horizon, 4)
#                 ),
#                 "peds_prediction_cov": gym.spaces.Box(
#                     low=-np.inf,
#                     high=np.inf,
#                     shape=(self._peds_padding, ped_tracker.horizon, 2, 2)
#                 ),
#                 "robot": gym.spaces.Box(
#                     # d_goal, p_x^goal, p_y^goal, theta, v_x, v_y, omega
#                     low=np.array([0., -np.inf, -np.inf, -np.pi, -2., -2., -2 * np.pi]),
#                     high=np.array([np.inf, np.inf, np.inf, np.pi, 2., 2., 2 * np.pi]),
#                     shape=(7,)
#                 ),
#                 "robot_global": gym.spaces.Box(
#                     low=np.array([-np.inf, -np.inf, -np.pi]),
#                     high=np.array([np.inf, np.inf, np.pi]),
#                     shape=(3,)
#                 ),
#                 "goal_global": gym.spaces.Box(
#                     low=np.array([-np.inf, -np.inf]),
#                     high=np.array([np.inf, np.inf]),
#                     shape=(2,)
#                 )
#             })
#
#         self.action_space = self._sim_wrap.action_space
#
#     def update_curriculum(self):
#         self._sim_wrap.update_curriculum()
#
#     def step(self, action: np.ndarray):
#         previous_robot_pose = self._sim_wrap.sim_state.world.robot.pose
#         previous_predictions = self._sim_wrap.ped_tracker.get_predictions()
#         goal = self._sim_wrap.goal
#
#         collision, truncated, success, separation = self._sim_wrap.step(action)
#         robot_pose = self._sim_wrap.sim_state.world.robot.pose
#         next_predictions = self._sim_wrap.ped_tracker.get_predictions()
#
#         reward_context = RewardContext()
#         reward_context.set("goal", goal)
#         reward_context.set("robot_pose", robot_pose)
#         reward_context.set("robot_velocity", self._sim_wrap.sim_state.world.robot.velocity)
#         reward_context.set("previous_robot_pose", previous_robot_pose)
#         reward_context.set("previous_ped_predictions", previous_predictions)
#         reward_context.set("next_ped_predictions", next_predictions)
#         reward_context.set("separation", separation)
#
#         # https://stable-baselines3.readthedocs.io/en/master/common/logger.html#eval
#         if collision:
#             done = True
#             info = {"done_reason": "collision",
#                     "is_success": False}
#             reward_context.set("collision", True)
#         elif truncated:
#             done = True
#             info = {"done_reason": "truncated",
#                     "is_success": False,
#                     "TimeLimit.truncated": True}  # https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
#             reward_context.set("truncated", True)
#         elif success:
#             done = True
#             info = {"done_reason": "success",
#                     "is_success": True}
#             reward_context.set("success", True)
#         else:
#             done = False
#             info = {}
#
#         reward, reward_info = self._reward(reward_context)
#         info.update({"reward": reward_info})
#
#         observation = self._build_obs()
#
#         return observation, reward, done, info
#
#     def reset(self):
#         self._sim_wrap.reset()
#         observation = self._build_obs()
#         return observation
#
#     def render(self, mode="human"):
#         pass
#
#     def enable_render(self):
#         self._sim_wrap.enable_render()
#
#     @staticmethod
#     def _build_robot_obs(robot_pose: np.ndarray, robot_vel: np.ndarray, goal: np.ndarray) -> np.ndarray:
#         return np.array([np.linalg.norm(goal[:2] - robot_pose[:2]),
#                          goal[0] - robot_pose[0],
#                          goal[1] - robot_pose[1],
#                          robot_pose[2],
#                          robot_vel[0],
#                          robot_vel[1],
#                          robot_vel[2]]).astype(np.float32)
#
#     def _build_peds_obs_prediction(self, robot_pose: np.ndarray,
#                                    current_poses: Dict[int, np.ndarray], predictions: Dict[int, np.ndarray]) -> \
#             Tuple[np.ndarray, np.ndarray]:
#         obs_ped_traj = np.ones((self._peds_padding, self._rl_tracker_horizon + 1, 2)) * 100.
#         obs_peds_ids = current_poses.keys()
#         obs_peds_vis = np.zeros(self._peds_padding, dtype=np.bool)
#         for k in obs_peds_ids:
#             obs_ped_traj[k, 0, :] = current_poses[k] - robot_pose[:2]
#             obs_ped_traj[k, 1:, :] = predictions[k][:self._rl_tracker_horizon, :] - robot_pose[:2]
#             obs_peds_vis[k] = True
#
#         # TODO: Should we make soring optional?
#         distances = np.linalg.norm(obs_ped_traj[:, 0, :], axis=1)
#         sorted_indices = np.argsort(distances)
#         obs_ped_traj = obs_ped_traj[sorted_indices]
#         obs_peds_vis = obs_peds_vis[sorted_indices]
#
#         return obs_ped_traj, obs_peds_vis
#
#     def _build_peds_obs_current(self, robot_pose: np.ndarray,
#                                 current_poses: Dict[int, np.ndarray]) -> \
#             Tuple[np.ndarray, np.ndarray]:
#         obs_peds = np.ones((self._peds_padding, 5))
#         obs_peds_ids = current_poses.keys()
#         obs_peds_vis = np.zeros(self._peds_padding, dtype=np.bool)
#         for k in range(self._peds_padding):
#             if k in obs_peds_ids:
#                 obs_peds[k, :2] = current_poses[k] - robot_pose[:2]
#                 obs_peds[k, 2] = PEDESTRIAN_RADIUS
#                 obs_peds[k, 3] = np.linalg.norm(current_poses[k] - robot_pose[:2])
#                 obs_peds[k, 4] = PEDESTRIAN_RADIUS + ROBOT_RADIUS
#                 obs_peds_vis[k] = True
#             else:
#                 obs_peds[k, :2] = 100. - robot_pose[:2]
#                 obs_peds[k, 2] = 0.
#                 obs_peds[k, 3] = np.linalg.norm(100. - robot_pose[:2])
#                 obs_peds[k, 4] = 0.
#                 obs_peds_vis[k] = False
#
#         # TODO: Should we make soring optional?
#         sorted_indices = np.argsort(obs_peds[:, 3])
#         obs_peds = obs_peds[sorted_indices]
#         obs_peds_vis = obs_peds_vis[sorted_indices]
#
#         return obs_peds, obs_peds_vis
#
#     def _build_obs_v_learning(self):
#         goal = self._sim_wrap.goal
#         robot_pose = self._sim_wrap.sim_state.world.robot.pose
#         robot_vel = self._sim_wrap.sim_state.world.robot.velocity
#         current_poses = self._sim_wrap.ped_tracker.get_current_poses(return_velocities=True)
#         predictions = self._sim_wrap.ped_tracker.get_predictions()
#
#         obs_peds = np.tile([-10., -10., 0., 0., 100.], (self._peds_padding, 1))
#         obs_peds_vis = np.zeros(self._peds_padding, dtype=np.bool)
#         for k, v in current_poses.items():
#             # p_x, p_y, v_x, v_y, d (in robot frame)
#             obs_peds[k, :4] = current_poses[k]
#             obs_peds[k, 4] = np.linalg.norm(current_poses[k][:2] - robot_pose[:2])
#             obs_peds_vis[k] = True
#
#         obs_peds_prediction = np.tile([-10., -10., 0., 0.],
#                                       (self._peds_padding, self._sim_wrap.ped_tracker.horizon, 1))
#         obs_peds_prediction_cov = np.tile(np.eye(2) * 0.001,
#                                           (self._peds_padding, self._sim_wrap.ped_tracker.horizon, 1, 1))
#         for k in predictions.keys():
#             prediction = predictions[k][0]
#             obs_peds_prediction[k, :, :2] = prediction[:, :2]
#             vel_estimation = np.stack((np.ediff1d(prediction[:, 0]), np.ediff1d(prediction[:, 1])), axis=1)
#             vel_estimation = np.concatenate((vel_estimation, vel_estimation[-1, np.newaxis]), axis=0)
#             obs_peds_prediction[k, :, 2:] = vel_estimation
#             obs_peds_prediction_cov[k] = predictions[k][1]
#
#         sorted_indices = np.argsort(obs_peds[:, 4])
#         obs_peds = obs_peds[sorted_indices]
#         obs_peds_vis = obs_peds_vis[sorted_indices]
#         obs_peds_prediction = obs_peds_prediction[sorted_indices]
#         obs_peds_prediction_cov = obs_peds_prediction_cov[sorted_indices]
#
#         # d_goal, p_x^goal, p_y^goal, theta, v_x, v_y, omega
#         obs_robot = np.array([
#             np.linalg.norm(robot_pose[:2] - goal[:2]),
#             robot_pose[0] - goal[0],
#             robot_pose[1] - goal[1],
#             robot_pose[2],
#             robot_vel[0],
#             robot_vel[1],
#             robot_vel[2]
#         ])
#
#         obs_robot_global = robot_pose.copy()
#         obs_goal_global = goal.copy()
#
#         return {
#             "peds": obs_peds,
#             "peds_visibility": obs_peds_vis,
#             "peds_prediction": obs_peds_prediction,
#             "peds_prediction_cov": obs_peds_prediction_cov,
#             "robot": obs_robot,
#             "robot_global": obs_robot_global,
#             "goal_global": obs_goal_global
#         }
#
#     def _build_obs(self) -> Dict[str, np.ndarray]:
#         goal = self._sim_wrap.goal
#         robot_pose = self._sim_wrap.sim_state.world.robot.pose
#         robot_vel = self._sim_wrap.sim_state.world.robot.velocity
#         current_poses = self._sim_wrap.ped_tracker.get_current_poses()
#         predictions = {k: v[0] for k, v in self._sim_wrap.ped_tracker.get_predictions().items()}
#
#         if self._obs_mode != "v_learning":
#             robot_obs = SocialNavGraphEnv._build_robot_obs(robot_pose, robot_vel, goal)
#             obs_ped_traj, obs_peds_vis = self._build_peds_obs(robot_pose, current_poses, predictions) \
#                 if self._obs_mode == "prediction" else self._build_peds_obs_current(robot_obs, current_poses)
#
#             return {
#                 "peds_traj": obs_ped_traj,
#                 "peds_visibility": obs_peds_vis,
#                 "robot_state": robot_obs
#             }
#         return self._build_obs_v_learning()
#
#
# @nip
# class SocialNavGraphEnvFactory(AbstractEnvFactory):
#
#     def __init__(self,
#                  action_space_config: AbstractActionSpaceConfig,
#                  sim_config: SimConfig,
#                  curriculum: AbstractCurriculum,
#                  tracker_factory: Callable,
#                  reward: AbstractReward,
#                  peds_padding: int,
#                  rl_tracker_horizon: int,
#                  controller_factory: Optional[AbstractControllerFactory] = None,
#                  obs_mode: str = "prediction",
#                  n_stacks: Optional[Union[int, Dict[str, int]]] = None):
#         self._action_space_config = action_space_config
#         self._sim_config = sim_config
#         self._curriculum = curriculum
#         self._ped_tracker_factory = tracker_factory
#         self._reward = reward
#         self._peds_padding = peds_padding
#         self._rl_tracking_horizon = rl_tracker_horizon
#         self._controller_factory = controller_factory
#         self._obs_mode = obs_mode
#         self._n_stacks = n_stacks
#
#     def __call__(self, is_eval: bool) -> SocialNavGraphEnv:
#         controller = self._controller_factory() if self._controller_factory is not None else None
#         env = SocialNavGraphEnv(action_space_config=self._action_space_config,
#                                 sim_config=self._sim_config,
#                                 curriculum=self._curriculum,
#                                 ped_tracker=self._ped_tracker_factory(),
#                                 reward=self._reward,
#                                 peds_padding=self._peds_padding,
#                                 rl_tracker_horizon=self._rl_tracking_horizon,
#                                 controller=controller,
#                                 obs_mode=self._obs_mode,
#                                 is_eval=is_eval)
#         if self._n_stacks is not None:
#             env = StackHistoryWrapper(env, self._n_stacks)
#         return env
