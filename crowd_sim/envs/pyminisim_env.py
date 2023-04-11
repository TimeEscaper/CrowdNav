import logging
from typing import Optional, Union, Tuple, Dict, List

import numpy as np

import gym
from gym.core import ObsType, ActType

from pms.core import Simulation, WorldState
from pms.robot import SimpleHolonomicRobotModel
from pms.pedestrians import HeadedSocialForceModelPolicy, RandomWaypointTracker
from pms.sensors import PedestrianDetectorNoise, PedestrianDetector
from pms.visual import Renderer, CircleDrawing
from pms.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS

from crowd_sim.envs.utils.state import ObservableState, FullState
from crowd_sim.envs.utils.info import *


class PyMiniSimEnv(gym.Env):
    _SIM_DT = 0.01

    def __init__(self, render: bool = False):
        super(PyMiniSimEnv, self).__init__()

        self._render = render

        self.config: Optional = None
        self.time_limit: Optional[int] = None
        self.time_step: Optional[float] = None
        self.success_reward: Optional[float] = None
        self.collision_penalty: Optional[float] = None
        self.discomfort_dist: Optional[float] = None
        self.discomfort_penalty_factor: Optional[float] = None
        self.human_num: Optional[int] = None
        self.global_time: Optional[int] = None
        self.human_times: Optional[List[int]] = None
        self.robot: Optional = None
        self.case_capacity: Optional[Dict] = None
        self.case_size: Optional[Dict] = None

        self._sim: Optional[Simulation] = None
        self._renderer: Optional[Renderer] = None

        self._current_goal: Optional[np.ndarray] = None

        self._parallel_sim: Simulation = None
        self._pedestrians_traj_with_robot: List[np.ndarray] = []
        self._pedestrians_traj_no_robot: List[np.ndarray] = []

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        # if self.config.get('humans', 'policy') == 'orca':
        #     self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        #     self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
        #                       'test': config.getint('env', 'test_size')}
        #     self.train_val_sim = config.get('sim', 'train_val_sim')
        #     self.test_sim = config.get('sim', 'test_sim')
        #     self.square_width = config.getfloat('sim', 'square_width')
        #     self.circle_radius = config.getfloat('sim', 'circle_radius')
        #     self.human_num = config.getint('sim', 'human_num')
        # else:
        #     raise NotImplementedError
        self.human_num = config.getint('sim', 'human_num')
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                          'test': config.getint('env', 'test_size')}

        # logging.info('human number: {}'.format(self.human_num))
        # if self.randomize_attributes:
        #     logging.info("Randomize human's radius and preferred speed")
        # else:
        #     logging.info("Not randomize human's radius and preferred speed")
        # logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        # logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def reset(self, phase='test', test_case=None) -> Union[ObsType, tuple[ObsType, dict]]:
        self._init_sim()
        self._pedestrians_traj_with_robot.append(self._sim.current_state.world.pedestrians.poses[:, :2])
        self._pedestrians_traj_no_robot.append(self._parallel_sim.current_state.world.pedestrians.poses[:, :2])
        self._sample_goal()
        self.robot.set(self._sim.current_state.world.robot.pose[0],
                       self._sim.current_state.world.robot.pose[1],
                       self._current_goal[0],
                       self._current_goal[1],
                       0,
                       0,
                       self._sim.current_state.world.robot.pose[2])
        self.global_time = 0
        self.robot.time_step = self.time_step
        self.robot.policy.time_step = self.time_step
        return self._get_observation()

    def step(self, action, update=True) -> Tuple[ObsType, float, bool, dict]:
        backup_state = self._sim.current_state.world
        backup_parallel_state = self._parallel_sim.current_state.world

        control = np.array([action[0], action[1], 0.])
        elapsed_time = 0.
        collision = False
        while (not collision) and elapsed_time < self.time_step:
            self._sim.step(control)
            self._parallel_sim.step()
            elapsed_time += PyMiniSimEnv._SIM_DT
            collisions_info = self._sim.current_state.world.robot_to_pedestrians_collisions
            if collisions_info is not None and len(collisions_info) > 0:
                collision = True
            if self._render:
                self._renderer.render()
        self._pedestrians_traj_with_robot.append(self._sim.current_state.world.pedestrians.poses[:, :2])
        self._pedestrians_traj_no_robot.append(self._parallel_sim.current_state.world.pedestrians.poses[:, :2])

        reaching_goal = np.linalg.norm(
            self._sim.current_state.world.robot.pose[:2] - self._current_goal) < ROBOT_RADIUS

        dmin = np.min(np.linalg.norm(self._sim.current_state.world.robot.pose[:2] -
                                     self._sim.current_state.world.pedestrians.poses[:, :2]))

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward + self._calculate_social_disturbance()
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        obs = self._get_observation()

        if not update:
            self._sim.reset_to_state(backup_state)
            self._parallel_sim.reset_to_state(backup_parallel_state)
        else:
            self.robot.step(action)
            self.global_time += self.time_step

        return obs, reward, done, info

    def render(self, mode: str = "human", output_file: Optional[str] = None):
        pass

    def _init_sim(self):
        self._sim = self._create_sim()
        if self._render:
            self._renderer = Renderer(simulation=self._sim,
                                      resolution=80.0,
                                      screen_size=(500, 500))
        self._parallel_sim = self._create_sim()
        self._parallel_sim.reset_to_state(self._sim.current_state.world)
        self._parallel_sim.set_robot_enabled(False)

    def _create_sim(self) -> Simulation:
        tracker = RandomWaypointTracker(world_size=(7.0, 7.0))
        pedestrians_model = HeadedSocialForceModelPolicy(n_pedestrians=self.human_num,
                                                         waypoint_tracker=tracker,
                                                         pedestrian_linear_velocity_magnitude=1.5)
        robot_model = SimpleHolonomicRobotModel(initial_pose=np.array([2.0, 3.85, 0.]),
                                                initial_control=np.array([0., 0., 0.]))
        return Simulation(robot_model=robot_model,
                          pedestrians_model=pedestrians_model,
                          sensors=[],
                          sim_dt=PyMiniSimEnv._SIM_DT)

    def _get_observation(self):
        assert self._sim is not None
        humans_states = np.concatenate([self._sim.current_state.world.pedestrians.poses[:, :2],
                                        self._sim.current_state.world.pedestrians.velocities[:, :2]], axis=1)
        return [ObservableState(e[0], e[1], e[2], e[3], PEDESTRIAN_RADIUS) for e in humans_states]

    def _sample_goal(self, agent_position: Optional[np.ndarray] = None) -> None:
        if agent_position is None:
            agent_position = self._sim.current_state.world.robot.pose[:2]

        for _ in range(100):
            sampled_point = np.random.uniform(low=np.array([0., 0.]),
                                              high=np.array([7., 7.]))
            if np.linalg.norm(agent_position - sampled_point) < 3.:
                continue
            self._current_goal = sampled_point
            if self._renderer is not None:
                self._renderer.clear_drawings()
                self._renderer.draw("goal", CircleDrawing(self._current_goal, 0.05, (255, 0, 0)))
            return
        raise RuntimeError("Failed to sample waypoint")

    def _calculate_social_disturbance(self) -> float:
        traj_with_robot = np.array(self._pedestrians_traj_with_robot)
        traj_no_robot = np.array(self._pedestrians_traj_no_robot)
        social_disturbance = np.mean((traj_with_robot - traj_no_robot) ** 2)
        print(f"Social disurbance: {social_disturbance}")
        return -10. * social_disturbance
