import gym
import dartpy
from stable_baselines3 import SAC as algo
from stable_baselines3.common.noise import NormalActionNoise
from utils_sb3 import SACDensePolicy
import numpy as np
import RobotDART as rd
import matplotlib.pyplot as plt
import scipy.stats
from typing import Callable


class TiagoEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_graphics=True, enable_record=True, seed=-1, dt=0.01):
        self.setup_simu(dt)
        if (enable_graphics):
            self.setup_graphics(enable_record)
        self.setup_env()

    def step(self):
        self.simu.step_world()
        observation = 0
        reward = 0
        done = False
        return observation, reward, done, {}

    def get_state(self):
        return self.robot.body_pose(self.eef_link_name).translation()

    def render(self):
        if (self.enable_graphics):
            graphics = rd.gui.Graphics()
            self.simu.set_graphics(graphics)
            self.simu.scheduler().set_sync(False)
            graphics.look_at([0, 0., 10.75], [0., 0., 10.])
            graphics.record_video("Tiago.mp4")
        # print(self.state)

    def setup_simu(self, dt):
        self.dt = dt
        # init simu
        self.simu = rd.RobotDARTSimu(self.dt)
        self.simu.set_collision_detector("fcl")
        # add checkerboard floor
        self.simu.add_checkerboard_floor()

    def setup_graphics(self, enable_record):
        self.gconfig = rd.gui.Graphics.default_configuration()
        self.gconfig.width = 640
        self.gconfig.height = 480
        self.graphics = rd.gui.Graphics(self.gconfig)
        self.simu.set_graphics(self.graphics)
        self.graphics.look_at((1.5, 1.5, 4.), (0.0, 0.0, 0.5))
        if (enable_record):
            self.graphics.camera().record(True)
            self.graphics.record_video(
                "cerial-env.mp4", self.simu.graphics_freq())

    def setup_table(self):
        table_packages = [("table", "urdfs/table")]
        self.table = rd.Robot("urdfs/table/table.urdf",
                              table_packages, "table")
        self.table.set_color_mode("material")
        self.table.fix_to_world()
        self.simu.add_robot(self.table)

    def setup_robot(self):
        self.robot = rd.Franka()
        self.simu.add_robot(self.robot)
        self.eef_link_name = "panda_ee"
        tf = self.robot.base_pose()
        tf.set_translation([-0.7, 0, 0.78])
        self.robot.set_base_pose(tf)
        self.robot.fix_to_world()
        self.robot.set_position_enforced(True)
        self.robot.set_actuator_types("servo")
        self.reset_robot()

    def reset_robot(self):
        self.robot.reset()
        positions = self.robot.positions()
        positions[4] = np.pi / 2.0
        positions[5] = np.pi / 2.0
        positions[6] = 3*np.pi/4
        positions[7] = 0.022
        positions[8] = 0.022
        self.robot.set_positions(positions)

    def setup_cereal_box(self):
        cereal_packages = [("cereal", "urdfs/cereal")]
        self.cereal_box = rd.Robot(
            "urdfs/cereal/cereal.urdf",  cereal_packages, "cereal_box")
        self.cereal_box.set_color_mode("material")
        self.simu.add_robot(self.cereal_box)
        self.reset_cereal_box()

    def reset_cereal_box(self):
        tf = self.cereal_box.base_pose()
        tf.set_translation(self.robot.body_pose(
            self.eef_link_name).translation() + [0.0, 0.065, 0.05])
        self.cereal_box.set_base_pose(tf)

    def setup_bowl(self):
        bowl_packages = [("bowl", "urdfs/bowl")]
        self.bowl = rd.Robot("urdfs/bowl/bowl.urdf",  bowl_packages, "bowl")
        self.bowl.set_color_mode("material")
        self.reset_bowl()

    def reset_bowl(self):
        tf = self.bowl.base_pose()
        tf.set_translation(self.table.base_pose().translation() + [0, 0, 0.78])
        self.bowl.set_base_pose(tf)
        self.simu.add_robot(self.bowl)

    def add_cereal(self, count=5):
        self.cereal_arr = []
        box_tf = self.cereal_box.base_pose()
        cereal_dims = [0.03, 0.03, 0.03]
        cereal_mass = 0.001
        cereal_color = [0.96, 0.82, 0.24, 1.]
        for i in range(count):
            cereal_pose = [0., 0., 0., box_tf.translation()[0], box_tf.translation()[1] - 0.05 + (i % 2) / 10, box_tf.translation()[2]-0.01 + i/100 + 0.018]
            self.cereal_arr.append(self.simu.add_robot(rd.Robot.create_box(cereal_dims, cereal_pose, "free", mass=cereal_mass, color=cereal_color, box_name="cereal " + str(i))))

    def setup_env(self):
        print("Initializing Rd Environment")
        self.setup_table()
        print("Added table")
        self.setup_robot()
        print("Added robot")
        self.setup_cereal_box()
        print("Added cereal box")
        self.add_cereal()
        print("Added cereal")
        self.setup_bowl()
        print("Added bowl")

    def reset(self):
        yield


env = TiagoEnv()
for _ in range(3000):
    env.step()