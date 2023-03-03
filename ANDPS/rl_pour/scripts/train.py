import gym
import dartpy
from stable_baselines3 import SAC as algo
from stable_baselines3.common.noise import NormalActionNoise
from utils_sb3 import SACDensePolicy
from utils import damped_pseudoinverse
import numpy as np
import RobotDART as rd
import matplotlib.pyplot as plt
import scipy.stats
from typing import Callable
def EREulerXYZ(eulerXYZ):
    x = eulerXYZ[0]
    y = eulerXYZ[1]
    # z = eulerXYZ[2]
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)

    R = np.zeros((3, 3))
    R[0, 0] = 1.
    R[0, 2] = sin_y
    R[1, 1] = cos_x
    R[1, 2] = -cos_y * sin_x
    R[2, 1] = sin_x
    R[2, 2] = cos_x * cos_y

    return R
MAX_STEPS = 300
EPOCHS = 1000
class PourEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_graphics=True, enable_record=True, seed=-1, dt=0.01):
        self.setup_simu(dt)
        if (enable_graphics):
            self.setup_graphics(enable_record)
        self.setup_env()

        # cartesianVel, eulerXYZ
        # self.state = np.array([[0.,0.,0.,0.,0.,0.]])
        self.seed = seed
        self.it = 0
        self.max_steps = MAX_STEPS

        # define action space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([-2., -2., -2., -np.pi, -np.pi/2, -np.pi]), high=np.array([2., 2., 2., np.pi, np.pi/2, np.pi]), shape=(6,), dtype=np.float32)

    def step(self, action):
        self.simu.step_world()
        observation = 0
        reward = 0
        done = False

        vel_rot = EREulerXYZ(self.get_state()[:3]) @ action[3:]

        jac_pinv = damped_pseudoinverse(self.robot.jacobian(self.eef_link_name))
        cmd = jac_pinv @ np.append(vel_rot,action[:3])
        self.robot.set_commands(cmd)
        self.simu.step_world()

        observation = self.get_state()
        reward = self.calc_reward()

        if(self.it == self.max_steps):
            done = True
            self.it = -1


        self.it += 1
        return observation, reward, done, {}

    def get_state(self):
        poseXYZ = self.robot.body_pose(self.eef_link_name).translation()
        eulerXYZ = dartpy.math.matrixToEulerXYZ(self.robot.body_pose(self.eef_link_name).matrix()[:3, :3])
        return np.append(poseXYZ,eulerXYZ)

    def render(self):
        print(self.get_state())

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
        tf = dartpy.math.Isometry3()
        tf.set_translation(self.robot.body_pose(
            self.eef_link_name).translation() + [0.0, 0.065, 0.05])
        self.cereal_box.set_base_pose(tf)

    def setup_bowl(self):
        bowl_packages = [("bowl", "urdfs/bowl")]
        self.bowl = rd.Robot("urdfs/bowl/bowl.urdf",  bowl_packages, "bowl")
        self.bowl.set_color_mode("material")
        self.simu.add_robot(self.bowl)
        self.reset_bowl()

    def reset_bowl(self):
        tf = self.bowl.base_pose()
        tf.set_translation(self.get_state()[:3] + [0, 0, -0.65])
        self.bowl.set_base_pose(tf)

    def add_cereal(self, count=5):
        self.cereal_arr = []
        box_tf = self.cereal_box.base_pose()
        cereal_dims = [0.02, 0.02, 0.02]
        cereal_mass = 0.001
        cereal_color = [0.96, 0.82, 0.24, 1.]
        for i in range(count):
            cereal_pose = [0., 0., 0., box_tf.translation()[0], box_tf.translation()[1] - 0.05 + (i % 2) / 10, box_tf.translation()[2]-0.01 + i/100 + 0.018]
            cereal = rd.Robot.create_box(cereal_dims, cereal_pose, "free", mass=cereal_mass, color=cereal_color, box_name="cereal " + str(i))
            self.cereal_arr.append(cereal)
            self.simu.add_robot(cereal)

    def reset_cereal(self):
        box_tf = self.cereal_box.base_pose()
        for i in range(len(self.cereal_arr)):
            cereal_pose = [0., 0., 0., box_tf.translation()[0], box_tf.translation()[1] - 0.05 + (i % 2) / 10, box_tf.translation()[2]-0.01 + i/100 + 0.018]
            self.cereal_arr[i].set_base_pose(cereal_pose)

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
        self.reset_robot()
        self.reset_cereal_box()
        self.reset_cereal()
        self.reset_bowl()
        print("Env reset successfully")
        return self.get_state()

    def calc_reward(self):
        # reward is the sum of distances of every "cereal" to the bowl
        reward = 0
        for cereal in (self.cereal_arr):
            reward += np.linalg.norm(self.bowl.base_pose().translation() - cereal.base_pose().translation())

        return -reward * reward

env = PourEnv()




model = algo(SACDensePolicy, env, verbose=1, learning_rate=0.001)#, train_freq=int(MAX_STEPS/2), gradient_steps=200, batch_size=256, learning_starts=256)#, action_noise=NormalActionNoise(0., 1.))
# model = algo("MlpPolicy", env, verbose=1, learning_rate=5e-4)#, train_freq=MAX_STEPS, gradient_steps=200, batch_size=256, learning_starts=256)#, action_noise=NormalActionNoise(0., 1.))
# model = algo.load("reach_counter")
# model.set_env(env)
# model.learning_rate = 5e-4
model.learn(total_timesteps = 800 * EPOCHS)
model.save("cereal_killer")


# obs = env.reset()
# for i in range(env.max_steps):
#     action, _state = model.predict(obs, deterministic=True)
#     print(action, _state)
#     obs, reward, done, info = env.step(action.reshape(6,))
#     # if i == 0:
#         # env.render()
#     if done:
#         break