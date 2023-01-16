import gym

import numpy as np
import dartpy
from stable_baselines3 import SAC as algo

from utils_sb3 import SACDensePolicy
import RobotDART as rd


class TiagoEnv(gym.Env):

    def __init__(self, enable_graphics=False, seed=-1, dt=0.001):
        super(TiagoEnv, self).__init__()

        # x, y, theta
        self.state = np.array([[0., 0.]])
        self.seed = seed
        self.dt = dt
        self.it = 0
        self.max_steps = 1000
        self.bounds = 5.

        # Define actions and observation space
        self.action_space = gym.spaces.Box(low=-20., high=20., shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-self.bounds, high=self.bounds, shape=(1, 2), dtype=np.float32)

        # init robot dart
        self.simu = rd.RobotDARTSimu(self.dt)
        self.simu.add_checkerboard_floor(10.,10.)
        self.simu.set_collision_detector("bullet")
        if (enable_graphics):
            graphics = rd.gui.Graphics()
            self.simu.set_graphics(graphics)
            graphics.look_at([0.5, 5., 0.75], [0.5, 0., 0.2])
            graphics.record_video("Talos.mp4")
        # configure robot
        self.robot = rd.Tiago()
        self.simu.add_robot(self.robot)

        # set robot initial pose
        self.reset_robot()

        # Control base - make the base fully controllable
        self.robot.set_actuator_type("servo", "rootJoint", False, True, False)

        # set target
        self.target = np.array([[-1., 0., 0.]])

        # add target visualization
        target_tf = dartpy.math.Isometry3()
        target_tf.set_translation(self.target[0, 0:3])
        self.simu.add_visual_robot(rd.Robot.create_ellipsoid(
            [0.4, 0.4, 0.4], target_tf, 'fixed', 1.0, [0.0, 1.0, 0.0, 1.0], "target"))

    def step(self, action):

        self.it += 1
        self.robot.set_commands(action, ['rootJoint_pos_x', 'rootJoint_pos_y'])
        self.simu.step_world()
        self.state = self.robot.positions(['rootJoint_pos_x', 'rootJoint_pos_y'])
        observation = self.state.copy()

        err_pos = np.linalg.norm(observation[0:2] - self.target[0, 0:2])

        reward = - err_pos
        done = False
        if (np.abs(reward) < 1e-2 or self.it == self.max_steps):
            done = True
            self.it = -1
        return observation, reward, done, {}

    def reset(self):

        self.it = 0
        self.reset_robot()

        return self.observation_space.sample()

    def reset_robot(self):
        self.robot.reset()
        tf = self.robot.base_pose()
        translation = tf.translation()
        translation[0] = 2.5

        tf.set_translation(translation)
        self.robot.set_base_pose(tf)
        self.state = self.robot.positions(
            ['rootJoint_pos_x', 'rootJoint_pos_y'])

    def render(self):
        print(self.state)

env = TiagoEnv(enable_graphics=True)


model = algo(SACDensePolicy, env, verbose=1)
model.learn(total_timesteps=100000)
model.save("True")
# model = algo.load("simple_sac.zip")

obs = env.reset()
for i in range(env.max_steps):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
