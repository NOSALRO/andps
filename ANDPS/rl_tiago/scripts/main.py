import gym
import dartpy
from stable_baselines3 import SAC as algo
from utils_sb3 import SACDensePolicy
import numpy as np
import RobotDART as rd


class TiagoEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_graphics=False, seed=-1, dt=0.01):
        super(TiagoEnv, self).__init__()

        # x, y, theta
        self.state = np.array([[0., 0.]])

        self.seed = seed
        self.dt = dt
        self.it = 0
        self.max_steps = 500
        self.bounds = 5.

        # Define actions and observation space
        self.action_space = gym.spaces.Box(low=-2., high=2., shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-self.bounds, high=self.bounds, shape=(2,), dtype=np.float32)

        # init robot dart
        self.simu = rd.RobotDARTSimu(self.dt)
        self.simu.add_checkerboard_floor(10., 10.)
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
        self.target = np.array([[-1., 0.]])

        # add target visualization
        target_tf = dartpy.math.Isometry3()
        target_tf.set_translation([-1., 0., 0.])
        self.simu.add_visual_robot(rd.Robot.create_ellipsoid([0.4, 0.4, 0.4], target_tf, 'fixed', 1.0, [0.0, 1.0, 0.0, 1.0], "target"))

    def step(self, action):

        self.it += 1
        self.robot.set_commands(action, ['rootJoint_pos_x', 'rootJoint_pos_y'])

        self.simu.step_world()
        self.state = self.robot.base_pose().translation()[0:2]
        observation = self.state.copy()

        dist = np.linalg.norm(self.target-observation)
        # dist = np.inner(diff, diff)[0][0]
        # p = 0.2
        reward = -dist  # np.exp(-0.5*dist/(p*p))[0]
        done = False

        # penalize large actions
        # reward+= -0.1*np.linalg.norm(action)
        # if(dist < 0.1) :
        #     reward = 1000
        if (self.it == self.max_steps) or (abs(self.state[0]) >= self.bounds or abs(self.state[1]) >= self.bounds):
            self.it == -1
            done = True
        return observation, reward, done, {}

    def reset(self):

        self.it = 0
        self.reset_robot()
        return self.state

    def reset_robot(self):
        self.robot.reset()
        tf = self.robot.base_pose()
        self.robot.set_position_enforced(True)
        translation = tf.translation()
        translation[0] = 2.5
        self.robot.set_positions([0], ['rootJoint_rot_z'])
        tf.set_translation(translation)
        self.robot.set_base_pose(tf)
        self.state = self.robot.base_pose().translation()[0:2].reshape(1,2)

    def render(self):
        print(self.state)


env = TiagoEnv(enable_graphics=True)


model = algo(SACDensePolicy, env, verbose=1)
model.learn(total_timesteps=100000)
model.save("True")
# model = algo.load("True.zip")

obs = env.reset()
for i in range(env.max_steps):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
