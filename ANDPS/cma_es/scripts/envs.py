import gym
import numpy as np
import dartpy
import RobotDART as rd
from utils import EREulerXYZ, damped_pseudoinverse
MAX_STEPS = 1000


np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))

# wrap the angle between -pi and pi


def angle_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PendulumEnv(gym.Env):
    def __init__(self, dt=0.001, gui=False):
        super(PendulumEnv).__init__()
        self.name = "PendulumEnv"

        # the observation space is the angle of the joint
        self.observation_space = gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(1,))

        # action space is the torque applied to the joint
        self.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,))
        self.it = 0
        self.MAX_STEPS = 1000
        #### RobotDART ####
        self.robot = rd.Robot("assets/pendulum.urdf")
        self.robot.fix_to_world()
        self.robot.set_actuator_types("torque")
        self.robot.set_position_enforced(False)
        self.dt = dt
        self.simu = rd.RobotDARTSimu(self.dt)
        self.simu.add_robot(self.robot)
        self.seed = 0
        if (gui):
            self.gconfig = rd.gui.Graphics.default_configuration()
            self.gconfig.width = 640
            self.gconfig.height = 480
            self.graphics = rd.gui.Graphics(self.gconfig)
            self.simu.set_graphics(self.graphics)
            self.graphics.look_at((1.5, 1.5, 2.), (0.0, 0.0, 0.5))

    def _step(self, action):
        self.it += 1
        obs = angle_wrap(self.robot.positions())
        # print(obs)
        reward = 0
        done = False

        self.robot.set_forces(action)
        # input()
        self.simu.step_world()

        reward = -obs[0]**2 + 0.1*action[0]**2
        if (self.it == self.MAX_STEPS):
            self.it = 0
            done = True
        return obs, reward, done

    def _reset(self):
        # print("Reset")
        self.robot.reset()
        # starting state is a random angle in [-pi, pi]
        rand_angle = np.random.uniform(-np.pi, np.pi)
        self.robot.set_positions([rand_angle])
        # self.robot.set_velocities([0.0])
        # self.robot.set_forces([0.0])
        return angle_wrap(self.robot.positions())


displacements = [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.0],
                [-0.02, 0.3, 0.0],
                [-0.1, -0.2, 0.0],
                [0.05, -0.03, 0.0],
                [0.2, -0.5, 0.0],
]


class PourEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, enable_graphics=False, enable_record=False, seed=-1, dt=0.01):
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
        self.action_space = gym.spaces.Box(low=np.array(
            [-1, -1, -1, -1, -1, -1]), high=np.array([1, 1, 1, 1, 1, 1]), shape=(6,), dtype=np.float32)
        self.low_bounds = [self.table.base_pose().translation(
        )[0]-1.5, self.table.base_pose().translation()[1]-1., self.table.base_pose().translation()[2]+0.35]
        self.high_bounds = [self.table.base_pose().translation()[0]+1.5, self.table.base_pose(
        ).translation()[1]+1., self.robot.body_pose("panda_ee").translation()[2]+1.]
        self.observation_space = gym.spaces.Box(low=np.array([np.float32(np.float32(-np.pi)), np.float32(-np.pi/2), np.float32(-np.pi), np.float32(self.low_bounds[0]), np.float32(
            self.low_bounds[1]), np.float32(self.low_bounds[2])]), high=np.array([np.pi, np.pi/2, np.pi, self.high_bounds[0], self.high_bounds[1], self.high_bounds[2]]), shape=(6,), dtype=np.float32)

    def step(self, action):
        self.simu.step_world()
        # print("_"*10)
        # print(action)
        # print("_"*10)
        observation = 0
        reward = 0
        done = False

        eulerXYZ = self.get_state()[:3]
        vel_rot = EREulerXYZ(eulerXYZ) @ action[0, :3]
        jac_pinv = damped_pseudoinverse(
            self.robot.jacobian(self.eef_link_name))
        # print("_"*10)
        # print(action)
        # print(vel_rot)
        # print(np.append(vel_rot,action[:3]))
        # print("_"*10)
        cmd = jac_pinv @ np.append(vel_rot, action[0, :3])
        self.robot.set_commands(cmd)
        self.simu.step_world()

        observation = self.get_state()
        reward = self.calc_reward()

        if (self.it == self.max_steps) or any(observation[3:6] > self.high_bounds) or any(observation[3:6] < self.low_bounds):
            done = True
            self.it = -1

        self.it += 1
        return observation, reward, done, {}

    def get_state(self):
        poseXYZ = self.robot.body_pose(self.eef_link_name).translation()
        eulerXYZ = dartpy.math.matrixToEulerXYZ(
            self.robot.body_pose(self.eef_link_name).matrix()[:3, :3])
        bowlXYZ = self.bowl.base_pose().translation()
        return np.append(np.append(eulerXYZ, poseXYZ), bowlXYZ)

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
        self.graphics.look_at((1.5, 1.5, 2.), (0.0, 0.0, 0.5))
        if (enable_record):
            self.graphics.camera().record(True)
            self.graphics.record_video(
                "cerial-env_2.mp4", self.simu.graphics_freq())

    def setup_table(self):
        # table_packages = [("table", "urdfs/table")]
        # self.table = rd.Robot("urdfs/table/table.urdf",   tabie_packages, "table")
        # self.table.set_color_mode("material")
        table_dims = [3., 2., 0.7]
        table_pose = [0, 0, 0, 0, 0, 0.35]
        table_color = [0.933, 0.870, 0.784, 0.]
        self.table = rd.Robot.create_box(
            table_dims, table_pose, "fix", mass=30., color=table_color, box_name="table")
        self.table.fix_to_world()
        self.simu.add_robot(self.table)

    def setup_robot(self):
        franka_packages = [
            ("franka_description", "urdfs/franka/franka_description")]
        self.robot = rd.Robot("urdfs/franka/franka.urdf",
                              franka_packages, "franka")
        self.robot.set_color_mode("material")
        self.simu.add_robot(self.robot)
        self.eef_link_name = "panda_ee"
        tf = self.robot.base_pose()
        tf.set_translation([-0.7, 0, 0.7])
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
        # positions[7] = 0.015
        # positions[8] = 0.015
        self.robot.set_positions(positions)

    # def setup_cereal_box(self):
    #     cereal_packages = [("cereal", "urdfs/cereal")]
    #     self.cereal_box = rd.Robot(
    #         "urdfs/cereal/cereal.urdf",  cereal_packages, "cereal_box")
    #     self.cereal_box.set_color_mode("material")
    #     self.simu.add_robot(self.cereal_box)
    #     self.reset_cereal_box()

    # def reset_cereal_box(self):
    #     tf = dartpy.math.Isometry3()
    #     tf.set_translation(self.robot.body_pose(
    #         self.eef_link_name).translation() + [0.0, 0.065, 0.05])
    #     self.cereal_box.set_base_pose(tf)

    def setup_bowl(self):
        bowl_packages = [("bowl", "urdfs/bowl")]
        self.bowl = rd.Robot("urdfs/bowl/bowl.urdf",  bowl_packages, "bowl")
        self.bowl.set_color_mode("material")
        self.simu.add_robot(self.bowl)
        self.reset_bowl()

    def reset_bowl(self):
        tf = self.bowl.base_pose()
        # add some noise to the bowl position with a 50% chance
        # if np.random.random() > 0.5:
        # pick a random integer between 0 and 5
        idx = np.random.randint(0, 5)
        print(idx)
        translation_noise = displacements[idx]
        # else:
        # translation_noise = np.zeros(3)
        tf.set_translation(self.table.base_pose().translation() +
                           [-0.3, 0.2, 0.37] + translation_noise)
        self.bowl.set_base_pose(tf)
        self.bowl.fix_to_world()

    def add_cereal(self, count=4):
        box_tf = self.robot.body_pose("panda_ee")
        self.cereal_arr = []

        cereal_dims = [0.02, 0.02, 0.02]
        cereal_mass = 1
        cereal_color = [1, 0., 0., 1.]

        for i in range(count):
            cereal_pose = [0., 0., 0., box_tf.translation()[0], box_tf.translation(
            )[1] - 0.06 + (i % 2)*0.01, box_tf.translation()[2] + (i % (2+1)) * 0.001]
            # cereal = rd.Robot.create_ellipsoid(cereal_dims, cereal_pose, "free", mass=cereal_mass, color=cereal_color, ellipsoid_name="cereal_" + str(i))
            cereal = rd.Robot.create_box(
                cereal_dims, cereal_pose, "free", mass=cereal_mass, color=cereal_color, box_name="cereal " + str(i))
            self.cereal_arr.append(cereal)
            self.simu.add_robot(cereal)

    def reset_cereal(self):
        box_tf = self.robot.body_pose("cereal_box")
        for i in range(len(self.cereal_arr)):
            cereal_pose = [0., 0., 0., box_tf.translation()[0], box_tf.translation()[
                1]-0.05, box_tf.translation()[2] + i * 0.015]
            self.cereal_arr[i].reset()
            self.cereal_arr[i].set_base_pose(cereal_pose)

    def setup_env(self):
        print("Initializing Rd Environment")
        self.setup_table()
        print("Added table")
        self.setup_robot()
        print("Added robot")
        # self.setup_cereal_box()
        # print("Added cereal box")
        self.add_cereal()
        print("Added cereal")
        self.setup_bowl()
        print("Added bowl")

    def reset(self):
        self.reset_robot()
        # self.reset_cereal_box()
        self.reset_cereal()
        self.reset_bowl()
        # print("Env reset successfully")
        for _ in range(100):
            self.simu.step_world()
        return self.get_state()

    def calc_reward(self):
        reward = 0
        p = 0.2

        reward_rot = np.cos(angle_wrap(-np.pi/2 - dartpy.math.matrixToEulerXYZ(
            self.robot.body_pose("panda_ee").rotation())[2]))
        # print("Reward rot:", reward_rot)
        reward_pos = np.exp(-0.5 * np.linalg.norm(self.bowl.base_pose().translation()[
                            :2] - self.robot.body_pose("panda_ee").translation()[:2])/(p**2))
        for cereal in self.cereal_arr:
            reward += np.exp(-0.5 * np.linalg.norm(self.bowl.base_pose(
            ).translation() - cereal.base_pose().translation())/(p**2))
            # reward+= int(box_into_bowl(cereal.base_pose().translation(),self.bowl.base_pose().translation(), dartpy.math.matrixToEulerXYZ(self.bowl.base_pose().rotation())[2]))
        return reward + reward_rot + reward_pos

    def get_limits(self):

        # lim_eX = [-np.pi, np.pi]
        # lim_eY = [-np.pi/2, np.pi/2]
        # lim_eZ = [-np.pi, np.pi]
        # lim_z = [self.robot.base_pose().translation()[2], self.robot.base_pose().translation()[0] + 1.190]
        # lim_x = [self.robot.base_pose().translation()[0]-0.855, self.robot.base_pose().translation()[1] + 0.855]
        # lim_y = [self.robot.base_pose().translation()[1]-0.855, self.robot.base_pose().translation()[2] + 0.855]

        # I want to write the lims in the form y= Ax + b
        # x has to be 1 value and not a tuble, and as a result i want the limits to be symmetrical so:

        b_eX = 0
        b_eY = 0
        b_eZ = 0
        b_z = self.robot.base_pose().translation()[2] + 1.190/2
        b_x = self.robot.base_pose().translation()[0]
        b_y = self.robot.base_pose().translation()[1]

        A_eX = np.pi  # * (-1,1)
        A_eY = np.pi/2
        A_eZ = np.pi
        A_z = 1.190/2
        A_x = 0.855
        A_y = 0.855
        return np.array([[b_eX, b_eY, b_eZ, b_x, b_y, b_z], [A_eX, A_eY, A_eZ, A_x, A_y, A_z]]).copy()

    def viz_target(self, position):
        tf = dartpy.math.Isometry3()
        tf.set_translation(position)
        target_robot = rd.Robot.create_box([0.1, 0.1, 0.1], tf, "fixed", mass=0.1, color=[
                                           0, 1., 0., 1.], box_name="target")
        self.simu.add_robot(target_robot)
