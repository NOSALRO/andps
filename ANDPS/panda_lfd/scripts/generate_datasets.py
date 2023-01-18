import numpy as np
import RobotDART as rd
from utils import PIDTask
from utils import damped_pseudoinverse


enable_graphics = True # set to False if you want to run the script without graphics

# Get Joint Positions (Angles)
def get_joints_state(robot):
    return robot.positions()[:7]

# get end effector position (xyz)
def get_eef_state(robot):
    return robot.body_pose(eef_link_name).translation()

# Set Franka to Initial Configuration
def set_initial_pos(robot):
    robot.reset()
    positions = robot.positions()
    positions[5] = np.pi / 2.0
    positions[6] = np.pi / 4.0
    positions[7] = 0.04
    positions[8] = 0.04
    robot.set_positions(positions)



# RobotDART Simulation
dt = 0.001
control_freq = 100
simu = rd.RobotDARTSimu(dt)
simu.set_collision_detector("fcl")
simu.add_checkerboard_floor()

# RobotDART Graphics
if(enable_graphics):
    gconfig = rd.gui.Graphics.default_configuration()
    gconfig.width = 640
    gconfig.height = 480
    graphics = rd.gui.Graphics(gconfig)
    simu.set_graphics(graphics)
    graphics.look_at((2.5, 0.0, 1.), (0.0, 0.0, 0.5))
    graphics.camera().record(True)
    simu.enable_status_bar(enable=False)
    graphics.record_video("media/videos/spiral_demo.mp4")



# RobotDART Robot
robot = rd.Franka()
robot.fix_to_world()
robot.set_position_enforced(True)
robot.set_actuator_types("servo")
eef_link_name = "panda_ee"
simu.add_robot(robot)

set_initial_pos(robot)


# Controller for fixed eef orientation
Kp = np.array([20., 20., 20., 10., 10., 10.])
Kd = Kp * 0.01
Ki = Kp * 0.1
controller = PIDTask(1./control_freq, Kp, Ki, Kd)
controller.set_target(robot.body_pose(eef_link_name))




# Initialize datasets
joints_state_cur = get_joints_state(robot)
joints_dataset = np.array(joints_state_cur)

eef_state_cur = get_eef_state(robot)
eef_trajectory = np.array(eef_state_cur)

# helper variables
timestep_counter = 0
t = 0
leave_trace = 0
continue_simu = True

while(continue_simu):
    t += dt
    update = False
    if simu.scheduler().schedule(control_freq):
        update = True
        eef_tf = robot.body_pose(eef_link_name)
        vel_rot = controller.update(eef_tf)[0][:3]
        vel_xyz = np.zeros_like(vel_rot)
        vel_xyz[0] =  -0.15 * np.sin(1/2* np.pi * t/1)
        vel_xyz[1] = 0.07
        vel_xyz[2] =  -0.15 * np.cos(1/2 * np.pi * t/1)

        vel = np.concatenate((vel_rot, vel_xyz))

        # Jacobian pseudo-inverse
        jac_pinv = damped_pseudoinverse(robot.jacobian(eef_link_name))

        # Hacky way to slow down the robot at the end of the trajectory
        if(timestep_counter >= 550) and (timestep_counter < 700):
            a = (700 - timestep_counter)/150.
            vel = a * np.array(vel) + (1. - a) * np.zeros([6])
        elif (timestep_counter >= 700):
            vel = np.zeros([6])

        # Compute joint commands
        cmd = jac_pinv @ vel
        robot.set_commands(cmd)

        # Add spheres to visualize the trajectory in the GUI
        if(enable_graphics):
            leave_trace = leave_trace + 1
            if leave_trace == 1 or leave_trace % 10 == 0:
                simu.add_visual_robot(rd.Robot.create_ellipsoid([0.05, 0.05, 0.05], robot.body_pose(
                eef_link_name), "fix",  color=[0.8, 8, 123, 1], ellipsoid_name=str(leave_trace)))

        timestep_counter += 1

        if(timestep_counter == 720):
            continue_simu = False
    simu.step_world()
    if update:
        joints_state_cur = get_joints_state(robot)
        joints_dataset = np.vstack((joints_dataset, joints_state_cur))
        eef_trajectory = np.vstack((eef_trajectory, robot.body_pose(eef_link_name).translation()))


# print(joints_dataset.shape)

# compute velocity (x_t+1 - x_t)/dt
joints_x = joints_dataset[:-1, :]
joints_vx = control_freq * (joints_dataset[1:, :] - joints_dataset[:-1, :])

eef_x = eef_trajectory[:-1, :]
eef_vx = control_freq * (eef_trajectory[1:, :] - eef_trajectory[:-1, :])

joints_data = np.concatenate((joints_x, joints_vx), axis=1)
eef_data = np.concatenate((eef_x, eef_vx), axis=1)
print(joints_data.shape)
print(eef_data.shape)

joints_data = np.asarray(joints_data)
eef_data = np.asarray(eef_data)
np.save('data/spiral_joint_space.npy', joints_data)
np.save('data/spiral_eef_space.npy', eef_trajectory)

