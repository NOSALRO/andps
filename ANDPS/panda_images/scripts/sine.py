import RobotDART as rd
import numpy as np
import dartpy
from utils import *

# get end effector position (xyz)
def get_eef_state(robot, eef_link_name = "panda_ee"):
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

# convert rd image to grayscale array
def image_as_grayscale_array(rd_image):
    image = np.array(rd.gui.convert_rgb_to_grayscale(rd_image).data).reshape(rd_image.width, rd_image.height)
    return image

target = [0.40050368, 0.4042623,  0.52949236]

# RobotDART Simulation
dt = 0.001
control_freq = 100
simu = rd.RobotDARTSimu(dt)
simu.set_collision_detector("fcl")
simu.add_checkerboard_floor()



# RobotDART Robot
robot = rd.Franka()
robot.fix_to_world()
robot.set_position_enforced(True)
robot.set_actuator_types("servo")
eef_link_name = "panda_ee"
simu.add_robot(robot)
set_initial_pos(robot)


# box
sine_box_packages = [("sine_box", "urdfs/sine_box")]
sine_box = rd.Robot("urdfs/sine_box/sine_box.urdf",  sine_box_packages,"sine_box")
sine_box.set_color_mode("material")
sine_box.fix_to_world()
box_tf = sine_box.base_pose()
box_tf.set_translation([3.5, 0., 1.])
sine_box.set_base_pose(box_tf)
simu.add_robot(sine_box)



# graphics
gconfig = rd.gui.Graphics.default_configuration()
gconfig.width = 480
gconfig.height = 480
gconfig.shadowed = False
graphics = rd.gui.Graphics(gconfig)
simu.set_graphics(graphics)
simu.enable_status_bar(enable=False)
graphics.look_at((-2.5, 1.0, 1.), (0.0, 0.0, 0.5))
graphics.camera().record(True)


# record images from main camera/graphics
graphics.camera().record(True)
graphics.record_video("video-main.mp4", simu.graphics_freq())

# attach camera
camera = rd.sensor.Camera(graphics.magnum_app(), 64, 64)
camera.camera().record(True)
camera.record_video("video-camera.mp4")
tf = dartpy.math.Isometry3()
rot =  dartpy.math.AngleAxis(-np.pi/2, [0., 1., 0.])
rot = rot.multiply( dartpy.math.AngleAxis(1.57, [0., 0., 1.])).to_rotation_matrix()
tf.set_translation([0., 0., 0.1])
tf.set_rotation(rot)
camera.attach_to_body(robot.body_node("panda_ee"), tf) # cameras are looking towards -z by default
simu.add_sensor(camera)

# Controller for fixed eef orientation
Kp = np.array([20., 20., 20., 10., 10., 10.])
Kd = Kp * 0.01
Ki = Kp * 0.1
controller = PIDTask(1./control_freq, Kp, Ki, Kd)
controller.set_target(robot.body_pose(eef_link_name))

# visualize target
target_tf = dartpy.math.Isometry3()
target_tf.set_translation(target)
simu.add_visual_robot(rd.Robot.create_ellipsoid([0.05, 0.05, 0.05], target_tf, "fix",  color=[0.211, 0.596, 0.184, 1], ellipsoid_name="target"))

# warmstart simulation
for _ in range(10):
    simu.step_world()

# Initialize datasets
eef_trajectory = np.array(get_eef_state(robot))
images = np.array(image_as_grayscale_array(camera.image()))

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
        vel_xyz[0] = 1/2*( target[0] -eef_tf.translation()[0] )
        # if(timestep_counter<200):
        vel_xyz[1] = 0.12
        vel_xyz[2] =  -0.3 * np.sin(  np.pi * t/1)

        vel = np.concatenate((vel_rot, vel_xyz))

        # Jacobian pseudo-inverse
        jac_pinv = damped_pseudoinverse(robot.jacobian(eef_link_name))

         # Hacky way to slow down the robot at the end of the trajectory
        if(timestep_counter >= 350) and (timestep_counter < 700):
            # a = (700 - timestep_counter)/150.
            vel[3:] =  2*(target - robot.body_pose(eef_link_name).translation().reshape(3,))
        elif (timestep_counter >= 700):

            vel = np.zeros([6])


        # Compute joint commands
        cmd = jac_pinv @ vel
        robot.set_commands(cmd)

        timestep_counter += 1

        if(timestep_counter == 720):
            continue_simu = False
    simu.step_world()
    if update:
        eef_trajectory = np.vstack((eef_trajectory, robot.body_pose(eef_link_name).translation()))
        images = np.dstack((images, image_as_grayscale_array(camera.image())))


eef_x = eef_trajectory[:-1, :]
eef_vx = control_freq * (eef_trajectory[1:, :] - eef_trajectory[:-1, :])

eef_data = np.concatenate((eef_x, eef_vx), axis=1)
images_flat = images.copy().reshape(64*64,-1).T

np_X = np.concatenate([eef_x,images_flat[1:,:]],axis = 1)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(eef_x[:,0], eef_x[:,1], eef_x[:,2])
# plt.show()
np.savez('data/sine.npz', eef_x =eef_x, eef_vx=eef_vx, images = images[:,:,1:], np_X =np_X)
