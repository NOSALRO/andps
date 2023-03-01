import RobotDART as rd
import numpy as np
import dartpy
from utils import *
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import andps_images as andps,  CustomDataset
from torchvision import transforms
convert_tensor = transforms.ToTensor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_state(robot,rd_image):
    pos = robot.body_pose(eef_link_name).translation()
    img = image = np.array(rd.gui.convert_rgb_to_grayscale(rd_image).data)
    return np.concatenate([pos,img])

target = [0.40082605, 0.40457038, 0.52994658]

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
line_box_packages = [("line_box", "urdfs/line_box")]
line_box = rd.Robot("urdfs/line_box/line_box.urdf",  line_box_packages,"line_box")
line_box.set_color_mode("material")
line_box.fix_to_world()
box_tf = line_box.base_pose()
box_tf.set_translation([3.5, 0., 1.])
line_box.set_base_pose(box_tf)
simu.add_robot(line_box)

# graphics
gconfig = rd.gui.Graphics.default_configuration()
gconfig.width = 480
gconfig.height = 480
gconfig.shadowed = False
graphics = rd.gui.Graphics(gconfig)
simu.set_graphics(graphics)
simu.enable_status_bar(enable=False)
graphics.look_at((-2.5, 1.0, 1.), (0.0, 0.0, 0.5))

# record images from main camera/graphics
graphics.camera().record(True)
graphics.record_video("line-env-eval.mp4", simu.graphics_freq())

# add camera
im_width = 64
im_height = 64
camera = rd.sensor.Camera(graphics.magnum_app(), im_width, im_height)
camera.camera().record(True)
camera.record_video("line-pov-eval.mp4")

# make camera look forward, attach to ee, add to simu
tf = dartpy.math.Isometry3()
rot =  dartpy.math.AngleAxis(-np.pi/2, [0., 1., 0.])
rot = rot.multiply( dartpy.math.AngleAxis(1.57, [0., 0., 1.])).to_rotation_matrix()
tf.set_translation([0., 0., 0.1])
tf.set_rotation(rot)
camera.attach_to_body(robot.body_node("panda_ee"), tf)
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


# Number of dynamical systems
num_DS = 6
net = andps(3, num_DS, target, 3,device)
net.to(device)
net.load_state_dict(torch.load("models/panda_image_200.pt"))
net.eval()
while(continue_simu):
    t += dt
    update = False
    if simu.scheduler().schedule(control_freq):
        update = True
        eef_tf = robot.body_pose(eef_link_name)
        vel_rot = controller.update(eef_tf)[0][:3]

        # print(get_state(robot,camera.image()))
        vel_xyz = net(torch.Tensor(get_state(robot,camera.image()).reshape(-1,64*64 +3)).to(device)).detach().cpu().numpy().copy()
        vel = np.concatenate((vel_rot, vel_xyz.reshape(3,)))

        # Jacobian pseudo-inverse
        jac_pinv = damped_pseudoinverse(robot.jacobian(eef_link_name))

        # Compute joint commands
        cmd = jac_pinv @ vel
        robot.set_commands(cmd)

        timestep_counter += 1

        if(timestep_counter == 1000):
            continue_simu = False
    simu.step_world()
    if update:
        eef_trajectory = np.vstack((eef_trajectory, robot.body_pose(eef_link_name).translation()))
        images = np.dstack((images, image_as_grayscale_array(camera.image())))


import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.scatter(eef_trajectory[:,0],eef_trajectory[:,1],eef_trajectory[:,2], c='r')
data = np.load('data/line.npz')
eef_x = data['eef_x']

ax.scatter(eef_x[:,0],eef_x[:,1],eef_x[:,2], c='b')
plt.show()


