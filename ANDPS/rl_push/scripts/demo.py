import RobotDART as rd
import copy
import numpy as np

# Simu
simu = rd.RobotDARTSimu(0.01)
simu.set_collision_detector("ode")

# Graphics
graphics = rd.gui.Graphics()
simu.set_graphics(graphics)
graphics.look_at([0., 3., 2.75], [0., 0., 0.75])

# Iiwa custom eef
iiwa_packages = [("iiwa_description", "robots/iiwa/iiwa_description")]
robot = rd.Robot("robots/iiwa/iiwa.urdf", iiwa_packages, "iiwa")
robot_base_pose = [0., 0., np.pi/2., 0., -0.5, 0.75]
robot.set_base_pose(robot_base_pose)


# Set initial pos
init_positions = copy.copy(robot.positions())
# init_positions[0] = -2.
init_positions[3] = -np.pi / 2.0
init_positions[5] = np.pi / 2.0
robot.set_positions(init_positions)
robot.fix_to_world()
robot.set_position_enforced(True)
robot.set_actuator_types("servo")


table_packages = [("table", "robots/table")]
table = rd.Robot("robots/table/table.urdf",   table_packages, "table")
table.set_color_mode("material")
# Box table (ugly)
# table_dims = [3., 2., 0.7]
# table_pose = [0, 0, 0, 0, 0, 0.35]
# table_color = [0.933, 0.870, 0.784, 1.]
# table = rd.Robot.create_box(
# table_dims, table_pose, "fix", mass=30., color=table_color, box_name="table")
table.fix_to_world()

# Box to be moved
box = rd.Robot.create_box([0.05, 0.05, 0.05], [0., 0., 0.,0., 0., 0.79], "free", mass=0.01, color=[0.0, 0.8, 0.0, 1.], box_name="box")

# Ghost target
target = rd.Robot.create_ellipsoid([0.5, 0.5, 0.001], [0., 0., 0.,0.5, 0.5, 0.8], "fixed", mass=0.01, color=[1.0, 0.0, 0.0, 0.5], ellipsoid_name="target")

# Αdd robots
simu.add_robot(table)
simu.add_robot(robot)
simu.add_robot(box)
simu.add_visual_robot(target)
# Add nice floor
simu.add_checkerboard_floor()

# Run simulation for 50 seconds
simu.run(50.)
