import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import ast
from scipy.spatial.transform import Rotation as R
from bagpy import bagreader
from matplotlib.animation import FuncAnimation, PillowWriter
from plot_utils import decorate_axis

import sys
import csv


def smoothing(x_raw, a=0.1):
    """
    Apply exponential smoothing to a sequence of data.

    Args:
        x_raw (list or numpy.ndarray): Input sequence of data to be smoothed.
        a (float, optional): Smoothing factor (0 < a < 1). Default is 0.1.

    Returns:
        list: Smoothed sequence of data.

    This function applies exponential smoothing to a sequence of data using the specified smoothing factor 'a'.
    Exponential smoothing computes the current smoothed value as a weighted average of the current raw value
    and the previous smoothed value. The higher the 'a' value, the more weight is given to the current raw value.

    Note:
        - 'x_raw' can be a list or a NumPy array.
        - Ensure that 0 < 'a' < 1 to achieve meaningful smoothing.

    Example:
        raw_data = [10, 15, 20, 25, 30]
        smoothed_data = smoothing(raw_data, a=0.2)
    """
    x_smooth = []
    x_smooth.append(x_raw[0])
    for i in range(1, len(x_raw)):
        x_smooth.append(a*x_raw[i] + (1-a)*x_smooth[i-1])

    return x_smooth


def unpack_to_npz(bagfiles: list, image_topics: list, data_topic: str, delete_after=True, verbose=True, smoothing=True, plot=False):
    """
    Unpacks a list of bag files and stores them to a file.

    Args:
        bagfiles (list): List of bag file names (including .bag extension) to be unpacked and are located under data/.
        image_topics (list): List of topics containing images.
        data_topic (str): Topic containing the data.
        delete_after (bool, optional): Indicates whether to delete bagfiles after unpacking. Default is False.
        verbose (bool, optional): Indicates whether to print the unpacking process. Default is False.
        smoothing (bool, optional): Indicates whether to smooth the data. Default is True.
        plot(bool, optional): Indicates whether to plot the data. Default is False.
    Returns:
        dataframes (list): List of dataframes extracted from the bag files.
    """

    failed_files = []
    for bagfile in bagfiles:
        # add /data/ to the path
        bagfile_path = os.path.join(os.getcwd(), "data", bagfile)
        print(bagfile_path)
        # try:
        b = bagreader(bagfile_path)
        # except:
        # failed_files.append(bagfile)
        # continue

        if (verbose):
            print("Unpacking bagfile: ", bagfile, "\n")
            print("="*70)
            print((bagfile + " Topic Table").center(70),
                  "\n", "="*70, "\n", b.topic_table)
            print("="*70)

        # read message from the bag using topic
        message_csv_path = b.message_by_topic(data_topic)

        # Read CSV file
        message_df = pd.read_csv(message_csv_path)

        # drop rows were id == 0
        # get number of rows were id == 0
        skip = message_df[message_df.id == 0].shape[0]

        # skip extra 3 timesteps to avoid 0 velocities
        message_df = message_df[skip+3:]

        if (verbose):
            print("="*70)
            print((bagfile + " Message Dataframe Information").center(70), "\n", "="*70)
            message_df.info()
            print("="*70)

        # extract the 2d position of the datagrame
        pos2d = extract_2d_translation(message_df, smoothing)
        thetas, all_R = extract_rotation(message_df)
        f = calulate_average_frequency(message_df)
        # get the images
        cam = extract_images_from_topic(b, image_topics[0], skip+3)

        if (verbose):
            print("="*70)
            print((bagfile + " Image Dataframe Information").center(70), "\n", "="*70)
            print("pos2d: ", pos2d.shape)
            print("thetas: ", thetas.shape)
            print("cam: ", cam.shape)
            print("freq: ", f)
            print("="*70)

        # if (plot):
        vizualize_data(bagfile, pos2d, all_R, cam)
        # path were the csvs generated from the bagfile will be stored
        unpacked_data_path = 'data/' + os.path.basename(bagfile).split('.')[0]

        np.savez('data/' + os.path.basename(bagfile).split('.')
                 [0] + '.npz', position=pos2d, rotation=thetas, images=cam, frequency=f)

        if delete_after:
            os.system("rm -rf " + unpacked_data_path)


def extract_2d_translation(df: pd.DataFrame, smooth=True):
    """
    Extracts 2D translation data from a DataFrame.
    extracts the 'x' and 'z' components and returns them as a 2D numpy array
    representing a 2D translation.

    Args:
        df (pd.DataFrame): The input DataFrame with 'x' and 'z' positional data.

    Returns:
        np.ndarray: A 2D numpy array where each row contains [x, z] components
                    of the positional data.
    """

    # Y-axis is up, so we ignore it
    if (smooth):
        pos_x = smoothing(df['x'].to_numpy())
        pos_z = smoothing(df['z'].to_numpy())
    else:
        pos_x = df['x'].to_numpy()
        pos_z = df['z'].to_numpy()

    return np.array([pos_x, pos_z]).T


def extract_rotation(df: pd.DataFrame):
    quat_w = df['qw'].to_numpy()
    quat_x = df['qx'].to_numpy()
    quat_y = df['qy'].to_numpy()
    quat_z = df['qz'].to_numpy()

    quats = np.array([quat_w, quat_x, quat_y, quat_z]).T

    angles = []
    Rs = []
    for i in range(quats.shape[0]):
        r = R.from_quat(quats[i, :])
        Rs.append(r)
        angle = r.as_euler("yzx")[0]
        angles.append(angle)

    angles = np.array(angles).T

    return angles, Rs

# caluclate average frequency


def calulate_average_frequency(df: pd.DataFrame):
    times = df['Time'].to_numpy()
    freq = 0
    for i in range(len(times)-1):
        freq += times[i+1] - times[i]
    freq = len(times)/freq

    return freq


def extract_images_from_topic(bag, image_topic: str, skip_first_n: int):
    """
    Extract and process images from a specified topic in a ROS bag file.

    Args:
        bag (rosbag.Bag): Input ROS bag containing image data.
        image_topic (str): The name of the ROS topic with image messages.
        skip_first_n (int): Number of initial messages to skip.

    Returns:
        numpy.ndarray: Processed images in the format (num_images, height, width, channels).

    This function reads image data from a specified ROS topic in a bag file. It processes each image by parsing
    the binary data and reshaping it based on provided dimensions. Any failed image processing attempts are
    recorded and printed. The images are then passed through the 'edit_img' function to perform additional edits.

    Note:
        - Make sure to import necessary libraries and modules like 'bagpy', 'np', 'pd', 'ast', and 'edit_img'.
        - The 'edit_img' function should be defined before using this function.

    Example:
        bag = rosbag.Bag('data.bag')
        processed_images = extract_images_from_topic(bag, '/camera/image_raw', skip_first_n=5)
    """

    image_path = bag.message_by_topic(image_topic)
    df = pd.read_csv(image_path, engine='python')
    failed_imgs = []

    cam_1 = np.empty((df.shape[0], 85, 85, 3), dtype=np.uint8)
    for index, row in df.iterrows():
        try:
            im = np.frombuffer(ast.literal_eval(row['data']), dtype=np.uint8).reshape(
                row['height'], row['width'], -1)
        except:
            failed_imgs.append(index)
            print("Failed to read image at index: ", index)
            continue

        cam_1[index, :, :, :] = edit_img(im)
        # plt.imshow(edit_img(im))
        # plt.show()
    return cam_1[skip_first_n:, :, :, :]


def edit_img(im):
    """
    Edit and modify an image using channel reordering and downsampling.

    Args:
        im (numpy.ndarray): Input image with shape (height, width, channels).

    Returns:
        numpy.ndarray: Modified image with adjusted channels and downsampling.

    This function applies two types of edits to the input image:
    1. Channel Reordering: The function reorders the color channels from BGR to RGB, and slices the third axis.
    2. Downsampling: The function reduces the image size by selecting a subset of pixels. It performs row and column
       slicing with specific step sizes, resulting in a downsampled image.

    Example:
        edited_image = edit_img(input_image)
    """

    # edit channels
    im = im[:, :im.shape[1]//2, [2, 1, 0]]

    # downsample imag
    im = im[40:380:4, 70:410:4, :]

    return im


def vizualize_data(bag_name, pos, rots, imgs):
    """
    Visualize and animate data for a given trajectory.

    Args:
        bag_name (str): The name of the trajectory bag file.
        pos (numpy.ndarray): The position data with shape (num_frames, 2).
        rots (list of Rotation): List of rotation matrices representing orientation over time.
        imgs (numpy.ndarray): Image data with shape (num_frames, height, width, channels).

    This function creates a dynamic visualization of trajectory data, including position, orientation, and camera images.
    It uses Matplotlib to generate an animated GIF that showcases the evolution of the data over time.

    The trajectory's position is shown using scatter plots, while orientation is depicted with 3D quiver plots.
    Images captured by the camera are also displayed, forming a comprehensive overview of the dataset.

    Note:
        - Ensure Matplotlib and NumPy are installed.
        - The `Rotation` class should provide the `as_matrix()` method for rotation matrices.

    Example:
        visualize_data('trajectory.bag', pos_data, rotation_list, image_data)
    """

    total_frames = pos.shape[0]

    # Create some sample data
    x = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, x)
    Z1 = np.sin(np.sqrt(X**2 + Y**2))
    Z2 = np.cos(np.sqrt(X**2 + Y**2))

    # Create a figure with subplots
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Data Visualization for trajectory ' +
                 bag_name.split('.')[0], fontsize=16)
    # Subplot 1 (3D)
    ax1 = fig.add_subplot(2, 2, 1)
    # set limits based on position dataset
    max_x = np.max(pos[:, 0])
    min_x = np.min(pos[:, 0])

    max_y = np.max(pos[:, 1])
    min_y = np.min(pos[:, 1])

    ax1.set_xlim([min_x, max_x])
    ax1.set_ylim([min_y, max_y])
    ax1.scatter(pos[0, 0], pos[0, 1], color='slateblue')
    ax1.set_title('Position over time')

    # Subplot 2 (3D)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.grid(False)

    ax2.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    ax2.yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
    ax2.zaxis.set_pane_color((0.3, 0.3, 0.3, 1.0))

    ax2.set_title('Orientation over time')

    # Define the original coordinate system
    origin = np.array([0, 0, 0])
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    rotated_x_axis = np.dot(rots[0].as_matrix(), x_axis)
    rotated_y_axis = np.dot(rots[0].as_matrix(), y_axis)
    rotated_z_axis = np.dot(rots[0].as_matrix(), z_axis)

    ax2.quiver(*origin, *rotated_x_axis, color='r', linestyle='dotted')
    ax2.quiver(*origin, *rotated_y_axis, color='g', linestyle='dotted')
    ax2.quiver(*origin, *rotated_z_axis, color='b', linestyle='dotted')
    ax2.set_facecolor('black')
    ax2.view_init(elev=-84, azim=-90)

    # Subplot 3 (Image)
    ax3 = fig.add_subplot(2, 1, 2)
    image = np.random.random((100, 100))  # Replace with your image data
    im = ax3.imshow(image, cmap='gray')
    ax3.set_title('Cam 1')

    # Animation update function
    def update(frame):
        print(frame)

        new_image = imgs[frame-1, :, :, :]

        # Update the image data
        im.set_array(new_image)

        # clear ax1, redifine limits and plot new data
        ax1.clear()
        ax1.set_title('Position over time')
        ax1.set_xlim([min_x-100, max_x+100])
        ax1.set_ylim([min_y-100, max_y+100])
        ax1.plot(pos[:frame, 0], pos[:frame, 1], color='slateblue')
        ax1.scatter(pos[frame-1, 0], pos[frame-1, 1], color='slateblue')

        rotated_x_axis = np.dot(rots[frame-1].as_matrix(), x_axis)
        rotated_y_axis = np.dot(rots[frame-1].as_matrix(), y_axis)
        rotated_z_axis = np.dot(rots[frame-1].as_matrix(), z_axis)
        ax2.clear()
        ax2.set_title('Orientation over time')
        ax2.set_facecolor('black')
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])
        ax2.set_zlim([-1, 1])

        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.grid(False)

        ax2.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        ax2.yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
        ax2.zaxis.set_pane_color((0.3, 0.3, 0.3, 1.0))

        # Set the view angle to make y-axis point up
        ax2.view_init(elev=-84, azim=-90)

        ax2.quiver(*origin, *rotated_x_axis, color='r')
        ax2.quiver(*origin, *rotated_y_axis, color='g')
        ax2.quiver(*origin, *rotated_z_axis, color='b')
        ax2.text(*(origin + rotated_x_axis + rotated_x_axis/10), 'X',
                 color='white', fontsize=12, ha='center', va='center')
        ax2.text(*(origin + rotated_z_axis + rotated_z_axis/10), 'Z',
                 color='white', fontsize=12, ha='center', va='center')
        ax2.text(*(origin + rotated_y_axis + rotated_y_axis/10), 'Y',
                 color='white', fontsize=12, ha='center', va='center')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.arange(
        1, total_frames, 1), interval=1)

    writergif = PillowWriter()

    ani.save('plots/'+bag_name.split(".")[0] + '.gif', writer=writergif)


def create_final_dataset(trajectories: list, filename: str, plot=False):

    mean_target = np.zeros([1, 3])
    count = 0
    # locate all npz files (unpack to npz must be called before this)
    for trajectory in trajectories:
        # load corresponding npz
        data = np.load("data/" + trajectory + ".npz")
        traj_x = np.vstack(
            [data["position"][:, 0], data["position"][:, 1], data["rotation"]])
        freq = data["frequency"]

        # calculate trajectory xdot, reshape, and transpose
        traj_xdot = ((traj_x[:, 1:]-traj_x[:, :-1]) * freq).transpose()
        traj_x = traj_x[:, 1:].transpose()
        cam_1 = data['images'][1:, :, :, :]

        if (trajectory == trajectories[0]):
            np_X = traj_x
            np_Y = traj_xdot
            images = cam_1

        else:
            np_X = np.vstack([np_X, traj_x])
            np_Y = np.vstack([np_Y, traj_xdot])
            images = np.concatenate([images, cam_1], axis=0)
            assert np_X.shape[0] == np_Y.shape[0] == images.shape[0]

        mean_target += traj_x[-1]
        count += 1
        # use only one image for now

    # repeat final points  for 5 times
    # for _ in range(5):
    #     np_X = np.vstack([np_X, np_X[-1, :]])
    #     np_Y = np.vstack([np_Y, np.zeros_like(np_Y[-1, :])])
    #     images = np.vstack([images, images[-1, :, :, :]])

    mean_target /= count
    assert np_X.shape[0] == np_Y.shape[0] == images.shape[0]

    # flatten image and concatenate with np_X
    np_X = np.concatenate([np_X, images.reshape(images.shape[0], -1)], axis=1)
    print(np_X.shape)

    if (plot):
        fig = plt.figure(figsize=(25, 10))
        ax = fig.add_subplot(111)
        decorate_axis(ax)
        ax.scatter(np_X[:, 0], np_X[:, 1], c='#022B3A',
                   label="Trajectories",  s=2.5)

        ax.quiver(np_X[:, 0], np_X[:, 1], np_Y[:, 0], np_Y[:, 1],
                  color='#648DE5', alpha=0.2, label='Velocities')
        ax.scatter(mean_target[0, 0], mean_target[0, 1],
                   c='#C20114', label="Mean Target", marker='x', s=140.)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/'+filename.split(".")[0]+"all_demos.jpg")

    np.savez("data/" + filename, X=np_X, Y=np_Y, mean_target=mean_target)


if __name__ == "__main__":
    maxInt = sys.maxsize
    # edit the field size limit
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

    bags = ["left_1.bag",
            "left_3.bag",
            "left_5.bag",
            "right_2.bag",
            "right_4.bag",
            "right_6.bag",
            "straight_2.bag",
            "straight_4.bag",
            "straight_6.bag",
            "left_2.bag",
            "left_4.bag",
            "right_1.bag",
            "right_3.bag",
            "right_5.bag",
            "straight_1.bag",
            "straight_3.bag",
            "straight_5.bag"]

    bags_final = ["left_1",
                  "left_3",
                  "left_5",
                  "right_2",
                  "right_4",
                  "right_6",
                  "straight_2",
                  "straight_4",
                  "straight_6",
                  "left_2",
                  "left_4",
                  "right_1",
                  "right_3",
                  "right_5",
                  "straight_1",
                  "straight_3",
                  "straight_5"]

    unpack_to_npz(bags,  ["/camera1/image_raw"], "/phasespace_single")

    create_final_dataset(bags_final, "test.npz", True)
