import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from palettable.cartocolors.qualitative import Pastel_5

colors = Pastel_5.mpl_colors

params = {
    'axes.labelsize': 12,
    # 'font.size': 8,
    # 'figure.titlesize': 16,
    'legend.fontsize': 12,
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    # 'text.usetex': False,
    # # 'figure.figsize': [1.8, 5]
    # 'figure.figsize': [20, 15]
    'figure.figsize': [15, 5]
}
plt.rcParams.update(params)


def decorate_axis(ax, remove_left=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)

    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax.grid(axis='x', color="0.9", linestyle='-', linewidth=1)
    ax.set_axisbelow(True)



# read data

angle = np.load("data/angle.npz")
angle_eval = np.load("data/angle_results.npz")

line = np.load("data/line.npz")
line_eval = np.load("data/line_results.npz")

sine = np.load("data/sine.npz")
sine_eval = np.load("data/sine_results.npz")


rows = 3
cols = 4
fig, axs = plt.subplots(rows, cols, sharey=False, squeeze=False)
fig.tight_layout(pad=1.5)
fig.suptitle("Multi-Task Experiment via ANDPs", fontsize=16)

demos_x = [angle['eef_x'],line['eef_x'],sine['eef_x']]
eval_x = [angle_eval['eef_trajectory'],line_eval['eef_trajectory'],sine_eval['eef_trajectory']]
imgs_x = [angle['images'][:,:,2],line['images'][:,:,2],sine['images'][:,:,2]]
ids = [0, 1, 2]
titles = ['x-axis','y-axis','z-axis']

for i in range(0,3):
    for j in range(1,4):
        ax = axs[i][j]
        if j == 3:
            ax.set_ylim([0.2,0.7])
        if j == 2:
            ax.set_ylim([0.0, 0.5])
        if j == 1:
            ax.set_ylim([0.3,0.7])

        if(i==0):
            ax.set_title(titles[j-1])
        if(i == 2):
            ax.set_xlabel('time (s)')
        decorate_axis(ax)
        if j == 1:
            ax.set_ylabel("EEF")
        demo_hndl = ax.plot([i*0.01 for i in range(len(demos_x[i][:700,j-1]-1))], demos_x[i][:700,j-1], color = colors[2], label="Demonstration")
        eval_hndl = ax.plot([i*0.01 for i in range(len(demos_x[i][:700,j-1]))], eval_x[i][:700,j-1], color = colors[0], label="Evaluation")
        target_hndl = ax.scatter([i*0.01 for i in range(len(demos_x[i][:700,j-1]))][-1], demos_x[i][:700,j-1][-1], color=colors[4],marker ='x',label='Target')
    img = axs[i,0]
    if i == 0:
        img.set_title("Non Controllable Part")
    img.imshow(imgs_x[i], cmap='gray')
    img.set_xticks([])
    img.set_yticks([])
    img.set_xlabel("t=0s")



fig.legend(handles=[demo_hndl[0] ,eval_hndl[0], target_hndl], loc = "lower center", bbox_to_anchor=(0.5, 0.0005), ncol=3, fancybox=True, shadow=True)
plt.tight_layout(pad=0.4, h_pad=1.0, rect=(0, 0.075, 1, 1))
plt.savefig("panda_images.svg")
plt.savefig("results.jpg")
plt.savefig("/home/dtotsila/Desktop/andps/panda_images.pdf")

plt.show()
