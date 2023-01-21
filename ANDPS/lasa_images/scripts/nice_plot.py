import numpy as np
import matplotlib.pyplot as plt

from palettable.scientific.diverging import Vik_11
colors = Vik_11.mpl_colors

params = {
    'axes.labelsize': 12,
    # 'font.size': 8,
    # 'figure.titlesize': 16,
    'legend.fontsize': 12,
    'axes.titleweight' : 'bold',
    'figure.titleweight' : 'bold',
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


EXPERIMENT = 'CNN'
NICE_NAME = 'CNN'
TITLE = 'ANDPs Multi-Task with Images'



names = ["JShape", "Angle", "Khamesh"]
num_DSs = [0, 1, 2]


rows = 1
cols = 3

fig, axs = plt.subplots(rows, cols, sharey=False, squeeze=False)
fig.suptitle(TITLE)

for id in range(len(names)):
    name = names[id]
    num_DS = num_DSs[id]
    all_data = np.load("data/lasa_image_data_"+ name + '.npz', allow_pickle=True)

    ax_id_i = int(id / cols)
    ax_id_j = id % cols

    ax0 = axs[ax_id_i][ax_id_j]
    decorate_axis(ax0)


    demos = all_data['demo']

    starting_points = all_data['init']
    target = all_data['x_tar']

    ax0.set_title(name)
    train_eval = all_data['trajectory']
    Y, X = all_data['Y'], all_data['X']
    U, V = all_data['U'], all_data['V']

    streams = ax0.streamplot(X, Y, U, V, linewidth=1, color=colors[4], zorder = 1,  arrowstyle ='->' )
    demon = ax0.scatter(demos[0:-1:8,0], demos[0:-1:8,1], color=colors[6], s=3, label = 'Demonstrations', zorder=2)
    evalu_train = ax0.scatter(train_eval[:,0], train_eval[:,1], color=colors[0], s=1, label='Evaluation of train data', zorder=3)
    init_pos = ax0.scatter(starting_points[:, 0], starting_points[:, 1], marker="X", color='maroon', label='Initial Position', zorder = 4)
    target_pos = ax0.scatter(0, 0, marker="X", c='seagreen',s = 80, label='Target Position', zorder = 5)
    w_x = [np.min(X), np.max(X)]
    w_y = [np.min(Y), np.max(Y)]

    ax0.set_xlim(w_x)
    ax0.set_ylim(w_y)
    ax0.set_xlabel('x')
    ax0.set_ylabel('y', rotation=0)

    ax0.set_aspect('auto')

fig.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1, rect=(0, 0.075, 1, 1))
lgnd = fig.legend(handles=[demon, evalu_train, init_pos, target_pos], loc = "lower center", bbox_to_anchor=(0.5, 0.0005),ncol=5, fancybox=True, shadow=True)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
lgnd.legendHandles[2]._sizes = [30]
lgnd.legendHandles[3]._sizes = [30]
# lgnd.legendHandles[4]._sizes = [30]
plt.savefig("plots/"+NICE_NAME + ".svg", dpi=100)
plt.savefig("plots/"+NICE_NAME + ".png", dpi=100)
plt.savefig("plots/"+NICE_NAME + ".pdf", dpi=100)
plt.show()
plt.close()
