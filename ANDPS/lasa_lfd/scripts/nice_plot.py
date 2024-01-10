import numpy as np
import matplotlib.pyplot as plt
import os
from palettable.scientific.diverging import Vik_11
colors = Vik_11.mpl_colors

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


# EXPERIMENT = 'simple_nn'
# NICE_NAME = 'NN'
# TITLE = 'Simple Neural Network'
# EXPERIMENT = 'andps'
# NICE_NAME = 'ANDPs'
# TITLE = 'ANDPs'
EXPERIMENT = 'seds'
NICE_NAME = 'SEDS'
TITLE = 'SEDS'



# names = ["JShape", "Angle", "Khamesh", "LShape", "PShape", "RShape", "Sharpc", "Sine", "Spoon", "Trapezoid", "Worm", "WShape"]
# num_DSs = [6, 3, 4, 6, 7, 6, 5, 5, 6, 6, 5, 6]
names = ["JShape", "Khamesh", "Sine", "Spoon", "Trapezoid"]
ks = [10, 25, 7, 5, 24]
num_DSs = [6, 6, 6, 5, 5]
temp = sorted(zip(names, num_DSs))

num_DSs = [x for _, x in temp]
names = [x for x, _ in temp]

np.random.seed(3)

rows = 1
cols = 5

fig, axs = plt.subplots(rows, cols, sharey=False, squeeze=False)
fig.suptitle(TITLE)

for id in range(len(names)):
    name = names[id]
    k = ks[id]
    dataset_folder = os.path.join('plots','data', EXPERIMENT)


    path = dataset_folder +'/'+name+'_' + str(k)
    all_data = np.load(path + '.npz', allow_pickle=True)

    ax_id_i = int(id / cols)
    ax_id_j = id % cols

    ax0 = axs[ax_id_i][ax_id_j]
    decorate_axis(ax0)

    demos = all_data['dataset_pos']
    starting_points = all_data['initial_pos']
    target = all_data['target_pos']

    ax0.set_title(name)

    train_eval = all_data['train_trajectory']

    test_eval = all_data['test_trajectory']

    Y, X = all_data['Y'], all_data['X']
    U, V = all_data['U'], all_data['V']


    streams = ax0.streamplot(X, Y, U, V, linewidth=1,color=colors[4], zorder=1,  arrowstyle='->')
    demon = ax0.scatter(demos[0:-1:8, 0], demos[0:-1:8, 1], color=colors[6], s=3, label='Demonstrations', zorder=2)
    evalu_train = ax0.scatter(train_eval[:,0], train_eval[:,1], color=colors[0], s=1, label='Evaluation of train data', zorder=3)
    evalu_test = ax0.scatter(test_eval[:,0], test_eval[:,1],  color=colors[2], s=1, label='Evaluation of test data', zorder=4)
    init_pos = ax0.scatter(starting_points[:, 0], starting_points[:, 1],marker="X", color=colors[10], label='Initial Position', zorder=5)
    target_pos = ax0.scatter(0, 0, marker="X", c='g',s=80, label='Target Position', zorder=6)

    w_x = [np.min(X), np.max(X)]
    w_y = [np.min(Y), np.max(Y)]

ax0.set_xlim(w_x)
ax0.set_ylim(w_y)
ax0.set_xlabel('x')
ax0.set_ylabel('y', rotation=0)

ax0.set_aspect('auto')


fig.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1,
                 rect=(0, 0.075, 1, 1))  # , h_pad=4.75)
lgnd = fig.legend(handles=[demon, evalu_train, evalu_test, init_pos, target_pos], loc="lower center", bbox_to_anchor=(0.5, 0.0005),
                  ncol=5, fancybox=True, shadow=True)  # , loc=0)#,loc = "upper left")
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
lgnd.legendHandles[2]._sizes = [30]
lgnd.legendHandles[3]._sizes = [30]
lgnd.legendHandles[4]._sizes = [30]
# plt.show()
# save png
plt.savefig("plots/results/"+EXPERIMENT+"/" + NICE_NAME + ".png")
# save pdf
plt.savefig("plots/results/"+EXPERIMENT+"/" + NICE_NAME + ".pdf")
#save svg
plt.savefig("plots/results/"+EXPERIMENT+"/" + NICE_NAME + ".svg")