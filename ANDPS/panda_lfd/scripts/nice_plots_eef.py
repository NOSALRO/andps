import matplotlib.pyplot as plt
import numpy as np
from palettable.colorbrewer.qualitative import Dark2_8
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
    # 'figure.figsize': [10, 5]
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


def get_index(i, N):
    if i<N:
        return i
    return N-1

EXPERIMENT = "eef"
POLICY = "ANDPS"
# POLICY = "NN"


# load data
data = np.load("data/"+POLICY+"_spiral_eval_eef.npz", allow_pickle=True)

dt = 0.01
max_t = data['end_time']
N_steps = int(max_t/dt)

rows = 2
cols = 3

fig, axs = plt.subplots(rows, cols, sharey=False, squeeze=False)

fig.suptitle("Movement in Task Space using "+POLICY, fontsize=16)

demo = np.array([data['demo'][get_index(i, len(data['demo']))] for i in range(N_steps)])
repro = np.array([data['eef_trajectory'][get_index(i, len(data['eef_trajectory']))] for i in range(N_steps)])

ids = [0, 1, 2]
titles = ['x-axis', 'y-axis', 'z-axis']
for k in range(3):
    ax = axs[0][k]
    ax.set_title(titles[k])
    ax.set_xlabel('time (s)')

    if k == 0:
        # ax.set_ylabel('m')
        ax.set_ylabel('EEF')

        ax.plot([i*dt for i in range(len(demo))], demo[:, ids[k]], label='Demonstrated trajectory', color=colors[6])
        ax.plot([i*dt for i in range(len(repro))], repro[:, ids[k]], label='Evaluation', color=colors[2])
    else:
        ax.plot([i*dt for i in range(len(demo))], demo[:, ids[k]], color=colors[6])
        ax.plot([i*dt for i in range(len(repro))], repro[:, ids[k]], color=colors[2])
    decorate_axis(ax)

plt.tight_layout(pad=0.4, h_pad=2.5, rect=(0, 0.075, 1, 1))

# plot results with force application
data = np.load("data/"+POLICY+"_spiral_eval_eef_force_push.npz", allow_pickle=True)
demo = np.array([data['demo'][get_index(i, len(data['demo']))] for i in range(N_steps)])
repro = np.array([data['eef_trajectory'][get_index(i, len(data['eef_trajectory']))] for i in range(N_steps)])
for k in range(3):
    ax = axs[1][k]
    # ax.set_title(titles[k])

    ax.set_xlabel('time (s)')
    if k == 0:
        ax.set_ylabel('EEF')
        force_app_plt = ax.axvline(x=data["time_force"][0], color='k', linestyle='--', linewidth=1, label='Time of force application')
        ax.axvline(x=data["time_force"][1], color='k', linestyle='--', linewidth=1)
        demo_plt = ax.plot([i*dt for i in range(len(demo))], demo[:, ids[k]], label='Demonstrated trajectory', color=colors[6])
        eval_plt = ax.plot([i*dt for i in range(len(repro))], repro[:, ids[k]], label='Evaluation', color=colors[2])
    else:
        ax.axvline(x=data["time_force"][0], color='k', linestyle='--', linewidth=1)
        ax.axvline(x=data["time_force"][1], color='k', linestyle='--', linewidth=1)
        ax.plot([i*dt for i in range(len(demo))], demo[:, ids[k]], color=colors[6])
        ax.plot([i*dt for i in range(len(repro))], repro[:, ids[k]], color=colors[2])
    decorate_axis(ax)

# add legend
fig.legend(handles=[force_app_plt,demo_plt[0],eval_plt[0]], loc = "lower center", bbox_to_anchor=(0.5, 0.0005), ncol=3, fancybox=True, shadow=True)
# save figure
plt.savefig("plots/" + EXPERIMENT + "_" + POLICY + '.svg', dpi=500)
plt.savefig("plots/" + EXPERIMENT + "_" + POLICY + '.png', dpi=500)
plt.show()
