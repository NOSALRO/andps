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
    # 'figure.figsize': [10, 5]
    'figure.figsize': [15, 5]
}
plt.rcParams.update(params)

def get_index(i, N):
    if i < N:
        return i
    return N-1
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





EXPERIMENT = 'joint'
POLICY = "ANDPS"
# POLICY = "NN"

data = np.load("data/" + POLICY + "_spiral_eval_" + EXPERIMENT + ".npz")

dt = 0.01
max_t = data['end_time']
N_steps = int(max_t/dt)

rows = 1
cols = 3



fig, axs = plt.subplots(rows, cols, sharey=False, squeeze=False)


fig.suptitle("Movement in Joint Space using "+POLICY, fontsize=16)


demo = np.array([data['demo'][get_index(i, len(data['demo']))] for i in range(N_steps)])
repro = np.array([data['joints_dataset'][get_index(i, len(data['joints_dataset']))] for i in range(N_steps)])


ids = [0, 1, 5]
titles = []
for i in ids:
    titles.append('joint-'+str(i+1))

for k in range(3):
    ax = axs[0][k]
    ax.set_title(titles[k])
    ax.set_xlabel('time (s)')
    if k == 0:
        ax.set_ylabel('radians')

    ax.plot([i*dt for i in range(len(demo))], demo[:, ids[k]], label='Demonstrated trajectory', color=colors[6])
    ax.plot([i*dt for i in range(len(repro))], repro[:, ids[k]], label='Evaluation', color=colors[2])

    decorate_axis(ax)

plt.tight_layout(pad=0.4, h_pad=2.5, rect=(0, 0.075, 1, 1))
fig.legend(labels=['Demonstrated trajectory', 'Evaluation'], loc = "lower center", bbox_to_anchor=(0.5, 0.0005), ncol=3, fancybox=True, shadow=True)


# save figure
plt.savefig("plots/" + EXPERIMENT + "_" + POLICY + '.svg', dpi=500)
plt.savefig("plots/" + EXPERIMENT + "_" + POLICY + '.png', dpi=500)
plt.show()
