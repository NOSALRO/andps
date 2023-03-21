import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


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


data = np.load('data/reactive_2.npz')
demo = np.load('data/angle.npz')
img_1 = np.load('data/line.npz')["images"][:,:,1]
img_2 = np.load('data/angle.npz')["images"][:,:,200]
fig = plt.figure(tight_layout = True)
gs = gridspec.GridSpec(1,4)

fig.suptitle("Reactiveness of ANDPs on changes in the Non Controllable part of the state", fontsize=16)

ax_img_1 = fig.add_subplot(gs[0,0])
ax_img_1.imshow(img_2,cmap='gray')
ax_img_1.set_xticks([])
ax_img_1.set_yticks([])
ax_img_1.set_title("Non Controllable Part")
ax_img_1.set_xlabel("t=0s")


for j in range(1,4):
    ax = fig.add_subplot(gs[:,j])
    eval_hndl = ax.plot([i*0.01 for i in range(len(demo["eef_x"][:1000,j-1]))], demo["eef_x"][:1000,j-1], color = colors[2], label="Demonstration")
    demo_hndl = ax.plot([i*0.01 for i in range(len(data["eef_trajectory"][:1000,j-1]))], data["eef_trajectory"][:1000,j-1], color = colors[0], label="Evaluation")
    target_hndl = ax.scatter([i*0.01 for i in range(len(data["eef_trajectory"][:1000,j-1]))][-1], data["eef_trajectory"][:1000,j-1][-1], color=colors[4],marker ='x',label='Target')
    decorate_axis(ax)
    force_app_plt = ax.axvline(x=0., color='k', linestyle='--', linewidth=1, label='Time of Force application')
    force_app_plt = ax.axvline(x=5., color='k', linestyle='--', linewidth=1, label='Time of Force application')
    ax.set_xlabel("time (s)")
    if j == 3:
        ax.set_ylim([0.2,0.7])
    if j == 2:
        ax.set_ylim([0.0, 0.5])
    if j == 1:
        ax.set_ylim([0.3,0.7])
        ax.set_ylabel("EEF")

fig.legend(handles=[force_app_plt,demo_hndl[0], eval_hndl[0], target_hndl], loc = "lower center", bbox_to_anchor=(0.5, 0.0005), ncol=5, fancybox=True, shadow=True)
plt.tight_layout(pad=0.4, h_pad=0.2, rect=(0, 0.075, 1, 1))
plt.savefig("/home/dtotsila/Desktop/andps/reactive_2.pdf")
plt.show()