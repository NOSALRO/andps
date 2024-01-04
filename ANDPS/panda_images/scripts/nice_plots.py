import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from palettable.scientific.diverging import Vik_15
colors = Vik_15.mpl_colors

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
    'figure.figsize': [20, 5]
}
plt.rcParams.update(params)


def decorate_axis(ax, remove_left=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(not remove_left)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax.grid(axis='x', color="0.9", linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

target = np.load("data/target.npy")


############## EXPERIMENT 1: Simple Multi-Task show all evaluations in a 3x4 plot ################
def plot_multi_task():
    TITLE = 'ANDPs Multi-Task with Images'
    NICE_NAME = 'andps_multi_task_images'
    # one row for each image
    rows = 3

    # image, x, y, z
    cols = 4

    fig, axs = plt.subplots(rows, cols, sharey=False, squeeze=False)
    fig.suptitle(TITLE)

    names = ["angle", "line", "sine"]
    axis_names = ["x-axis", "y-axis", "z-axis"]
    for i in range(rows):
        # read data
        data = np.load("data/"+names[i]+"_eval.npz")
        
        # fist plot the image (get the 5th)
        axs[i, 0].imshow(data["images"][:, :, 5], cmap='gray')
        
        print(data["images"].shape)
        print(data["train"].shape)

        # disable ticks
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])

        # set x label t = 0s
        axs[i, 0].set_xlabel("t=0s")

        # then plot the x, y, z over time (one plot for each) also divide the x axis ticks by 100
        for j in range(3):
            demon = axs[i, j+1].plot(data["train"][:1000, j], color=colors[9], label="Demonstration")
            evalu_train = axs[i, j+1].plot(data["test"][:1000, j], color=colors[3], label="Evaluation")
            # scatter plot the target with a green x
            target_pos = axs[i, j+1].scatter(1000, target[j], color="green", marker='x', label="target", s=100, zorder=10)
            if i == 0: 
                axs[i, j+1].set_title(axis_names[j])
            if j == 0:
                axs[i, j+1].set_ylim([0.5, 0.6])
                axs[i, j+1].set_yticks([0.5, 0.55,0.6])
                axs[i, j+1].set_yticklabels([.5, "",.6])
                axs[i, j+1].set_ylabel("EEF")
            elif j == 1:
                axs[i, j+1].set_ylim([0., 0.5])
                axs[i, j+1].set_yticks([0., 0.25, 0.5])
                axs[i, j+1].set_yticklabels([.0, "", .5])
            else:
                axs[i, j+1].set_ylim([0.3, 0.7])
                axs[i, j+1].set_yticks([0.3, 0.5,0.7])
                axs[i, j+1].set_yticklabels([0.3, "",0.7])
            decorate_axis(axs[i, j+1])
            axs[i, j+1].set_aspect('auto')
            axs[i, j+1].set_xlim([0, 1100])

            # convert the 1000 ticks to 10s
            axs[i, j+1].set_xticks([0, 250, 500, 750, 1000])
            axs[i, j+1].set_xticklabels([0, 2.5, 5, 7.5, 10])
            axs[i, j+1].set_xlabel("time (s)")
        axs[0, 0].set_title("Non controllable part")

    fig.tight_layout(pad=0.5, w_pad=0.15, h_pad=0.1, rect=(0, 0.075, 1, 1))
    plt.legend(bbox_to_anchor=(-0.6, -1.3), loc='lower center', ncol=3, fancybox=True, shadow=True)
    # lgnd.legendHandles[4]._sizes = [30]
    plt.savefig("plots/"+NICE_NAME + ".svg", dpi=100)
    plt.savefig("plots/"+NICE_NAME + ".png", dpi=100)
    plt.savefig("plots/"+NICE_NAME + ".pdf", dpi=100)
    plt.show()
    plt.close()

############## EXPERIMENT 2: Force perturbations, 1x4 ################
def plot_force():
    TITLE = "Robustness of ANDPs to external force perturbations"
    NICE_NAME = "andps_robustness"

    # one row for each image
    rows = 1

    # image, x, y, z
    cols = 4

    fig, axs = plt.subplots(rows, cols, sharey=False, squeeze=False)
    fig.suptitle(TITLE)

    # set figzise height to 1/3 of the params
    fig.set_figheight(fig.get_figheight()/1.5)
    name = "angle"

    # read data
    data = np.load("data/"+name+"_push_eval.npz")

    # fist plot the image (get the 5th)
    axs[0, 0].imshow(data["images"][:, :, 5], cmap='gray')

    # disable ticks
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    # set x label t = 0s
    axs[0, 0].set_xlabel("t=0s")

    # then plot the x, y, z over time (one plot for each)
    for j in range(3):
        demon = axs[0, j+1].plot(data["train"][:1000, j], color=colors[9], label="Demonstration")
        evalu_train = axs[0, j+1].plot(data["test"][:1000, j], color=colors[3], label="Evaluation")
        # scatter plot the target with a green x
        target_pos = axs[0, j+1].scatter(1000, target[j], color="green", marker='x', label="target", s=100, zorder=10)
        if j == 0:
            axs[0, j+1].set_ylim([0.4, 0.7])
            axs[0, 1].set_ylabel("EEF")
        elif j == 1:
            axs[0, j+1].set_ylim([0., 0.5])
        else:
            axs[0, j+1].set_ylim([0.25, 0.7])
        decorate_axis(axs[0, j+1])
        axs[0, j+1].set_aspect('auto')
        # get the "t_application" from the data, and draw a dashed vertical line with label "Time of force application"
        axs[0, j+1].axvline(x=data["t_application"][0]*100, color="black", linestyle="--", label="Time of force application")
        axs[0, j+1].axvline(x=data["t_application"][1]*100, color="black", linestyle="--")

        # convert the 1000 ticks to 10s
        axs[0, j+1].set_xticks([0, 250, 500, 750, 1000])
        axs[0, j+1].set_xticklabels([0, 2.5, 5, 7.5, 10])
        axs[0, j+1].set_xlabel("time (s)")

    axs[0, 0].set_title("Non controllable part")
    fig.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1, rect=(0, 0.075, 1, 1))
    plt.legend(bbox_to_anchor=(-0.6, -0.45), loc='lower center', ncol=4, fancybox=True, shadow=True)
    plt.savefig("plots/"+NICE_NAME + ".svg", dpi=100)
    plt.savefig("plots/"+NICE_NAME + ".png", dpi=100)
    plt.savefig("plots/"+NICE_NAME + ".pdf", dpi=100)
    plt.show()
    plt.close()

############## EXPERIMENT 3: White noise, 1x4 ################
def plot_noise():
    TITLE = "Robustness of ANDPs to i.i.d gaussian noise"
    NICE_NAME = "andps_robustness_noise"

    # one row for each image
    rows = 1

    # image, x, y, z
    cols = 4

    fig, axs = plt.subplots(rows, cols, sharey=False, squeeze=False)
    fig.suptitle(TITLE)

    # set figzise height to 1/3 of the params
    fig.set_figheight(fig.get_figheight()/1.5)

    name = "sine"

    # read data
    data_a = np.load("data/"+name+"_noise_1_perc_eval.npz")
    data_b = np.load("data/"+name+"_noise_3_perc_eval.npz")

    # fist plot the image (get the 5th)
    axs[0, 0].imshow(data_a["images"][:, :, 5], cmap='gray')

    # disable ticks
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    # set x label t = 0s
    axs[0, 0].set_xlabel("t=0s")

    # then plot the x, y, z over time (one plot for each)

    for j in range(3):
        demon = axs[0, j+1].plot(data_a["train"][:1000, j], color=colors[9], label="Demonstration")
        evalu_train = axs[0, j+1].plot(data_a["test"][:1000, j], color=colors[3], label="Evaluation with 1% noise")
        evalu_train = axs[0, j+1].plot(data_b["test"][:1000, j], color=colors[5], label="Evaluation with 3% noise")
        

        # scatter plot the target with a green x
        target_pos = axs[0, j+1].scatter(1000, target[j], color="green", marker='x', label="target", s=100, zorder=10)
        if j == 0:
            axs[0, 1].set_ylabel("EEF")
            axs[0, 1].set_ylim([0.5, 0.58])
        elif j == 1:
            axs[0, 2].set_ylim([0., 0.5])
        else:
            axs[0, 3].set_ylim([0.3, 0.55])
        decorate_axis(axs[0, j+1])
        axs[0, j+1].set_aspect('auto')

        # convert the 1000 ticks to 10s
        axs[0, j+1].set_xticks([0, 250, 500, 750, 1000])
        axs[0, j+1].set_xticklabels([0, 2.5, 5, 7.5, 10])
        axs[0, j+1].set_xlabel("time (s)")

    axs[0, 0].set_title("Non controllable part")
    fig.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1, rect=(0, 0.075, 1, 1))
    plt.legend(bbox_to_anchor=(-0.6, -0.45), loc='lower center', ncol=4, fancybox=True, shadow=True)
    plt.savefig("plots/"+NICE_NAME + ".svg", dpi=100)
    plt.savefig("plots/"+NICE_NAME + ".png", dpi=100)
    plt.savefig("plots/"+NICE_NAME + ".pdf", dpi=100)
    plt.show()
    plt.close()

############## EXPERIMENT 4: Change image, 1x4 ################
def plot_reactive():
    TITLE = "ANDPs reactiveness to changes in the non-controllable part of the state"
    NICE_NAME = "andps_reactiveness"

    # now we need a gridspec because we have 2 images for 1 set of plots
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)
    # add a title
    fig.suptitle(TITLE)

    # get the original data for both images
    demo_sine = np.load("data/sine.npz")
    demo_line = np.load("data/line.npz")

    eval_data = np.load("data/line_to_sine_eval.npz")
    # plot the 2 images in the first column of the gridspec
    axs = fig.add_subplot(spec[0, 0])
    axs.set_title("Non controllable part")

    axs.imshow(eval_data["images"][:, :, 5], cmap='gray')
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_xlabel("t=0s")

    axs = fig.add_subplot(spec[1, 0])
    axs.imshow(eval_data["images"][:, :, 100], cmap='gray')
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_xlabel("t="+str(eval_data["t_change"])+"s")

    demo_line = np.load("data/line.npz")["eef_x"]
    demo_sine = np.load("data/sine.npz")["eef_x"]
    
    # repeat the last value for 500 steps and append it to the end of the array
    demo_line = np.append(demo_line, np.repeat(demo_line[-2:, :], 500, axis=0), axis=0)
    demo_sine = np.append(demo_sine, np.repeat(demo_sine[-2:, :], 500, axis=0), axis=0)
    
    # now plot the trajectories

    for j in range(3):
        axs = fig.add_subplot(spec[:, j+1])
        evalu_train = axs.plot(eval_data["test"][:1500, j], color=colors[3], label="Evaluation",zorder=9)

        # demos for both images
        axs.plot(demo_line[:1500, j], color="lightgray", label="Demonstration Linear",zorder=8, alpha=0.5, linestyle="dotted")
        axs.plot(demo_sine[:1500, j], color="gray", label="Demonstration Sinusoidal",zorder=7, alpha=0.5, linestyle="dotted")
        # scatter plot the target with a green x
        target_pos = axs.scatter(1500, target[j], color="green", marker='x', label="target", s=100, zorder=10)
        if j == 0:
            axs.set_ylim([0.5, 0.6])
            axs.set_ylabel("EEF")
        elif j == 1:
            axs.set_ylim([0., 0.5])
        else:
            axs.set_ylim([0.25, 0.7])
        decorate_axis(axs)
        axs.set_aspect('auto')
        # get the "t_application" from the data, and draw a dashed vertical line with label "Time of force application"
        axs.axvline(x=eval_data["t_change"]*100, color="black", linestyle="--", label="Time of image change")

        # convert the 1000 ticks to 10s
        axs.set_xticks([0, 250, 500, 750, 1000, 1250, 1500])
        axs.set_xticklabels([0, 2.5, 5, 7.5, 10, 12.5, 15])
        axs.set_xlabel("time (s)")

    fig.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1, rect=(0, 0.075, 1, 1))
    plt.legend(bbox_to_anchor=(-0.6, -0.28), loc='lower center', ncol=5, fancybox=True, shadow=True)
    plt.savefig("plots/"+NICE_NAME + ".svg", dpi=100)
    plt.savefig("plots/"+NICE_NAME + ".png", dpi=100)
    plt.savefig("plots/"+NICE_NAME + ".pdf", dpi=100)
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("Which plot do you want to generate?")
    print("1: Multi-Task")
    print("2: Force")
    print("3: Noise")
    print("4: Reactiveness")
    print("5: All")
    inp = input()
    if inp == "1":
        plot_multi_task()
    elif inp == "2":
        plot_force()
    elif inp == "3":
        plot_noise()
    elif inp == "4":
        plot_reactive()
    elif inp == "5":
        plot_multi_task()
        plot_force()
        plot_noise()
        plot_reactive()
    else:
        print("Invalid input")