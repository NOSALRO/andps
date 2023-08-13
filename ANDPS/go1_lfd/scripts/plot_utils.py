def decorate_axis(ax, remove_left=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)

    ax.grid(axis='y', color="0.9", linestyle='solid', linewidth=0.8)
    ax.grid(axis='x', color="0.9", linestyle='solid', linewidth=0.8)
    ax.set_facecolor("#FBFEFB")
    # ax.grid.set
    ax.set_axisbelow(True)
