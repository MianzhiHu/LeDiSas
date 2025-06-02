import seaborn as sns
import matplotlib.pyplot as plt

# Side-by-side plot
def side_by_side_plot(data, x, y, x_label, y_label, title, block_div=None):
    global break_positions
    g = sns.FacetGrid(data, col="Group", sharey=True, height=4, aspect=1.5)
    g.map_dataframe(sns.lineplot, x=x, y=y)
    window_column = "continuous_window" if "continuous_window" in data.columns else "window_id"

    if block_div:
        break_positions = set()

        # Loop over each participant’s windows in ascending order
        for pid, df in data.groupby("participant_id"):
            df = df.sort_values(window_column, ignore_index=True)

            # Identify where the block changes
            block_shift = df["Block"].shift(1)
            change_mask = (df["Block"] != block_shift) & (block_shift.notna())
            new_block_windows = df.loc[change_mask, window_column].values
            break_positions.update(new_block_windows)

        break_positions = sorted(break_positions)

    for ax in g.axes.flatten():
        if block_div:
            for pos in break_positions:
                ax.axvline(x=pos, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    g.set_titles(col_template="{col_name}")  # Use the Group name as each panel’s title
    g.fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'./figures/{title}.png', dpi=600)
    plt.show()