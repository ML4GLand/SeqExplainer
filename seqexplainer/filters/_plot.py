import logomaker as lm
import matplotlib.pyplot as plt
from ._motifs import pfm_to_df, pfms_to_df_dict

def plot_filter_logo(
    mtx,
    mtx_type="counts",
    vocab="DNA",
    title=None
):
    df = pfm_to_df(mtx, vocab=vocab)
    plot_df = lm.transform_matrix(df, from_type=mtx_type, to_type="information", pseudocount=1)    
    logo = lm.Logo(plot_df)
    logo.style_xticks(spacing=5, anchor=25, rotation=45, fmt="%d", fontsize=14)
    logo.style_spines(visible=False)
    logo.style_spines(spines=["left", "bottom"], visible=True, linewidth=2)
    logo.ax.set_ylim([0, 2])
    logo.ax.set_yticks([0, 1, 2])
    logo.ax.set_yticklabels(["0", "1", "2"])
    logo.ax.set_ylabel("bits")
    logo.ax.set_title(title)
    plt.show()

def plot_filter_logos(
    mtxs,
    vocab="DNA",
    mtx_type="counts",
    title=None
):
    df_dict = pfms_to_df_dict(mtxs, vocab=vocab)
    for _, df in df_dict.items():
        plot_df = lm.transform_matrix(df, from_type=mtx_type, to_type="information", pseudocount=1)    
        logo = lm.Logo(plot_df)
        logo.style_xticks(spacing=5, anchor=25, rotation=45, fmt="%d", fontsize=14)
        logo.style_spines(visible=False)
        logo.style_spines(spines=["left", "bottom"], visible=True, linewidth=2)
        logo.ax.set_ylim([0, 2])
        logo.ax.set_yticks([0, 1, 2])
        logo.ax.set_yticklabels(["0", "1", "2"])
        logo.ax.set_ylabel("bits")
        logo.ax.set_title(title)
        plt.show()
