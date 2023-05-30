#import tfomics
import logomaker as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from ._utils import _make_dirs

def umap_plot(umap_data, umap1=0, umap2=1, color="b", loadings=None, labels=None, n=5):
    xs = umap_data[:, umap1]
    ys = umap_data[:, umap2]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    ax = plt.scatter(xs * scalex, ys * scaley, c=color)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("UMAP{}".format(1))
    plt.ylabel("UMAP{}".format(2))
    plt.show()
    return ax
    

def plot_attribution_logo(
    attrs: np.ndarray,
    vocab: str = "DNA",
    highlights: list = [],
    highlight_colors: list = ["lavenderblush", "lightcyan", "honeydew"],
    height_scaler: float = 1.8,
    title: str ="",
    ylab: str = "Attribution",
    xlab: str = "Position",
    **kwargs
):
    vocab = _get_vocab(vocab)
    if attrs.shape[-1] != 4:
        attrs = attrs.T
    
    # Create Logo object
    df = pd.DataFrame(attrs, columns=vocab)
    df.index.name = "pos"
    y_max = np.max(float(np.max(df.values)) * height_scaler, 0)
    y_min = np.min(float(np.min(df.values)) * height_scaler, 0)
    nn_logo = lm.Logo(df, **kwargs)

    # style using Logo methods
    nn_logo.style_spines(visible=False)
    nn_logo.style_spines(spines=["left"], visible=True, bounds=[y_min, y_max])

    # style using Axes methods
    nn_logo.ax.set_xlim([0, len(df)])
    nn_logo.ax.set_xticks([])
    nn_logo.ax.set_ylim([y_min, y_max])
    nn_logo.ax.set_ylabel(ylab)
    nn_logo.ax.set_xlabel(xlab)
    nn_logo.ax.set_title(title)
    for i, highlight in enumerate(highlights):
        nn_logo.highlight_position_range(
            pmin=highlight[0], 
            pmax=highlight[1], 
            color=highlight_colors[i]
        )
    
def plot_attribution_logos(
    attrs: np.ndarray,
    vocab: str = "DNA",
    height_scaler: float = 1.8,
    title: str ="",
    ylab: str = "Attribution",
    xlab: str = "Position",
    figsize: tuple = (10, 10),
    ncols: int = 1,
    **kwargs
):
    n_attrs = attrs.shape[0]
    nrows = int(np.ceil(n_attrs / ncols))
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()
    for i in range(n_attrs):
        plot_attribution_logo(
            attrs[i],
            vocab=vocab,
            height_scaler=height_scaler,
            title=title,
            ylab=ylab,
            xlab=xlab,
            ax=axes[i],
            **kwargs
        )
    plt.tight_layout()
    plt.show()

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