#import tfomics
import logomaker as lm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Union, Optional, List


def plot_attribution_logo(
    attrs: np.ndarray,
    vocab: Union[str, List] = "ACGT",
    highlights: list = [],
    highlight_colors: list = ["lavenderblush", "lightcyan", "honeydew"],
    height_scaler: float = 1.8,
    title: str ="",
    ylab: str = "Attribution",
    xlab: str = "Position",
    **kwargs
):
    if isinstance(vocab, str):
        vocab = [c for c in vocab]
    if attrs.shape[-1] != len(vocab):
        assert attrs.shape[0] == len(vocab), "attrs must be (L, A) or (A, L)"
        logo_attrs = attrs.T.copy()

    # Create Logo object
    df = pd.DataFrame(logo_attrs, columns=vocab)
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

def plot_attribution_logo_heatmap(
    attrs: np.ndarray,
    inputs: Optional[np.ndarray] = None,
    flip_sign: bool = False,    
    vocab: Union[str, List] = "ACGT",
    figsize: tuple = (10, 3)
):
    
    # Get the vocab
    if isinstance(vocab, str):
        vocab = [c for c in vocab]

    #
    logo_attrs = -attrs if flip_sign else attrs
    logo_attrs = logo_attrs.sum(axis=0) * inputs if inputs is not None else logo_attrs
    if logo_attrs.shape[-1] != len(vocab):
        assert logo_attrs.shape[0] == len(vocab), "attrs must be (L, A) or (A, L)"
        logo_attrs = logo_attrs.T.copy()
        
    # Create fig and axes
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Create Logo object
    
    df = pd.DataFrame(logo_attrs, columns=vocab)
    df.index.name = "pos"
    y_max = np.max(float(np.max(df.values)), 0)
    y_min = np.min(float(np.min(df.values)), 0)
    nn_logo = lm.Logo(df, ax=ax[0])

    # style using Logo methods
    nn_logo.style_spines(visible=False)

    # style using Axes methods
    nn_logo.ax.set_xlim([0, len(df)])
    nn_logo.ax.set_xticks([])
    nn_logo.ax.set_ylim([y_min, y_max])
    nn_logo.ax.set_yticks([])
    nn_logo.ax.set_ylabel("Attribution")

    # Create the heatmap
    sns.heatmap(attrs, cmap='coolwarm', cbar=False, center=0, ax=ax[1])
    ax[1].set_yticklabels(vocab, rotation=0)
    ax[1].set_xticks([])

    # Want to add a colorbar to the whole figure along the right side
    fig.colorbar(ax[1].get_children()[0], ax=ax, location='right', use_gridspec=True, pad=0.05)
    
    plt.show()
