#import tfomics
import logomaker as lm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Union, Optional
from ..preprocess._helpers import _get_vocab


def plot_attribution_logo(
    attrs: np.ndarray,
    inputs: Optional[np.ndarray] = None,
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
    
    if inputs is not None:
        if inputs.shape[-1] != 4:
            inputs = inputs.T
        attrs = attrs * inputs
    
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
    inputs: Optional[np.ndarray] = None,
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
    if inputs is None:
        inputs = [None] * n_attrs
        
    for i in range(n_attrs):
        plot_attribution_logo(
            attrs[i],
            inputs=inputs[i],
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


def plot_attribution_logo_heatmap(
    attrs: np.ndarray,
    inputs: Optional[np.ndarray] = None,
    flip_sign: bool = False,    
    vocab: str = "DNA",
    figsize: tuple = (10, 3)
):
    
    # Get the vocab
    vocab = _get_vocab(vocab)

    if flip_sign:
        logo_attrs = -attrs
    if inputs is not None:
        logo_attrs = logo_attrs.sum(axis=0) * inputs
    if logo_attrs.shape[-1] != 4:
        logo_attrs = logo_attrs.T
        
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
