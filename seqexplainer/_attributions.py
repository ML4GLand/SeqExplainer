from typing import Callable, Union

import logomaker as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift, DeepLiftShap, GradientShap, InputXGradient
from seqpro._helpers import _get_vocab
from tqdm.auto import tqdm

from ._perturb import perturb_seq_torch
from ._references import get_reference
from ._utils import _get_oned_contribs, _model_to_device, pca, umap


# Reference vs output difference methods
def delta(y, reference):
    """Difference between output and reference"""
    return (y - reference).sum(axis=-1)

def l1(y, reference):
    """L1 norm between output and reference"""
    return (y - reference).abs().sum(axis=-1)

def l2(y, reference):
    """L2 norm between output and reference"""
    return torch.sqrt(torch.square(y - reference).sum(axis=-1))

DIFF_REGISTRY = {
    "delta": delta,
    "l1": l1,
    "l2": l2,
}

# In silico mutagenesis methods
def _naive_ism(
    model, 
    inputs, 
    target=None, 
    batch_size=128, 
    diff_type="delta", 
    device="cpu"
):

    # Get the number of sequences, choices, and sequence length
    n_seqs, n_choices, seq_len = inputs.shape
    n = seq_len * (n_choices - 1)
    X_idxs = inputs.argmax(axis=1)

    # If target not provided aggregate over all outputs
    target = np.arange(model.output_dim) if target is None else target
    
    # Move the model to eval mode
    model = model.eval() 

    # Get the reference output
    reference = model(inputs)[:, target].unsqueeze(1)
    batch_starts = np.arange(0, n, batch_size)

    # Get the change in output for each perturbation
    isms = []
    for i in range(n_seqs):
        X = perturb_seq_torch(inputs[i])
        y = []
        for start in batch_starts:
            X_ = X[start : start + batch_size]
            y_ = model(X_)[:, target].unsqueeze(1)
            y.append(y_)
            del X_
        y = torch.cat(y)
        ism = DIFF_REGISTRY[diff_type](y, reference[i])
        isms.append(ism)
        
        if device[:4] == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # Clean up the output to be (N, A, L)
    isms = torch.stack(isms).to(device)
    isms = isms.reshape(n_seqs, seq_len, n_choices - 1)
    j_idxs = torch.arange(n_seqs * seq_len)
    X_ism = torch.zeros(n_seqs * seq_len, n_choices, device=device)
    for i in range(1, n_choices):
        i_idxs = (X_idxs.flatten() + i) % n_choices
        X_ism[j_idxs, i_idxs] = isms[:, :, i - 1].flatten()

    X_ism = X_ism.reshape(n_seqs, seq_len, n_choices).permute(0, 2, 1)
    return X_ism

ISM_REGISTRY = {
    "NaiveISM": _naive_ism,
}

def _ism_attributions(
    model: torch.nn.Module, 
    inputs: Union[tuple, torch.Tensor],
    method: Union[str, Callable],
    target: int = None,
    device: str = "cpu",
    **kwargs
):
    attrs = ISM_REGISTRY[method](model=model, inputs=inputs, target=target, device=device, **kwargs)
    return attrs

# Captum methods
CAPTUM_REGISTRY = {
    "InputXGradient": InputXGradient,
    "DeepLift": DeepLift,
    "DeepLiftShap": DeepLiftShap,
    "GradientShap": GradientShap,
}

def _captum_attributions(
    model: torch.nn.Module,
    inputs: tuple,
    method: str,
    target: int = 0,
    device: str = "cpu",
    **kwargs
):
    """
    """
    if isinstance(inputs, np.ndarray):
        inputs = torch.tensor(inputs)
    attributor = CAPTUM_REGISTRY[method](model)
    attrs = attributor.attribute(inputs=inputs, target=target, **kwargs)
    return attrs

# Attribution methods -- combination of above
ATTRIBUTIONS_REGISTRY = {
    "NaiveISM": _ism_attributions,
    "InputXGradient": _captum_attributions,
    "DeepLift": _captum_attributions,
    "GradientShap": _captum_attributions,
    "DeepLiftShap": _captum_attributions,
}

def attribute(
    model,
    inputs: torch.Tensor,
    method: Union[str, Callable],
    reference_type: str = None,
    target: int = 0,
    batch_size: int = 128,
    device: str = "cpu",
):
    # Disable cudnn for faster computations 
    torch.backends.cudnn.enabled = False
    
    # Put model on device
    model = _model_to_device(model, device)
    
    # Check type of inputs
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).float()

    # Create an empty list to hold attributions
    attrs = []
    starts = np.arange(0, inputs.shape[0], batch_size)

    # Loop through batches and compute attributions
    for _, start in tqdm(
        enumerate(starts),
        total=len(starts),
        desc=f"Computing attributions on batches of size {batch_size}",
    ):
        # Grab the current batch
        inputs_ = inputs[start : start + batch_size]

        # Put inputs on device
        if isinstance(inputs, tuple):
            inputs_ = tuple([i.requires_grad_().to(device) for i in inputs_])
        else:
            inputs_ = inputs_.requires_grad_().to(device)
        
        # Add reference if needed
        kwargs = {}
        if reference_type is not None:
            kwargs["baselines"] =  get_reference(inputs_, reference_type, device)

        # Get attributions and append
        curr_attrs = ATTRIBUTIONS_REGISTRY[method](
            model=model,
            inputs=inputs_,
            method=method,
            target=target,
            device=device,
            **kwargs
        ).detach().cpu()
        attrs.append(curr_attrs)

    # Concatenate the attributions
    attrs = torch.cat(attrs).numpy()

    # Return attributions
    return attrs

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

def attribution_pca(
    one_hot,
    hypothetical_contribs, 
    n_comp: int = 30, 
):
    oned_contr = _get_oned_contribs(one_hot, hypothetical_contribs)
    pca_obj, pca_df = pca(oned_contr, n_comp=n_comp)
    return pca_obj, pca_df

def attribution_umap(
    one_hot,
    hypothetical_contribs, 
):
    oned_contr = _get_oned_contribs(one_hot, hypothetical_contribs)
    umap_obj, umap_df = umap(oned_contr)
    return umap_obj, umap_df

def gradient_correction(
    hypothetical_contribs,
    one_hot,
    reference,
    diff_type="delta",
):
    pass