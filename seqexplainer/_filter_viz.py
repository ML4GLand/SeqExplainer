import torch
import numpy as np
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from seqpro import _get_vocab
from ._utils import _k_largest_index_argsort

def _get_filter_activators(
    activations: np.ndarray,
    sequences: np.ndarray,
    kernel_size: int,
    num_filters: int = None,
    method: str = "Alipanahi15",
    threshold: float = 0.5,
    num_seqlets: int = 100,
):
    """
    Get the sequences that activate a filter the most using a passed in method.
    We currently implement two methods, Alipanahi15 and Minnoye20.
    
    Parameters
    ----------
    activations : np.ndarray
        The activations from the layer.
    sequences : np.ndarray
        The sequences corresponding to the activations in a numpy array.
    kernel_size : int
        The kernel size of the layer.
    num_filters : int, optional
        The number of filters to get seqlets for, by default None
    method : str, optional
        The method to use, by default "Alipanahi15"
    threshold : float, optional
        The threshold for filtering activations, by default 0.5
    num_seqlets : int, optional
        The number of seqlets to get, by default 100

    Returns
    -------
    np.ndarray
        The sequences that activate the filter the most.

    Note
    ----
    We currently only use forward sequences for computing activations of the layer,
    we do not currenlty include reverse complements
    """
    num_filters = num_filters if num_filters is not None else activations.shape[1]
    if method == "Alipanahi15":
        assert (threshold is not None), "Threshold must be specified for Alipanahi15 method."
        filter_activators = []
        for i, filt in tqdm(
            enumerate(range(num_filters)),
            desc=f"Getting filter activators for {num_filters} filters",
            total=num_filters,
        ):
            single_filter = activations[:, filt, :]
            max_val = np.max(single_filter)
            activators = []
            for i in range(len(single_filter)):
                starts = np.where(single_filter[i] >= max_val * threshold)[0]
                for start in starts:
                    activators.append(sequences[i][start : start + kernel_size])
            filter_activators.append(activators)
    elif method == "Minnoye20":
        assert (num_seqlets is not None), "num_seqlets must be specified for Minnoye20 method."
        filter_activators = []
        for i, filt in tqdm(
            enumerate(range(num_filters)),
            desc=f"Getting filter activators for {num_filters} filters",
            total=num_filters,
        ):
            single_filter = activations[:, filt, :]
            inds = _k_largest_index_argsort(single_filter, num_seqlets)
            filter_activators.append(
                [
                    seq[inds[i][1] : inds[i][1] + kernel_size]
                    for i, seq in enumerate(sequences[inds[:, 0]])
                ]
            )
    return filter_activators

def _get_pfms(
    filter_activators: np.ndarray,
    kernel_size: int,
    vocab: str = "DNA",
):  
    """
    Generate position frequency matrices for the maximal activating seqlets in filter_activators 

    Parameters
    ----------
    filter_activators : np.ndarray
        The sequences that activate the filter the most.
    kernel_size : int
        The kernel size of the layer.
    vocab : str, optional
        The vocabulary to use, by default "DNA"

    Returns
    -------
    np.ndarray
        The position frequency matrices for the maximal activating seqlets in filter_activators
    """
    filter_pfms = {}
    vocab = _get_vocab(vocab)
    for i, activators in tqdm(
        enumerate(filter_activators),
        total=len(filter_activators),
        desc="Getting PFMs from filters",
    ):
        pfm = {
            vocab[0]: np.zeros(kernel_size),
            vocab[1]: np.zeros(kernel_size),
            vocab[2]: np.zeros(kernel_size),
            vocab[3]: np.zeros(kernel_size),
            "N": np.zeros(kernel_size),
        }
        for seq in activators:
            for j, nt in enumerate(seq):
                pfm[nt][j] += 1
        filter_pfm = pd.DataFrame(pfm)
        filter_pfm = filter_pfm.drop("N", axis=1)
        filter_pfms[i] = filter_pfm
        filter_pfms[i] = filter_pfms[i].div(filter_pfms[i].sum(axis=1), axis=0)
    return filter_pfms