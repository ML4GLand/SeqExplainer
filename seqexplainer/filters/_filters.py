import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .._utils import _k_largest_index_argsort

TINY = np.finfo(float).tiny

def get_activators_n_seqlets(
    activations,
    sequences,
    kernel_size,
    num_seqlets = 100,
    num_filters=None
):
    num_filters = num_filters if num_filters is not None else activations.shape[1]
    filter_activators = []
    for _, filter_num in tqdm(enumerate(range(num_filters)), desc=f"Getting filter activators for {num_filters} filters", total=num_filters,):
            single_filter = activations[:, filter_num, :]
            inds = _k_largest_index_argsort(single_filter, num_seqlets)
            filter_activators.append([seq[:, inds[i][1] : inds[i][1] + kernel_size] for i, seq in enumerate(sequences[inds[:, 0]])])
    return np.array(filter_activators)

def get_activators_max_seqlets(
    activations,
    sequences,
    kernel_size,
    activation_threshold = 0.5,
    num_filters=None
):
    num_filters = num_filters if num_filters is not None else activations.shape[1]
    filter_activators = []
    for _, filter_num in tqdm(enumerate(range(num_filters)), desc=f"Getting filter activators for {num_filters} filters", total=num_filters,):
            single_filter = activations[:, filter_num, :]
            inds = np.where(single_filter > activation_threshold * single_filter.max())
            filter_activators.append([seq[:, inds[1][i] : inds[1][i] + kernel_size] for i, seq in enumerate(sequences[inds[0]])])
    return filter_activators

def get_pfms(
    filter_activators,
):
    if isinstance(filter_activators, list):
        pfms = []
        for filter_acts in filter_activators:
             pfms.append(np.array(filter_acts).sum(axis=0))
        pfms = np.array(pfms)
    else:
        pfms = filter_activators.sum(axis=1)
    return pfms.transpose(0, 2, 1)
