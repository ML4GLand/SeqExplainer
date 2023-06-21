import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .._utils import _k_largest_index_argsort

TINY = np.finfo(float).tiny

def get_activators_n_seqlets(
    activations,
    sequences,
    kernel_size,
    padding=0,
    num_seqlets = 100,
    num_filters=None
):
    
    num_filters = num_filters if num_filters is not None else activations.shape[1]
    filter_activators = []
    for _, filter_num in tqdm(enumerate(range(num_filters)), desc=f"Getting filter activators for {num_filters} filters", total=num_filters,):
        single_filter = activations[:, filter_num, :]
        inds = _k_largest_index_argsort(single_filter, num_seqlets)
        single_filter_activators = []
        for i, seq in enumerate(sequences[inds[:, 0]]):
            start = inds[i][1] - padding
            end = inds[i][1] + kernel_size - padding
            if end > seq.shape[1]:
                pad_len = end - seq.shape[1]
                seq = np.pad(seq, ((0, 0), (0, pad_len)), constant_values=0)
            elif start < 0:
                pad_len = -start
                seq = np.pad(seq, ((0, 0), (pad_len, 0)), constant_values=0)
                start = 0
                end = end + pad_len
            activator = seq[:, start:end]
            single_filter_activators.append(activator)
        filter_activators.append(single_filter_activators)
    return np.array(filter_activators)

def get_activators_max_seqlets(
    activations,
    sequences,
    kernel_size,
    padding=0,
    activation_threshold = 0.5,
    num_filters=None
):
    num_filters = num_filters if num_filters is not None else activations.shape[1]
    filter_activators = []
    for _, filter_num in tqdm(enumerate(range(num_filters)), desc=f"Getting filter activators for {num_filters} filters", total=num_filters,):
        single_filter = activations[:, filter_num, :]
        inds = np.where(single_filter > activation_threshold * single_filter.max())
        single_filter_activators = []
        for i, pos in zip(inds[0], inds[1]):
            seq = sequences[i]
            start = pos - padding
            end = pos + kernel_size - padding
            if end > seq.shape[1]:
                pad_len = end - seq.shape[1]
                seq = np.pad(seq, ((0, 0), (0, pad_len)), constant_values=0)
            elif start < 0:
                pad_len = -start
                seq = np.pad(seq, ((0, 0), (pad_len, 0)), constant_values=0)
                start = 0
                end = end + pad_len
            activator = seq[:, start:end]
            single_filter_activators.append(activator)
        filter_activators.append(single_filter_activators)
    return filter_activators

def get_pfms(
    filter_activators,
    kernel_size=13
):
    if isinstance(filter_activators, list):
        pfms = []
        for i, filter_acts in enumerate(filter_activators):
            if len(filter_acts) == 0:
                 print("No activators found for filter", i, "creating uniform pfm")
                 pfms.append(np.ones((4, kernel_size)))
            else: 
                pfms.append(np.array(filter_acts).sum(axis=0))
        pfms = np.array(pfms)
    else:
        pfms = filter_activators.sum(axis=1)
    return pfms.transpose(0, 2, 1)
