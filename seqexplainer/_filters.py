import numpy as np
import pandas as pd
import logomaker as lm
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from seqpro._helpers import _get_vocab
from ._utils import _k_largest_index_argsort

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

def pfm_to_df(
    pfm,
    vocab: str = "DNA",
):
    vocab = _get_vocab(vocab)
    pfm_df = pd.DataFrame(pfm, columns=vocab, index=range(pfm.shape[0]))
    return pfm_df

def pfms_to_df_dict(
    pfms,
    vocab: str = "DNA",
):
    vocab = _get_vocab(vocab)
    pfm_dfs = {}
    for i, pfm in enumerate(pfms):
        pfm_df = pd.DataFrame(pfm, columns=vocab, index=range(pfms.shape[1]))
        pfm_dfs[f"filter{i}"] = pfm_df
    return pfm_dfs

def pfms_to_ppms(
    pfms,
    vocab: str = "DNA",
    pseudocount: int = 1
):
    vocab = _get_vocab(vocab)
    if pfms.shape[1] == len(vocab): 
        pfms = pfms[np.newaxis, :, :]
    pfms += pseudocount
    ppms = pfms / pfms.sum(axis=2, keepdims=True)
    return ppms

def ppms_to_pwms(
    ppms,
    bg = None,
    vocab: str = "DNA",
    pseudocount = TINY
):
    vocab = _get_vocab(vocab)
    if ppms.shape[1] == len(vocab): 
        ppms = ppms[np.newaxis, :, :]
    if bg is None:
        bg = np.ones(len(vocab)) / len(vocab)
    bg = np.tile(bg, (ppms.shape[0], ppms.shape[1], 1))
    pwms = np.log2(ppms + pseudocount) - np.log2(bg + pseudocount)
    return pwms

def per_position_ic(
    ppms, 
    vocab="DNA",
    bg=None,
    pseudocount=TINY
):
    vocab = _get_vocab(vocab)
    if ppms.shape[1] == len(vocab): 
        ppms = ppms[np.newaxis, :, :]
    if bg is None:
        bg = np.ones(len(vocab)) / len(vocab)
    bg = np.tile(bg, (ppms.shape[0], ppms.shape[1], 1))
    info_vecs = (ppms * (np.log2(ppms + pseudocount) - np.log2(bg + pseudocount))).sum(axis=2)
    return info_vecs

def ppms_to_igms(
    ppms,
    vocab = "DNA",
    bg=None,
    pseudocount=TINY
):
    info_vecs = per_position_ic(ppms, vocab=vocab, bg=bg, pseudocount=pseudocount)
    info_mtxs = (ppms * info_vecs[:, :, np.newaxis]) 
    return info_mtxs

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
