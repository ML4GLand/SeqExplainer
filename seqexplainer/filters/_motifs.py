from ..preprocess._helpers import _get_vocab
import numpy as np
TINY = np.finfo(float).tiny

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