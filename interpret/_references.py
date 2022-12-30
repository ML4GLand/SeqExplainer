import torch
import numpy as np
from typing import Union, Callable
from ._seq_preprocess import dinuc_shuffle_seqs

def zero_ref_inputs(inputs):
    """Return a Tensor of zeros with the same shape as inputs"""
    refs = torch.zeros(inputs.shape)
    return refs

def gc_ref_inputs(inputs, bg="uniform"):
    """Return a Tensor of GC content with the same shape as inputs"""
    if bg is "uniform":
        dists = torch.Tensor([0.3, 0.2, 0.2, 0.3])
        refs = dists.expand(inputs.shape[0], inputs.shape[2], 4).transpose(2, 1)
    elif bg is "batch":
        dists = torch.Tensor(inputs.sum(0)/inputs.shape[0])
        refs = dists.expand(inputs.shape[0], dists.shape[0], dists.shape[1])
    elif bg is "seq":
        dists = torch.Tensor(inputs.sum(dim=2)/inputs.shape[2])
        refs = dists.expand(inputs.shape[2], dists.shape[0], dists.shape[1]).permute(1, 2, 0)
    return refs 

def dinuc_shuffle_ref_inputs(inputs):
    """Return a Tensor of dinucleotide shuffled inputs"""
    refs = torch.Tensor(dinuc_shuffle_seqs(inputs.detach().cpu().numpy()))
    return refs

REFERENCE_REGISTRY = {
    "zero": zero_ref_inputs,
    "gc": gc_ref_inputs,
    "shuffle": dinuc_shuffle_ref_inputs,
}

def _get_reference(
    inputs: torch.Tensor,
    method: Union[str, Callable],
    device: str = "cpu"
):
    """
    Returns torch.Tensor reference
    """
    if isinstance(method, str):
        if method not in REFERENCE_REGISTRY:
            raise ValueError(f"Reference method {method} not in {list(REFERENCE_REGISTRY.keys())}")
        if isinstance(inputs, tuple):
            reference = tuple([REFERENCE_REGISTRY[method](inputs[i]).requires_grad_().to(device) for i in range(len(inputs))])
        else:
            reference = REFERENCE_REGISTRY[method](inputs).requires_grad_().to(device)
    elif callable(method):
        if isinstance(inputs, tuple):
            reference = tuple([method(inputs[i]).requires_grad_().to(device) for i in range(len(inputs))])
        else:
            reference = method(inputs).requires_grad_().to(device)
    else:
        raise ValueError(f"Reference method {method} not in {list(REFERENCE_REGISTRY.keys())}")
    return reference