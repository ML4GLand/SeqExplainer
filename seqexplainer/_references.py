from typing import Callable, List, Union

import torch
from numpy.typing import NDArray
from seqpro import dinuc_shuffle_seqs


def zero_ref_inputs(
    inputs: Union[torch.Tensor, NDArray]
):
    """Return a Tensor of zeros with the same shape as inputs
    
    Parameters
    ----------
    inputs : torch.Tensor or numpy.ndarray
        The input sequences to be used to generate a set of zero reference sequences.

    Returns
    -------
    refs : torch.Tensor
        A Tensor of zeros with the same shape as inputs.

    """
    refs = torch.zeros(inputs.shape)
    return refs

def random_ref_inputs(
    inputs: Union[torch.Tensor, NDArray]
):
    """Return a Tensor of random one hot encoded inputs
    
    Parameters
    ----------
    inputs : torch.Tensor or numpy.ndarray
        The input sequences to be used to generate a random set of reference sequences.

    Returns
    -------
    refs : torch.Tensor
        A Tensor of random one hot encoded sequences with the same shape as inputs.
    """
    ref_tokens = torch.randint(4, size=(inputs.shape[0], inputs.shape[2]))
    refs = torch.nn.functional.one_hot(ref_tokens, num_classes=4).float().transpose(1,2)
    return refs

def shuffle_ref_inputs(inputs):
    """Return a Tensor of shuffled inputs
    
    Parameters
    ----------
    inputs : torch.Tensor or numpy.ndarray
        The input sequences to be used to generate a set of shuffled reference sequences.
    
    Returns
    -------
    refs : torch.Tensor
        A Tensor of shuffled sequences with the same shape as inputs.
    """
    refs = inputs[torch.randperm(inputs.shape[0])]
    return refs

def dinuc_shuffle_ref_inputs(
    inputs: NDArray
):
    """Return a Tensor of dinucleotide shuffled inputs
    
    Parameters
    ----------
    inputs : numpy.ndarray
        The input sequences to be used to generate a set of dinucleotide shuffled reference sequences.

    Returns
    -------
    refs : torch.Tensor
        A Tensor of dinucleotide shuffled sequences with the same shape as inputs.

    Note
    ----
    This function is a wrapper for the dinuc_shuffle_seqs function from the seqpro package and currently only works with numpy arrays.
    """
    refs = torch.Tensor(dinuc_shuffle_seqs(inputs))
    return refs

def gc_ref_inputs(
    inputs: Union[torch.Tensor, NDArray],
    bg: str = "uniform",
    uniform_dist: List[float] = [0.25, 0.25, 0.25, 0.25]
):
    """Return a Tensor of GC content with the same shape as inputs
    
    Parameters
    ----------
    inputs : torch.Tensor or numpy.ndarray
        The input sequences to be used to generate a set of GC content reference sequences.
    bg : str, optional
        The background distribution to use for the GC content reference sequences. Options are "uniform", "batch", or "seq". Default is "uniform".
    uniform_dist : list, optional
        If bg is "uniform", the uniform distribution to use for the GC content reference sequences. Default is [0.25, 0.25, 0.25, 0.25].

    Returns
    -------
    refs : torch.Tensor
        A Tensor of GC content sequences with the same shape as inputs.

    Note
    ----
    TODO: Add support for numpy array inputs and seq dimension
    """
    if bg is "uniform":
        dists = torch.Tensor(uniform_dist)
        refs = dists.expand(inputs.shape[0], inputs.shape[2], 4).transpose(2, 1)
    elif bg is "batch":
        dists = torch.Tensor(inputs.sum(0)/inputs.shape[0])
        refs = dists.expand(inputs.shape[0], dists.shape[0], dists.shape[1])
    elif bg is "seq":
        dists = torch.Tensor(inputs.sum(dim=2)/inputs.shape[2])
        refs = dists.expand(inputs.shape[2], dists.shape[0], dists.shape[1]).permute(1, 2, 0)
    else:
        raise ValueError(f"Background distribution {bg} not in ['uniform', 'batch', 'seq']")
    return refs 

def profile_ref_inputs(
    inputs: torch.Tensor
):
    """Return a Tensor with a matched nucleotide profile as inputs
    
    Parameters
    ----------
    inputs : torch.Tensor
        The input sequences to be used to generate a set of nucleotide profile reference sequences.
    
    Returns
    -------
    refs : torch.Tensor
        A Tensor of nucleotide profile sequences with the same shape as inputs.

    Note
    ----
    TODO: This function currently only works with torch Tensors.
    """
    seq_model = torch.mean(torch.squeeze(inputs), axis=0)
    seq_model /= torch.sum(seq_model, axis=0, keepdims=True)

    L = seq_model.shape[1]
    refs = torch.zeros(inputs.shape)

    for n in range(inputs.shape[0]):

        # generate uniform random number for each nucleotide in sequence
        Z = torch.rand(L)

        # calculate cumulative sum of the probabilities
        cum_prob = seq_model.cumsum(axis=0)

        # find bin that matches random number for each position
        for l in range(L):
            index = [j for j in range(4) if Z[l] < cum_prob[j, l]][0]
            refs[n, index, l] = 1 
    return refs
    
REFERENCE_REGISTRY = {
    "zero": zero_ref_inputs,
    "random": random_ref_inputs,
    "shuffle": shuffle_ref_inputs,
    "dinuc_shuffle": dinuc_shuffle_ref_inputs,
    "gc": gc_ref_inputs,
    "profile": profile_ref_inputs,
}

def get_reference(
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
