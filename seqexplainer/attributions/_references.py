from typing import Callable, List, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from ..preprocess._preprocess import dinuc_shuffle_seqs


def zero_ref_inputs(
    inputs: NDArray
) -> NDArray:
    """Return a NumPy array of zeros with the same shape as the inputs.

    Inputs are expected to be one-hot encoded sequences with shape (N, A, L) 
    where N is the number of sequences, A is the number of nucleotides, and L is the length of the sequences.
    
    Parameters
    ----------
    inputs : NDArray
        The input sequences to be used to generate a set of zero reference sequences.

    Returns
    -------
    refs : NDArray
        A NumPy array of zeros with the same shape as inputs.
    """
    N, A, L = inputs.shape
    refs = np.zeros((N, A, L))
    return refs

def random_ref_inputs(
    inputs: NDArray
) -> NDArray:
    """Return a NumPy array of random one-hot encoded sequences with the same shape as the inputs.

    Inputs are expected to be one-hot encoded sequences with shape (N, A, L) 
    where N is the number of sequences, A is the number of nucleotides, and L is the length of the sequences.

    Parameters
    ----------
    inputs : NDArray
        The input sequences to be used to generate a random set of reference sequences.

    Returns
    -------
    refs : NDArray
        A NumpPy array of random one hot encoded sequences with the same shape as inputs.
    """
    N, A, L = inputs.shape
    ref_tokens = np.random.randint(4, size=(N, L))
    refs = np.eye(A)[ref_tokens]
    return refs

def shuffle_ref_inputs(
    inputs: NDArray
) -> NDArray:
    """Return a NumPy array of shuffled inputs with the same shape as the inputs.

    Inputs are expected to be one-hot encoded sequences with shape (N, A, L)
    where N is the number of sequences, A is the number of nucleotides, and L is the length of the sequences.
    Each sequence will be shuffled independently along the last axis.
    
    Parameters
    ----------
    inputs : NDArray
        The input sequences to be used to generate a set of shuffled reference sequences.
    
    Returns
    -------
    refs : 
        A NumPy array of shuffled sequences with the same shape as inputs.
    """
    N, A, L = inputs.shape

    # Generate a list of indices for shuffling along the last axis
    shuffled_indices = np.arange(L)

    # Shuffle the indices independently for each sequence
    shuffled_indices = np.tile(shuffled_indices, (N, 1))
    _ = np.apply_along_axis(np.random.shuffle, 1, shuffled_indices)

    # Use numpy advanced indexing to shuffle the sequences
    n_indices = np.arange(N)[:, None, None]
    a_indices = np.arange(A)[None, :, None]
    l_indices = shuffled_indices[:, None, :]

    shuffled_sequences = inputs[n_indices, a_indices, l_indices]

    return shuffled_sequences

def dinuc_shuffle_ref_inputs(
    inputs: NDArray,
    **kwargs
) -> NDArray:
    """Return a NumPy array of dinucleotide shuffled inputs with the same shape as the inputs.

    Inputs are expected to be one-hot encoded sequences with shape (N, A, L)
    where N is the number of sequences, A is the number of nucleotides, and L is the length of the sequences.
    
    Parameters
    ----------
    inputs : NDArray
        The input sequences to be used to generate a set of dinucleotide shuffled reference sequences.
    **kwargs
        Additional keyword arguments to pass to the dinuc_shuffle_seqs function from the seqpro package.

    Returns
    -------
    refs : NDArray
        A NumPy array of dinucleotide shuffled sequences with the same shape as inputs.

    Note
    ----
    This function is a wrapper for the dinuc_shuffle_seqs function from the seqpro package and currently only works with numpy arrays.
    """
    refs = kshuffle(inputs, **kwargs)
    return refs

def gc_ref_inputs(
    inputs: NDArray,
    bg: str = "uniform",
    uniform_dist: List[float] = [0.25, 0.25, 0.25, 0.25]
) -> NDArray:
    """Return a NumPy array with the same GC content and shape as the inputs.
    
    Inputs are expected to be one-hot encoded sequences with shape (N, A, L)
    where N is the number of sequences, A is the number of nucleotides, and L is the length of the sequences.

    Parameters
    ----------
    inputs : NDArray
        The input sequences to be used to generate a set of GC content reference sequences.
    bg : str, optional
        The background distribution to use for the GC content reference sequences. 
        Options are "uniform", "batch", or "seq". Default is "uniform".
        "uniform" will use a uniform distribution across every position of each input 
        The distribution across A is dependent on the uniform_dist parameter.
        "batch" will match the GC content of across the N dimension of the inputs. Meaning that each position
        across sequences will have the same GC content.
        "seq" will match the GC content of each sequence in the inputs. Meaning that each sequence across positions
        will have the same GC content.
    uniform_dist : list, optional
        If bg is "uniform", the uniform distribution to use for the GC content reference sequences. Default is [0.25, 0.25, 0.25, 0.25].
        If using a different alphabet, the distribution should be in the same order and length as the alphabet.
        If bg is "batch" or "seq", this parameter is ignored.

    Returns
    -------
    refs : NDArray
        A NumPy array with the same GC content and shape as the inputs.
    """
    N, A, L = inputs.shape
    if bg is "uniform":
        dists = np.array(uniform_dist)[:, None]
        dists = np.broadcast_to(dists, (A, L))
        refs = np.broadcast_to(dists[None, :, :], (N, A, L))
    elif bg is "batch":
        dists = inputs.sum(0)/N
        dists = dists[None, :]
        refs = np.broadcast_to(dists, (N, A, L))
    elif bg is "seq":
        dists = inputs.sum(2)/L
        refs = np.tile(dists[:, :, np.newaxis], (1, 1, L)) 
    else:
        raise ValueError(f"Background distribution {bg} not in ['uniform', 'batch', 'seq']")
    return refs 

def profile_ref_inputs(
    inputs: NDArray
) -> NDArray:
    """Return a NumPy array of nucleotide profile reference sequences with the same shape as the inputs.

    Inputs are expected to be one-hot encoded sequences with shape (N, A, L)
    where N is the number of sequences, A is the number of nucleotides, and L is the length of the sequences.
    For all sequences, we calculate the mean nucleotide profile and use that to generate a set of reference sequences.
    
    Parameters
    ----------
    inputs : NDArray
        The input sequences to be used to generate a set of nucleotide profile reference sequences.
    
    Returns
    -------
    refs : NDArray
        A NumPy array of nucleotide profile reference sequences with the same shape as the inputs.
    """
    N, A, L = inputs.shape
    seq_model = np.mean(np.squeeze(inputs), axis=0)
    seq_model /= np.sum(seq_model, axis=0, keepdims=True)

    refs = np.zeros((N, A, L))
    for n in range(N):

        # generate uniform random number for each nucleotide in sequence
        Z = np.random.uniform(0, 1, L)

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
    inputs: Union[NDArray, Tuple[NDArray]],
    method: Union[str, Callable],
    **kwargs
) -> Union[NDArray, Tuple[NDArray]]:
    """
    Returns a set of reference sequences with the same shape as the inputs.
    If the method is a string, the function will use one of the built-in reference methods.
    If the method is a callable function, the function will use the custom reference method.
    If inputs are a tuple of tensors, the function will return a tuple of tensors with the same shape as inputs.

    Parameters
    ----------
    inputs : NDArray
        The input sequences to be used to generate a set of reference sequences.
    method : str or callable
        The method to use to generate the reference sequences. 
        Options are "zero", "random", "shuffle", "dinuc_shuffle", "gc", "profile", or a callable function.
    kwargs : dict
        Additional arguments to pass to the reference method.
    
    Returns
    -------
    reference : NDArray
        A NumPy array of reference sequences with the same shape as the inputs. If inputs are a tuple
        of arrays, the function will return a tuple of arrays with the same shape as inputs.

    Note
    ----
    If you want to use a custom reference method, you can pass a callable function as the method argument. 
    The function must take an NDArray as input and return an NDArray with the same shape as the input.
    If the inputs are a tuple of arrays, the function must return a tuple of arrays with the same shape as the inputs.
    """
    if isinstance(method, str):
        if method not in REFERENCE_REGISTRY:
            raise ValueError(f"Reference method {method} not in {list(REFERENCE_REGISTRY.keys())}")
        if isinstance(inputs, tuple):
            reference = tuple([REFERENCE_REGISTRY[method](inputs[i], **kwargs) for i in range(len(inputs))])
        else:
            reference = REFERENCE_REGISTRY[method](inputs, **kwargs)
    elif callable(method):
        if isinstance(inputs, tuple):
            reference = tuple([method(inputs[i], **kwargs) for i in range(len(inputs))])
        else:
            reference = method(inputs, **kwargs)
    else:
        raise ValueError(f"Reference method {method} not in {list(REFERENCE_REGISTRY.keys())}")
    return reference
