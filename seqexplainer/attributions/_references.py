from typing import Callable, List, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import seqpro as sp
from bpnetlite.attributions import create_references


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

def gc_ref_inputs(
    inputs: NDArray,
    bg: str = "uniform",
    uniform_dist: List[float] = [0.3, 0.2, 0.3, 0.2]
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

def random_ref_inputs(
    inputs: NDArray,
    n_per_input: int = 1
):
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
    if n_per_input == 1:
        ref_tokens = np.random.randint(4, size=(N, L))
        refs = np.eye(A)[ref_tokens].transpose(0, 2, 1)
    else:
        ref_tokens = np.random.randint(4, size=(N, n_per_input, L))
        refs = np.eye(A)[ref_tokens].transpose(0, 1, 3, 2)
    return refs

def profile_ref_inputs(
    inputs: NDArray,
    n_per_input: int = 1
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

    if n_per_input == 1:
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
    else:
        refs = np.zeros((N, n_per_input, A, L))
        for n in range(N):
            for i in range(n_per_input):

                # generate uniform random number for each nucleotide in sequence
                Z = np.random.uniform(0, 1, L)

                # calculate cumulative sum of the probabilities
                cum_prob = seq_model.cumsum(axis=0)

                # find bin that matches random number for each position
                for l in range(L):
                    index = [j for j in range(4) if Z[l] < cum_prob[j, l]][0]
                    refs[n, i, index, l] = 1

        return refs

def k_shuffle_ref_inputs(
    inputs: NDArray,
    k: int = 2,
    n_per_input: int = 1
):
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
    if n_per_input == 1:
        input_seqs = sp.decode_ohe(inputs.astype(np.uint8), alphabet=sp.alphabets.RNA, ohe_axis=1)
        ref_seqs = sp.k_shuffle(input_seqs, k=k, length_axis=1)
        refs = sp.ohe(ref_seqs, alphabet=sp.alphabets.RNA).transpose(0, 2, 1)
    else:
        refs = np.zeros((inputs.shape[0], n_per_input, inputs.shape[1], inputs.shape[2]))
        for i in range(n_per_input):
            input_seqs = sp.decode_ohe(inputs.astype(np.uint8), alphabet=sp.alphabets.RNA, ohe_axis=1)
            ref_seqs = sp.k_shuffle(input_seqs, k=k, length_axis=1)
            refs[:, i, :, :] = sp.ohe(ref_seqs, alphabet=sp.alphabets.RNA).transpose(0, 2, 1)
    return refs

REFERENCE_REGISTRY = {
    "zero": zero_ref_inputs,
    "gc": gc_ref_inputs,
    "random": random_ref_inputs,
    "profile": profile_ref_inputs,
    "k_shuffle": k_shuffle_ref_inputs,
    "bpnetlite": create_references,
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
