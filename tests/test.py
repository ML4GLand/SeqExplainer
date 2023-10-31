import numpy as np
from numpy.typing import NDArray


def check_nucleotide_balance(
    seq_array: NDArray, 
    shuffled_array: NDArray
) -> bool:
    """Check that the nucleotide balance is preserved after shuffling.
    
    Parameters
    ----------
    seq_array : NDArray
        The original input sequences with shape (N, A, L).
    shuffled_array : NDArray
        The shuffled sequences with shape (N, A, L).
    
    Returns
    -------
    bool
        True if the nucleotide balance is preserved for each sequence, False otherwise.
    """
    # Compute the sum of each nucleotide along the length axis for the original sequences
    seq_sum = np.sum(seq_array, axis=2)
    
    # Compute the sum of each nucleotide along the length axis for the shuffled sequences
    shuffled_sum = np.sum(shuffled_array, axis=2)
    
    # Check that the sum of each nucleotide is the same before and after shuffling for each sequence
    return np.array_equal(seq_sum, shuffled_sum)