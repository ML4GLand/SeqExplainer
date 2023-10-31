import torch
import numpy as np
from .._utils import _k_largest_index_argsort
from .._ism import _naive_ism
from .._utils import _model_to_device


def best_k_muts(
    model: torch.nn.Module, 
    X: np.ndarray, 
    k: int = 1, 
    device: str = "cpu"
) -> np.ndarray:
    """
    Find and return the k highest scoring sequence from referenece sequence X.

    Using ISM, this function calculates all the scores of all possible mutations
    of the reference sequence using a trained model. It then returns the k highest
    scoring sequences, along with delta scores from the reference sequence and the indeces
    along the lengths of the sequences where the mutations can be found

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    X: np.ndarray
        The one-hot encoded sequence to calculate find mutations for.
    k: int, optional
        The number of mutated seqences to return
    device: str, optional
        Whether to use a 'cpu' or 'cuda'.

    Returns
    -------
    mut_X: np.ndarray
        The k highest scoring one-hot-encoded sequences
    maxs: np.ndarray
        The k highest delta scores corresponding to the k highest scoring sequences
    locs: np.ndarray
        The indeces along the length of the sequences where the mutations can be found
    """
    
    # Send model to device and put in eval mode 
    _model_to_device(model, device)

    # Need to expand dims if X is 2D
    X = np.expand_dims(X, axis=0) if X.ndim == 2 else X
    #X = X.transpose(0, 2, 1) if X.shape[2] == 4 else X
    
    # Perform ISM on the input
    X = torch.tensor(X, dtype=torch.float32, device=device)
    X_ism = _naive_ism(model, X, device=device, batch_size=1)
    X = X.detach().cpu().numpy()
    X_ism = X_ism.squeeze(axis=0).detach().cpu().numpy()
    
    # Find the k highest scoring sequences
    inds = _k_largest_index_argsort(X_ism, k)
    locs = inds[:, 1]
    maxs = np.max(X_ism, axis=0)[locs]
    mut_Xs = np.zeros((k, X.shape[1], X.shape[2]))
    for i in range(k):
        mut_X = X.copy().squeeze(axis=0)
        mut_X[:, inds[i][1]] = np.zeros(mut_X.shape[0])
        mut_X[:, inds[i][1]][inds[i][0]] = 1
        mut_Xs[i] = mut_X
    return mut_Xs, maxs, locs

def best_mut_seqs(
    model: torch.nn.Module,
    X: np.ndarray, 
    batch_size: int = 128, 
    device: str = "cpu"
) -> np.ndarray:
    """Find and return the highest scoring sequence for each sequence from a set reference sequences X. 
    
    X should contain one-hot-encoded sequences
    and should be of shape (n, 4, l). n is the number of sequences, 4 is the number of
    nucleotides, and l is the length of the sequence.

    Using ISM, this function calculates all the scores of all possible mutations
    of each reference sequence using a trained model. It then returns the highest
    scoring sequence, along with delta scores from the reference sequence and the indeces
    along the lengths of the sequences where the mutations can be found.

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    X: np.ndarray
        The one-hot encoded sequences to calculate find mutations for.
    batch_size: int, optional
        The number of sequences to score at once.
    device: str, optional
        Whether to use a 'cpu' or 'cuda'.

    Returns
    -------
    mut_X: np.ndarray
        The highest scoring one-hot-encoded sequences
    maxs: np.ndarray
        The highest delta scores corresponding to the highest scoring sequences
    locs: np.ndarray
        The indeces along the length of the sequences where the mutations can be found

    """
    model.eval().to(device)
    #X = X.transpose(0, 2, 1) if X.shape[2] == 4 else X
    X = torch.tensor(X, dtype=torch.float32, device=device)
    X_ism = _naive_ism(model, X, device=device, batch_size=batch_size)
    maxs, inds, mut_X = [], [], X.copy()
    for i in range(len(mut_X)):
        maxs.append(np.max(X_ism[i]))
        ind = np.unravel_index(X_ism[i].argmax(), X_ism[i].shape)
        inds.append(ind[1])
        mut_X[i][:, ind[1]] = np.zeros(mut_X.shape[1])
        mut_X[i][:, ind[1]][ind[0]] = 1
    return mut_X, np.array(maxs), np.array(inds)

def evolution(
    model: torch.nn.Module,
    X: np.ndarray,
    rounds: int = 10,
    k: int = 10,
    force_different: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    """Perform rounds rounds of in-silico evolution on a single sequence X.

    Using ISM, this function calculates all the scores of all possible mutations
    on a starting sequence X. It then mutates the sequence and repeats the process
    for rounds rounds. In the end, it returns new "evolved" sequence after rounds mutations
    and the delta scores from the reference sequence and the indeces along the lengths of the sequences
    with which the mutations occured

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    X: np.ndarray
        The one-hot encoded reference sequence to start the evolution with
    rounds: int, optional
        The number of rounds of evolution to perform
    force_different: bool, optional
        Whether to force the mutations to occur at different locations in the reference sequence
    k: int, optional
        The number of mutated sequences to consider at each round. This is in case
        the same position scores highest multiple times.
    """
    model.eval().to(device)
    curr_X = X.copy()
    mutated_positions, mutated_scores = [], []
    for r in range(rounds):
        mut_X, score, positions = best_k_muts(model, curr_X, k=k, device=device)
        if force_different:
            for i, p in enumerate(positions):
                if p not in mutated_positions:
                    curr_X = mut_X[i]
                    mutated_positions.append(p)
                    mutated_scores.append(score[i])
                    break
        else:
            curr_X = mut_X[0]
            mutated_positions.append(positions[0])
            mutated_scores.append(score[0])
    return curr_X, mutated_scores, mutated_positions
