import numpy as np
import torch
from seqexplainer.preprocess._preprocess import ohe_seqs
from seqexplainer._utils import _model_to_device
from tqdm.auto import tqdm

def deepstarr_motif_distance_cooperativity_gia(
    model,
    b_seqs,
    A_seqs,
    B_seqs,
    AB_seqs,
    motif_b_distances,
    batch_size=128,
    device="cpu"
):
    """
    Calculates a cooperativity score between pairs of motifs using the methodology outlined in DeepSTARR.

    We recommend that you use `embed_deepstarr_distance_cooperativity` to generate the sequences for this function.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use to calculate the cooperativity scores.
    b_seqs : np.array
        The background sequences to use to calculate the cooperativity scores.
    A_seqs : np.array
        The sequences with only MotifA in the center.
    B_seqs : np.array
        The sequences with only MotifB at a range of distances from the center.
    AB_seqs : np.array
        The sequences with both MotifA and MotifB at a range of distances from the center.
    motif_b_distances : np.array
        The distances between MotifA and MotifB for each sequence.
    batch_size : int, optional
        The batch size to use when calculating the cooperativity scores. Default is 128.
    device : str, optional
        The device to put the model on prior to prediction. Default is "cpu".

    Returns
    -------
    d_fold_changes: dict
        A dictionary with the fold changes for each distance between MotifA and MotifB.
    """

    # Put the model on the device
    _model_to_device(model, device)

    # Get b scores (background)
    print("Getting background scores")
    b_ohe = ohe_seqs(b_seqs, verbose=False)
    b_tensor = torch.tensor(b_ohe, dtype=torch.float32)
    b_scores = []
    for j in range(0, len(b_tensor), batch_size):
        curr_inputs = b_tensor[j:j+batch_size].to(device)
        b_scores.append(model(curr_inputs).detach().cpu().numpy())
    b_scores = np.concatenate(b_scores, axis=0)

    # Get motif A scores
    print("Getting motif A scores")
    A_ohe = ohe_seqs(A_seqs, verbose=False)
    A_tensor = torch.tensor(A_ohe, dtype=torch.float32)
    A_scores = []
    for j in range(0, len(A_tensor), batch_size):
        curr_inputs = A_tensor[j:j+batch_size].to(device)
        A_scores.append(model(curr_inputs).detach().cpu().numpy())
    A_scores = np.concatenate(A_scores, axis=0)

    d_fold_changes = {}
    for i, d in tqdm(enumerate(motif_b_distances), total=len(motif_b_distances), desc="Calculating fold changes for each provided distance between MotifA and MotifB"):
        curr_B_seqs = B_seqs[i]
        curr_B_ohe = ohe_seqs(curr_B_seqs, verbose=False)
        curr_B_tensor = torch.tensor(curr_B_ohe, dtype=torch.float32)
        curr_B_scores = []
        for j in range(0, len(curr_B_tensor), batch_size):
            curr_inputs = curr_B_tensor[j:j+batch_size].to(device)
            curr_B_scores.append(model(curr_inputs).detach().cpu().numpy())
        curr_B_scores = np.concatenate(curr_B_scores)
        
        curr_AB_seqs = AB_seqs[i]
        curr_AB_ohe = ohe_seqs(curr_AB_seqs, verbose=False)
        curr_AB_tensor = torch.tensor(curr_AB_ohe, dtype=torch.float32)
        curr_AB_scores = []
        for j in range(0, len(curr_AB_tensor), batch_size):
            curr_inputs = curr_AB_tensor[j:j+batch_size].to(device)
            curr_AB_scores.append(model(curr_inputs).detach().cpu().numpy())
        curr_AB_scores = np.concatenate(curr_AB_scores)
        
        # Normalize
        b_exp = np.exp2(b_scores)
        A_exp = np.exp2(A_scores)
        curr_B_exp = np.exp2(curr_B_scores)
        curr_AB_exp = np.exp2(curr_AB_scores)

        # Get denom
        denom = A_exp + curr_B_exp - b_exp

        # Get fold change
        fold_change = curr_AB_exp / denom

        # Put the fold changes in a dict
        d_fold_changes[d] = fold_change

    return d_fold_changes
