import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .gia._perturb import perturb_seq_torch
from ._utils import _model_to_device

# Reference vs output difference methods
def delta(y, reference):
    """Difference between output and reference"""
    return (y - reference).sum(axis=-1)

def l1(y, reference):
    """L1 norm between output and reference"""
    return (y - reference).abs().sum(axis=-1)

def l2(y, reference):
    """L2 norm between output and reference"""
    return torch.sqrt(torch.square(y - reference).sum(axis=-1))

DIFF_REGISTRY = {
    "delta": delta,
    "l1": l1,
    "l2": l2,
}

# In silico mutagenesis methods
def _naive_ism(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target=None, 
    batch_size=32, 
    diff_type="delta", 
    device="cpu"
):
    """Naive in silico mutagenesis

    Perturb each position in the input sequence and calculate the difference in output
    """

    # Get the number of sequences, choices, and sequence length
    N, A, L = inputs.shape
    n = L * (A - 1)
    X_idxs = inputs.argmax(axis=1)

    # If target not provided aggregate over all outputs
    target = np.arange(model.output_dim) if target is None else target
    
    # Get the reference output
    _model_to_device(model, device)
    reference = model.predict(inputs, batch_size=batch_size, verbose=False)[:, target].unsqueeze(dim=1).cpu()

    # Get the batch starts
    batch_starts = np.arange(0, n, batch_size)

    # Get the change in output for each perturbation
    isms = []
    for i in range(N):
        X = perturb_seq_torch(inputs[i])
        y = []
        for start in batch_starts:
            X_ = X[start : start + batch_size]
            with torch.no_grad():
                y_ = model.predict(X_, batch_size=batch_size, verbose=False)[:, target].unsqueeze(dim=1).cpu()
            y.append(y_)
            del X_, y_
            if device[:4] == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        y = torch.cat(y)
        ism = DIFF_REGISTRY[diff_type](y, reference[i]).cpu()
        isms.append(ism)
    
    # Clean up the output to be (N, A, L)
    isms = torch.stack(isms).cpu()
    isms = isms.reshape(N, L, A - 1)
    j_idxs = torch.arange(N * L)
    X_ism = torch.zeros(N * L, A)
    
    for i in range(1, A):
        i_idxs = (X_idxs.flatten() + i) % A 
        X_ism[j_idxs, i_idxs] = isms[:, :, i - 1].flatten()

    X_ism = X_ism.reshape(N, L, A).permute(0, 2, 1)
    return X_ism
