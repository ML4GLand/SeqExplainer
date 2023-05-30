import gc
from typing import Callable, Union


import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift, DeepLiftShap, GradientShap, InputXGradient
from tqdm.auto import tqdm

from .._perturb import perturb_seq_torch
from .._references import get_reference
from .._utils import _model_to_device, report_gpu

def _get_oned_contribs(
    one_hot,
    hypothetical_contribs,
):
    contr = one_hot * hypothetical_contribs
    oned_contr = contr.sum(axis=1)
    return oned_contr

def _gradient_correction(
    hypothetical_contribs,
    one_hot,
    reference,
    diff_type="delta",
):
    pass

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
    model, 
    inputs, 
    target=None, 
    batch_size=32, 
    diff_type="delta", 
    device="cpu"
):
    """Naive in silico mutagenesis

    Perturb each position in the input sequence and calculate the difference in output
    """

    # Get the number of sequences, choices, and sequence length
    print("inputs:", inputs.shape, inputs.device)
    N, A, L = inputs.shape
    n = L * (A - 1)
    X_idxs = inputs.argmax(axis=1)

    # If target not provided aggregate over all outputs
    target = np.arange(model.output_dim) if target is None else target
    
    # Get the reference output
    model.eval()
    reference = model(inputs)[:, target].unsqueeze(1).cpu()
    inputs = inputs.cpu()
    print("reference:", reference.shape, reference.device)
    print("inputs:", inputs.shape, inputs.device) 

    # Get the batch starts
    batch_starts = np.arange(0, n, batch_size)
    print("batch_starts:", batch_starts[:5])
    report_gpu()

    # Get the change in output for each perturbation
    isms = []
    for i in range(N):
        print("input:", inputs[i].shape, inputs[i].device)
        X = perturb_seq_torch(inputs[i]).cpu()
        print("X:", X.shape, X.device)
        y = []
        for start in batch_starts:
            model.to(device)
            print("model:", model.device)
            X_ = X[start : start + batch_size]
            X_ = X_.to(device)
            with torch.no_grad():
                y_ = model(X_)[:, target].unsqueeze(1).cpu()
                X_ = X_.detach().cpu()
                #print("y_:", y_.shape, y_.device)
            y.append(y_)
            del X_, y_
            if device[:4] == 'cuda':
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                report_gpu()
            #print("model:", model.device)
        y = torch.cat(y)
        print("y:", y.shape, y.device)
        ism = DIFF_REGISTRY[diff_type](y, reference[i]).cpu()
        print("ism:", ism.shape, ism.device)
        isms.append(ism)
        print("inputs:", inputs.shape, inputs.device)

        
    
    # Clean up the output to be (N, A, L)
    isms = torch.stack(isms).cpu()
    print("isms:", isms.shape, isms.device)

    isms = isms.reshape(N, L, A - 1)
    j_idxs = torch.arange(N * L)
    X_ism = torch.zeros(N * L, A)
    
    for i in range(1, A):
        i_idxs = (X_idxs.flatten() + i) % A 
        X_ism[j_idxs, i_idxs] = isms[:, :, i - 1].flatten()

    X_ism = X_ism.reshape(N, L, A).permute(0, 2, 1)
    return X_ism

ISM_REGISTRY = {
    "NaiveISM": _naive_ism,
}

def _ism_attributions(
    model: torch.nn.Module, 
    inputs: Union[tuple, torch.Tensor],
    method: Union[str, Callable],
    target: int = None,
    device: str = "cpu",
    **kwargs
):
    attrs = ISM_REGISTRY[method](model=model, inputs=inputs, target=target, device=device, **kwargs)
    return attrs

# Captum methods
CAPTUM_REGISTRY = {
    "InputXGradient": InputXGradient,
    "DeepLift": DeepLift,
    "DeepLiftShap": DeepLiftShap,
    "GradientShap": GradientShap,
}

def _captum_attributions(
    model: torch.nn.Module,
    inputs: tuple,
    method: str,
    target: int = 0,
    device: str = "cpu",
    **kwargs
):
    """
    """
    if isinstance(inputs, np.ndarray):
        inputs = torch.tensor(inputs)
    attributor = CAPTUM_REGISTRY[method](model)
    attrs = attributor.attribute(inputs=inputs, target=target, **kwargs)
    return attrs

# Attribution methods -- combination of above
ATTRIBUTIONS_REGISTRY = {
    "NaiveISM": _ism_attributions,
    "InputXGradient": _captum_attributions,
    "DeepLift": _captum_attributions,
    "GradientShap": _captum_attributions,
    "DeepLiftShap": _captum_attributions,
}

def attribute(
    model,
    inputs: torch.Tensor,
    method: Union[str, Callable],
    reference_type: str = None,
    target: int = 0,
    batch_size: int = 128,
    device: str = "cpu",
    verbose: bool = True,
):
    # Disable cudnn for faster computations 
    torch.backends.cudnn.enabled = False
    
    # Put model on device
    model = _model_to_device(model, device)

    # Create an empty list to hold attributions
    attrs = []
    starts = np.arange(0, inputs.shape[0], batch_size)

    # Loop through batches and compute attributions
    for _, start in tqdm(
        enumerate(starts),
        total=len(starts),
        desc=f"Computing attributions on batches of size {batch_size}",
        disable=not verbose,
    ):
        # Grab the current batch
        inputs_ = inputs[start : start + batch_size]
        inputs_ = inputs_.requires_grad_(True).to(device)
        #print("inputs_:", inputs_.shape, inputs_.device)
        
        # Add reference if needed
        kwargs = {}
        if reference_type is not None:
            refs = get_reference(inputs_, reference_type)
            refs = torch.tensor(refs, dtype=torch.float32).requires_grad_(True).to(device)
            #print("refs:", refs.shape, refs.device)
            kwargs["baselines"] = refs

        # Get attributions and append
        curr_attrs = ATTRIBUTIONS_REGISTRY[method](
            model=model,
            inputs=inputs_,
            method=method,
            target=target,
            device=device,
            **kwargs
        ).detach().cpu()
        attrs.append(curr_attrs)

    # Concatenate the attributions
    attrs = torch.cat(attrs).numpy()

    # Return attributions
    return attrs

