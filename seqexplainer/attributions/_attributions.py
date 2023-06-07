import gc
from typing import Callable, Union
import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift, DeepLiftShap, GradientShap, InputXGradient
from tqdm.auto import tqdm
from ._references import get_reference
from .._utils import _model_to_device
from .._ism import _naive_ism

def _get_oned_contribs(
    one_hot,
    hypothetical_contribs,
):
    contr = one_hot * hypothetical_contribs
    oned_contr = contr.sum(axis=1)
    return oned_contr

def _gradient_correction(
    grad: np.ndarray,
):
    grad -= np.mean(grad, axis=1, keepdims=True)


ISM_REGISTRY = {
    "NaiveISM": _naive_ism,
}

def _ism_attributions(
    model: torch.nn.Module, 
    inputs: Union[tuple, torch.Tensor],
    method: Union[str, Callable],
    target: int = None,
    device: str = "cpu",
    batch_size: int = 128,
    **kwargs
):
    attrs = ISM_REGISTRY[method](model=model, inputs=inputs, target=target, device=device, batch_size=batch_size, **kwargs)
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
    batch_size: int = 128,
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
        
        # Add reference if needed
        kwargs = {}
        if reference_type is not None:
            refs = get_reference(inputs_, reference_type)
            refs = torch.tensor(refs, dtype=torch.float32).requires_grad_(True).to(device)
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
