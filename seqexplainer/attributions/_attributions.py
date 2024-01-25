import gc
from typing import Callable, Union
import numpy as np
import pandas as pd
import torch
from captum.attr import InputXGradient, DeepLift, IntegratedGradients, GradientShap
from bpnetlite.attributions import DeepLiftShap
from tqdm.auto import tqdm
from ._references import get_reference
from .._utils import _model_to_device
from .._ism import _naive_ism


def gradient_correction(
    grad: Union[np.ndarray, torch.Tensor]
):
    """Gradient correction"""
    if isinstance(grad, torch.Tensor):
        grad -= grad.mean(dim=1, keepdim=True)
    else:
        grad -= np.mean(grad, axis=1, keepdims=True)
    return grad

# In silico mutagenesis methods
ISM_REGISTRY = {
    "NaiveISM": _naive_ism,
}

def _ism_attributions(
    model: torch.nn.Module, 
    inputs: Union[tuple, torch.Tensor],
    method: Union[str, Callable],
    target: int = None,
    attribution_func: Callable = None,
    device: str = "cpu",
    batch_size: int = 128,
    verbose: bool = False,
    **kwargs
):
    attrs = ISM_REGISTRY[method](model=model, inputs=inputs, target=target, device=device, batch_size=batch_size, **kwargs)
    if attribution_func is not None:
        attrs = attribution_func(attrs)
    return attrs

# Captum methods
CAPTUM_REGISTRY = {
    "InputXGradient": InputXGradient,
    "DeepLift": DeepLift,
    "IntegratedGradients": IntegratedGradients,
    "GradientShap": GradientShap,
}

def _captum_attributions(
    model: torch.nn.Module,
    inputs: tuple,
    method: str,
    target: int = 0,
    attribution_func: Callable = None,
    device: str = "cpu",
    batch_size: int = 128,
    verbose: bool = False,
    **kwargs
):
    """
    """
    if isinstance(inputs, np.ndarray):
        inputs = torch.tensor(inputs)
    attributor = CAPTUM_REGISTRY[method](model)
    attrs = attributor.attribute(inputs=inputs, target=target, **kwargs)
    if attribution_func is not None:
        attrs = attribution_func(attrs)
    return attrs

def _deepliftshap_attributions(
    model: torch.nn.Module,
    inputs: tuple,
    baselines: tuple,
    target: int = None,
    attribution_func: Callable = None,
    warning_threshold: float = 0.001,
    verbose: bool = False,
    device: str = "cpu",
    batch_size: int = 128,
    **kwargs
):
    """
    """
    if isinstance(inputs, np.ndarray):
        inputs = torch.tensor(inputs)
    attributor = DeepLiftShap(
        model=model,
        attribution_func=attribution_func,
        warning_threshold=warning_threshold,
        verbose=verbose,
    )
    attrs = attributor.attribute(
        inputs=inputs, 
        baselines=baselines,
        **kwargs
    )
    return attrs

# Attribution methods -- combination of above
ATTRIBUTIONS_REGISTRY = {
    "NaiveISM": _ism_attributions,
    "InputXGradient": _captum_attributions,
    "DeepLift": _captum_attributions,
    "IntegratedGradients": _captum_attributions,
    "GradientShap": _captum_attributions,
    "DeepLiftShap": _deepliftshap_attributions,
}

def attribute(
    model,
    inputs: torch.Tensor,
    method: Union[str, Callable], 
    references: Union[str, np.ndarray] = None,
    target: int = 0,
    attribution_func: Callable = None,
    hypothetical : bool = False,
    batch_size: int = 128,
    device: str = "cpu",
    verbose: bool = True,
    **kwargs
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
        
        # Add reference if needed
        if references is not None:
            if isinstance(references, str):
                refs = get_reference(inputs_, references)
                refs = torch.tensor(refs, dtype=torch.float32).requires_grad_(True).to(device)
            else:
                refs = torch.tensor(references[start : start + batch_size], dtype=torch.float32).requires_grad_(True).to(device)
            kwargs["baselines"] = refs
        else:
            assert method in ["NaiveISM", "InputXGradient"], f"Must provide references for {method}"
        
        # Convert to tensor and put on device
        inputs_ = torch.tensor(inputs_, dtype=torch.float32).requires_grad_(True).to(device)

        # Get attributions and append
        curr_attrs = ATTRIBUTIONS_REGISTRY[method](
            model=model,
            inputs=inputs_,
            method=method,
            target=target,
            attribution_func=attribution_func,
            device=device,
            **kwargs
        ).detach().cpu()
        attrs.append(curr_attrs)

    # Concatenate the attributions
    attrs = torch.cat(attrs).numpy()

    # if not hypothetical, multiply by the input
    if not hypothetical:
        attrs = attrs * inputs

    # Return attributions
    return attrs
