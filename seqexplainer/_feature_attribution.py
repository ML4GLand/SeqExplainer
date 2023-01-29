import torch
import numpy as np
from typing import Union, Callable
from tqdm.auto import tqdm
from captum.attr import InputXGradient, DeepLift, GradientShap, DeepLiftShap
from ._references import _get_reference
from ._seq import perturb_seq_torch
from ._utils import _model_to_device
from eugene import settings as settings


# Reference vs output difference methods
def delta(y, reference):
    return (y - reference).sum(axis=-1)

def l1(y, reference):
    return (y - reference).abs().sum(axis=-1)

def l2(y, reference):
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
    batch_size=128, 
    diff_type="delta", 
    device="cpu"
):

    # Get the number of sequences, choices, and sequence length
    n_seqs, n_choices, seq_len = inputs.shape
    n = seq_len * (n_choices - 1)
    X_idxs = inputs.argmax(axis=1)

    # If target not provided aggregate over all outputs
    target = np.arange(model.output_dim) if target is None else target
    
    # Move the model to eval mode
    model = model.eval() 

    # Get the reference output
    reference = model(inputs)[:, target].unsqueeze(1)
    batch_starts = np.arange(0, n, batch_size)

    # Get the change in output for each perturbation
    isms = []
    for i in range(n_seqs):
        X = perturb_seq_torch(inputs[i])
        y = []
        for start in batch_starts:
            X_ = X[start : start + batch_size]
            y_ = model(X_)[:, target].unsqueeze(1)
            y.append(y_)
            del X_
        y = torch.cat(y)
        ism = DIFF_REGISTRY[diff_type](y, reference[i])
        isms.append(ism)
        
        if device[:4] == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # Clean up the output to be (N, A, L)
    isms = torch.stack(isms).to(device)
    isms = isms.reshape(n_seqs, seq_len, n_choices - 1)
    j_idxs = torch.arange(n_seqs * seq_len)
    X_ism = torch.zeros(n_seqs * seq_len, n_choices, device=device)
    for i in range(1, n_choices):
        i_idxs = (X_idxs.flatten() + i) % n_choices
        X_ism[j_idxs, i_idxs] = isms[:, :, i - 1].flatten()

    X_ism = X_ism.reshape(n_seqs, seq_len, n_choices).permute(0, 2, 1)
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
    target: int = 0,
    device: str = "cpu",
    **kwargs
):
    # Set device
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device

    # Put model on device
    model = _model_to_device(model, device)

    # Put inputs on device
    if isinstance(inputs, tuple):
        inputs = tuple([i.requires_grad_().to(device) for i in inputs])
    else:
        inputs = inputs.requires_grad_().to(device)

    # Check kwargs for reference
    if "reference_type" in kwargs:
        ref_type = kwargs.pop("reference_type")
        kwargs["baselines"] = _get_reference(inputs, ref_type, device)

    # Get attributions
    attr = ATTRIBUTIONS_REGISTRY[method](
        model=model,
        inputs=inputs,
        method=method,
        target=target,
        device=device,
        **kwargs
    )

    # Return attributions
    return attr

def attribute_on_batch(
    model,
    inputs: torch.Tensor,
    method: Union[str, Callable],
    target: int = 0,
    device: str = "cpu",
    batch_size: int = 128,
    **kwargs
):
    # Disable cudnn for faster computations
    torch.backends.cudnn.enabled = False

    # Set device
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device

    # Put model on device
    model = _model_to_device(model, device)

    # Create an empty list to hold attributions
    attrs = []
    starts = np.arange(0, inputs.shape[0], batch_size)

    # Loop through batches and compute attributions
    for i, start in tqdm(
        enumerate(starts),
        total=len(starts),
        desc=f"Computing feature attributions on batches of size {batch_size}",
    ):
        inputs_ = inputs[start : start + batch_size]
        curr_attrs = attribute(
            model,
            inputs_,
            target=target,
            method=method,
            device=device,
            **kwargs
        )
        attrs.append(curr_attrs)

    # Concatenate the attributions
    attrs = torch.cat(attrs)

    # Return attributions
    return attrs