import torch
import numpy as np
from typing import Union, Callable
from tqdm.auto import tqdm
from captum.attr import InputXGradient, DeepLift, GradientShap, DeepLiftShap
from ._references import _get_reference
from ._interpret_utils import _model_to_device, _naive_ism
from eugene import settings as settings

# In silico mutagenesis methods
ISM_REGISTRY = {
    "NaiveISM": _naive_ism,
}

def _ism_attributions(
    model: torch.nn.Module, 
    inputs: tuple, 
    method: Union[str, Callable],
    target: int = 0,
    device: str = "cpu", 
    **kwargs
):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    if model.strand != "ss":
        raise ValueError("ISM currrently only works for single strand models, but we are working on this!")
    attrs = ISM_REGISTRY[method](
        model=model,
        X_0=inputs,  # Rename to inputs eventually
        device=device,
        **kwargs
    )
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
    **kwargs
):
    """
    """
    if isinstance(inputs, np.ndarray):
        inputs = torch.tensor(inputs)
    attributor = CAPTUM_REGISTRY[method](model)
    attrs = attributor.attribute(inputs=inputs, target=target, **kwargs)
    return attrs

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

    #print(inputs[0].requires_grad)#, kwargs["baselines"][0].requires_grad)
    #print(inputs.shape, kwargs["baselines"].shape, kwargs["additional_forward_args"].shape)
    #print(kwargs.keys(), method, target, model)
    # Get attributions
    attr = ATTRIBUTIONS_REGISTRY[method](
        model=model,
        inputs=inputs,
        method=method,
        target=target,
        **kwargs
    )

    # Return attributions
    return attr


def feature_attribution_sdata(
    model: torch.nn.Module,  # need to enforce this is a SequenceModel
    sdata,
    method: str = "DeepLiftShap",
    target: int = 0,
    aggr: str = None,
    multiply_by_inputs: bool = True,
    batch_size: int = None,
    num_workers: int = None,
    device: str = "cpu",
    transform_kwargs: dict = {},
    prefix: str = "",
    suffix: str = "",
    copy: bool = False,
    **kwargs
):
    """
    Wrapper function to compute feature attribution scores for a SequenceModel using the 
    set of sequences defined in a SeqData object.
    
    Allows for computing scores using different methods and different reference types on any task.
    
    Parameters
    ----------
    model : torch.nn.Module
       PyTorch model to use for computing feature attribution scores.
        Can be a EUGENe trained model or one you trained with PyTorch or PL.
    sdata : SeqData
        SeqData object containing the sequences to compute feature attribution scores on.
    method: str
        Type of saliency to use for computing feature attribution scores.
        Can be one of the following:
        - "gradxinput" (gradients x inputs)
        - "intgrad" (integrated gradients)
        - "intgradxinput" (integrated gradients x inputs)
        - "smoothgrad" (smooth gradients)
        - "smoothgradxinput" (smooth gradients x inputs)
        - "deeplift" (DeepLIFT)
        - "gradientshap" (GradientSHAP)
    target: int
        Index of the target class to compute scores for if there are multiple outputs. If there
        is a single output, this should be None
    batch_size: int
        Batch size to use for computing feature attribution scores. If not specified, will use the
        default batch size of the model
    num_workers: int
        Number of workers to use for computing feature attribution scores. If not specified, will use
        the default number of workers of the model
    device: str
        Device to use for computing feature attribution scores.
        EUGENe will always use a gpu if available
    transform_kwargs: dict
        Dictionary of keyword arguments to pass to the transform method of the model
    prefix: str
        Prefix to add to the feature attribution scores
    suffix: str
        Suffix to add to the feature attribution scores
    copy: bool
        Whether to copy the SeqData object before computing feature attribution scores. By default
        this is False
    **kwargs
        Additional arguments to pass to the saliency method. For example, you can pass the number of
        samples to use for SmoothGrad and Integrated Gradients
    Returns
    -------
    SeqData
        SeqData object containing the feature attribution scores
    """

    # Disable cudnn for faster computations
    torch.backends.cudnn.enabled = False
    
    # Copy the SeqData object if necessary
    sdata = sdata.copy() if copy else sdata

    # Configure the device, batch size, and number of workers
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers

    # Make a dataloader from the sdata
    sdataset = sdata.to_dataset(target_keys=None, transform_kwargs=transform_kwargs)
    sdataloader = sdataset.to_dataloader(batch_size=batch_size, shuffle=False)
    
    # Create an empty array to hold attributions
    dataset_len = len(sdataloader.dataset)
    example_shape = sdataloader.dataset[0][1].numpy().shape
    all_forward_explanations = np.zeros((dataset_len, *example_shape))
    if model.strand != "ss":
        all_reverse_explanations = np.zeros((dataset_len, *example_shape))

    # Loop through batches and compute attributions
    for i_batch, batch in tqdm(
        enumerate(sdataloader),
        total=int(dataset_len / batch_size),
        desc=f"Computing saliency on batches of size {batch_size}",
    ):
        _, x, x_rev_comp, y = batch
        if model.strand == "ss":
            curr_explanations = attribute(
                model,
                x,
                target=target,
                method=method,
                device=device,
                additional_forward_args=x_rev_comp[0],
                **kwargs,
            )
        else:
            curr_explanations = attribute(
                model,
                (x, x_rev_comp),
                target=target,
                method=method,
                device=device,
                **kwargs,
            )
        if (i_batch+1)*batch_size < dataset_len:
            if model.strand == "ss":
                all_forward_explanations[i_batch*batch_size:(i_batch+1)*batch_size] = curr_explanations.detach().cpu().numpy()
            else:
                all_forward_explanations[i_batch*batch_size:(i_batch+1)*batch_size] = curr_explanations[0].detach().cpu().numpy()
                all_reverse_explanations[i_batch * batch_size:(i_batch+1)*batch_size] = curr_explanations[1].detach().cpu().numpy()
        else:
            if model.strand == "ss":
                all_forward_explanations[i_batch * batch_size:dataset_len] = curr_explanations.detach().cpu().numpy()
            else:
                all_forward_explanations[i_batch*batch_size:dataset_len] = curr_explanations[0].detach().cpu().numpy()
                all_reverse_explanations[i_batch*batch_size:dataset_len] = curr_explanations[1].detach().cpu().numpy()
    
    # Add the attributions to sdata 
    if model.strand == "ss":
        sdata.uns[f"{prefix}{method}_imps{suffix}"] = all_forward_explanations
    else:
        if aggr == "max":
            sdata.uns[f"{prefix}{method}_imps{suffix}"] = np.maximum(all_forward_explanations, all_reverse_explanations)
        elif aggr == "mean":
            sdata.uns[f"{prefix}{method}_imps{suffix}"] = (all_forward_explanations + all_reverse_explanations) / 2
        elif aggr == None:
            sdata.uns[f"{prefix}{method}_forward_imps{suffix}"] = all_forward_explanations
            sdata.uns[f"{prefix}{method}_reverse_imps{suffix}"] = all_reverse_explanations
    return sdata if copy else None

def aggregate_importances_sdata(
    sdata, 
    uns_key,
    copy=False
):
    """
    Aggregate feature attribution scores for a SeqData
    
    This function aggregates the feature attribution scores for a SeqData object
    Parameters
    ----------
    sdata : SeqData
        SeqData object
    uns_key : str
        Key in the uns attribute of the SeqData object to use as feature attribution scores
    """
    sdata = sdata.copy() if copy else sdata
    vals = sdata.uns[uns_key]
    df = sdata.pos_annot.df
    agg_scores = []
    for i, row in df.iterrows():
        seq_id = row["Chromosome"]
        start = row["Start"]
        end = row["End"]
        seq_idx = np.where(sdata.names == seq_id)[0][0]
        agg_scores.append(vals[seq_idx, :, start:end].sum())
    df[f"{uns_key}_agg_scores"] = agg_scores
    ranges = pr.PyRanges(df)
    sdata.pos_annot = ranges
    return sdata if copy else None

def tfmodisco_sdata(
    
):
    print("Cmon")