import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from typing import Callable, Dict
from ._utils import _model_to_device


class FeatureExtractor(nn.Module):
    """A module to extract the outputs of a layer in a model
    TODO: Comment this
    """
    def __init__(self, model: nn.Module, key_word: str):
        super().__init__()
        self.model = model
        layers = sorted([k for k in dict([*model.named_modules()]) if key_word in k])
        self.features = {layer: torch.empty(0) for layer in layers}
        self.handles = dict()

        for layerID in layers:
            layer = dict([*self.model.named_modules()])[layerID]
            handle = layer.register_forward_hook(self.SaveOutputHook(layerID))
            self.handles[layerID] = handle
            
    def SaveOutputHook(self, layerID: str) -> Callable:
        def fn(layer, input, output):
            self.features[layerID] = output
        return fn

    def forward(self, x, **kwargs) -> Dict[str, torch.Tensor]:
        preds = self.model(x, **kwargs)
        return self.features, self.handles, preds
    
def list_available_layers(
    model: nn.Module,
    key_word = None
) -> list:
    """List the available layers in a model
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to list the layers of
    key_word : str, optional
        A key word to filter the layers by, by default None
    
    Returns
    -------
    list
        A list of the available layers in the model
    """
    layers = sorted([k for k in dict([*model.named_modules()])])
    if key_word is not None:
        layers = [layer for layer in layers if key_word in layer]
    return layers

def get_layer(
    model: nn.Module,
    layer_name: str
) -> nn.Module:
    """Get a layer from a model by name. Note that this will only work for
    named modules. If the model has unnamed modules, TODO

    Parameters
    ----------
    model : torch.nn.Module
        The model to get the layer from
    layer_name : str
        The name of the layer to get

    Returns
    -------
    torch.nn.Module
        The layer from the model
    """
    return dict([*model.named_modules()])[layer_name]
    
def get_layer_outputs(
        model: nn.Module, 
        inputs: torch.Tensor,
        layer_name: str,
        batch_size: int = 32,
        device: str = "cpu"
    ):
    """Get the outputs of a layer in a model for a given input

    Parameters
    ----------
    model : nn.Module
        The model to get the layer outputs from
    inputs : torch.Tensor
        The input to the model
    layer_name : str
        The name of the layer to get the outputs from
    batch_size : int, optional
        The batch size to use when computing the outputs, by default 32
    device : str, optional
        The device to use, by default "cpu"

    Returns
    -------
    np.ndarray
        The outputs of the layer for the given input
    """
    # Set-up model feature extractor on device
    model = _model_to_device(model, device)
    model_extract = FeatureExtractor(model, layer_name)
    
    # Create torch tensor from inputs if ndarray
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).float()
    assert isinstance(inputs, torch.Tensor), "inputs must be a torch.Tensor or np.ndarray"

    # Create an empty list to hold outputs
    layer_outs = []
    starts = np.arange(0, inputs.shape[0], batch_size)
    for _, start in tqdm(
        enumerate(starts),
        total=len(starts),
        desc=f"Computing layer outputs for layer {layer_name} on batches of size {batch_size}",
    ):
        inputs_ = inputs[start : start + batch_size]
        inputs_ = inputs_.to(device)
        layer_outs.append(model_extract(inputs_)[0][layer_name])
    layer_outs = torch.cat(layer_outs, dim=0).detach().cpu().numpy()
    return layer_outs
