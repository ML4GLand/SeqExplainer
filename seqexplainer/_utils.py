import logging
import os
from typing import Callable, Dict
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _list_available_layers(
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

def _get_layer(
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

def _model_to_device(
    model: torch.nn.Module,
    device: str = "cpu"
) -> torch.nn.Module:
    """Move a model to a device and set it in eval mode

    Parameters
    ----------
    model : torch.nn.Module
        The model to move to a device
    device : str, optional
        The device to move the model to, by default "cpu"

    Returns
    -------
    torch.nn.Module
        The model moved to the device and set in eval mode
    """
    model.eval()
    model.to(device)
    return model

def _report_gpu() -> None:
   """
    Garbage colllects, empties the cache, and synchronizes the GPU
   """
   gc.collect()
   torch.cuda.empty_cache()
   torch.cuda.synchronize()
   print(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")

def _make_dirs(
    output_dir: str,
    overwrite: bool = False
) -> None:
    """Make a directory with an option to overwrite it if it already exists

    Parameters
    ----------
    output_dir : str
        The directory to create
    overwrite : bool, optional
        Whether to overwrite the directory if it already exists, by default False
    """

    if os.path.exists(output_dir):
        if overwrite:
            logging.info("Overwriting existing directory: {}".format(output_dir))
            os.system("rm -rf {}".format(output_dir))
        else:
            print("Output directory already exists: {}".format(output_dir))
            return
    os.makedirs(output_dir)

def _path_to_image_html(path: str):
    """Create an HTML image tag from a path"""
    return '<img src="'+ path + '" width="240" >'

def _k_largest_index_argsort(
    a: np.ndarray, 
    k: int = 1
) -> np.ndarray:
    """Returns the indeces of the k largest values of a numpy array. 
    
    If a is multi-dimensional, the indeces are returned as an array k x d array where d is 
    the dimension of a. The kth row represents the kth largest value of the overall array.
    The dth column returned repesents the index of the dth dimension of the kth largest value.
    So entry [i, j] in the return array represents the index of the jth dimension of the ith
    largets value in the overall array.

    a = array([[38, 14, 81, 50],
               [17, 65, 60, 24],
               [64, 73, 25, 95]])

    k_largest_index_argsort(a, k=2)
    array([[2, 3],  # first largest value is at [2,3] of the array (95)
           [0, 2]])  # second largest value is at [0,2] of the array (81)


    Parameters
    ----------
    a : numpy array
        The array to get the k largest values from.
    k : int
        The number of largest values to get.
    
    Returns
    -------
    numpy array
        The indexes of the k largest values of a.

    Note
    ----
    This is from:
    https://stackoverflow.com/questions/43386432/how-to-get-indexes-of-k-maximum-values-from-a-numpy-multidimensional-arra
    """
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))

def _create_unique_seq_names(
    n_seqs: int
) -> np.ndarray:
    """Create a list of unique sequence names

    Parameters
    ----------
    n_seqs : int
        The number of sequences to create names for
    
    Returns
    -------
    np.ndarray
        A list of unique sequence names
    """
    n_digits = len(str(n_seqs - 1))
    return ["seq{num:0{width}}".format(num=i, width=n_digits) for i in range(n_seqs)]
