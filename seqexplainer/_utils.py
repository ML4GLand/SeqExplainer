import os
import logging
import torch
import torch.nn as nn
import numpy as np
from yuzu.utils import perturbations
from typing import Dict, Iterable, Callable

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, keyWord: str):
        super().__init__()
        self.model = model
        layers = sorted([k for k in dict([*model.named_modules()]) if keyWord in k])
        logging.info("{} model layers identified with key word {}".format(len(layers), keyWord))
        self.features = {layer: torch.empty(0) for layer in layers}
        self.handles = dict() 

        for layerID in layers:
            layer = dict([*self.model.named_modules()])[layerID]
            handle = layer.register_forward_hook(self.SaveOutputHook(layerID))
            self.handles[layerID] = handle
            
    def SaveOutputHook(self, layerID: str) -> Callable:
        def fn(laya, weValueYourInput, output): #laya = layer (e.g. Linear(...); weValueYourInput = input tensor
            self.features[layerID] = output
        return fn

    def forward(self, x, **kwargs) -> Dict[str, torch.Tensor]:
        preds = self.model(x, **kwargs)
        return self.features, self.handles, preds

def _model_to_device(model, device="cpu"):
    """
    """
    model.eval()
    model.to(device)
    return model

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
    n_seqs,
):
    n_digits = len(str(n_seqs - 1))
    return ["seq{num:0{width}}".format(num=i, width=n_digits) for i in range(n_seqs)]

def _make_dirs(
    output_dir,
    overwrite=False,
):
    if os.path.exists(output_dir):
        if overwrite:
            logging.info("Overwriting existing directory: {}".format(output_dir))
            os.system("rm -rf {}".format(output_dir))
        else:
            print("Output directory already exists: {}".format(output_dir))
            return
    os.makedirs(output_dir)

def _path_to_image_html(path):
    return '<img src="'+ path + '" width="240" >'
    
def compute_per_position_ic(ppm, background, pseudocount):
    alphabet_len = len(background)
    ic = ((np.log((ppm+pseudocount)/(1 + pseudocount*alphabet_len))/np.log(2))
          *ppm - (np.log(background)*background/np.log(2))[None,:])
    return np.sum(ic,axis=1)