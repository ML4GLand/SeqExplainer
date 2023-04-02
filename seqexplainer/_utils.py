import os
import torch
import logging
import numpy as np
import pandas as pd
from umap import UMAP
import torch.nn as nn
from typing import Dict, Callable
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

class FeatureExtractor(nn.Module):
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
    
def get_layer(
    model, 
    layer_name
):
    return dict([*model.named_modules()])[layer_name]

def get_layer_outputs(
        model, 
        inputs, 
        layer_name,
        batch_size=128,
        device="cpu"
    ):
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

def pca(mtx, n_comp=30, index_name='index', new_index=None):
    """
    Function to perform scaling and PCA on an input matrix
    Parameters
    ----------
    mtx : sample by feature
    n_comp : number of pcs to return
    index_name : name of index if part of input
    Returns
    sklearn pca object and pca dataframe
    -------
    """
    print("Make sure your matrix is sample by feature")
    scaler = StandardScaler()
    scaler.fit(mtx)
    mtx_scaled = scaler.transform(mtx)
    pca_obj = PCA(n_components=n_comp)
    pca_obj.fit(mtx_scaled)
    pca_df = pd.DataFrame(pca_obj.fit_transform(mtx_scaled))
    pca_df.columns = ['PC' + str(col+1) for col in pca_df.columns]
    pca_df.index = new_index if new_index is not None else _create_unique_seq_names(mtx.shape[0])
    pca_df.index.name = index_name
    return pca_obj, pca_df

def umap(mtx, index_name='index', new_index=None):
    """
    Function to perform scaling and UMAP on an input matrix
    Parameters
    ----------
    mtx : sample by feature
    index_name : name of index if part of input
    Returns
    umap object and umap dataframe
    -------
    """
    print("Make sure your matrix is sample by feature")
    scaler = StandardScaler()
    scaler.fit(mtx)
    mtx_scaled = scaler.transform(mtx)
    umap_obj = UMAP()
    umap_obj.fit(mtx_scaled)
    umap_df = pd.DataFrame(umap_obj.transform(mtx_scaled))
    umap_df.columns = ['UMAP' + str(col+1) for col in umap_df.columns]
    umap_df.index = new_index if new_index is not None else _create_unique_seq_names(mtx.shape[0])
    umap_df.index.name = index_name
    return umap_obj, umap_df
