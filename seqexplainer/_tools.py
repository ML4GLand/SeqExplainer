import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from ._utils import _create_unique_seq_names


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