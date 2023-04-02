#import tfomics
import logomaker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from ._utils import _make_dirs

def skree_plot(pca_obj, n_comp=30):
    """
    Function to generate and output a Skree plot using matplotlib barplot
    Parameters
    ----------
    pca_obj : scikit-learn pca object
    n_comp : number of components to show in the plot
    Returns
    -------
    """
    variance={}
    for i,val in enumerate(pca_obj.explained_variance_ratio_.tolist()):
        key="PC"+str(i+1)
        variance[key]=val
    plt.bar(["PC"+str(i) for i in range(1,n_comp+1)],pca_obj.explained_variance_ratio_.tolist())
    plt.xticks(rotation=90)
    plt.ylabel("Variance Explained")
    plt.xlabel("Principal Component")
    return variance

def loadings_plot(eigvecs):
    fig, ax = plt.subplots()
    im = ax.imshow(eigvecs, cmap="bwr", norm=colors.CenteredNorm())
    ax.set_xticks(np.arange(eigvecs.shape[1]))
    ax.set_yticks(np.arange(eigvecs.shape[0]))
    ax.set_xticklabels(["PC{}".format(str(i+1)) for i in range(eigvecs.shape[1])])
    ax.set_yticklabels(["feature_{}".format(str(i+1)) for i in range(eigvecs.shape[0])])
    for i in range(eigvecs.shape[1]):
        for j in range(eigvecs.shape[0]):
            text = ax.text(j, i, round(eigvecs[i, j], 2), ha="center", va="center", color="k")
    fig.tight_layout()
    plt.show()

def pca_plot(pc_data, pc1=0, pc2=1, color="b", loadings=None, labels=None, n=5):
    xs = pc_data[:, pc1]
    ys = pc_data[:, pc2]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    ax = plt.scatter(xs * scalex, ys * scaley, c=color)
    if loadings is not None:
        if n > loadings.shape[0]:
            n = loadings.shape[0]
        for i in range(n):
            plt.arrow(0, 0, loadings[0, i], loadings[1, i], color='r', alpha=0.5, head_width=0.07, head_length=0.07, overhang=0.7)
        if labels is None:
            plt.text(loadings[0, i] * 1.2, loadings[1, i] * 1.2, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(loadings[0, i] * 1.2, loadings[1, i] * 1.2, labels[i], color='g', ha='center', va='center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.show()
    return ax

def umap_plot(umap_data, umap1=0, umap2=1, color="b", loadings=None, labels=None, n=5):
    xs = umap_data[:, umap1]
    ys = umap_data[:, umap2]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    ax = plt.scatter(xs * scalex, ys * scaley, c=color)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("UMAP{}".format(1))
    plt.ylabel("UMAP{}".format(2))
    plt.show()
    return ax
    