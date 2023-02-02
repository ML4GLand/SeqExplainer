import tfomics
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
from matplotlib import colors
from ._utils import compute_per_position_ic, _make_dirs
from ._seq import nucleotide_content_seqs


def plot_saliency_map(explains, sort, width=13, height_per_explain=1):
    """
    Plot the saliency maps for each sequence
    """
    num_plot = len(explains)
    fig = plt.figure(figsize=(width, num_plot*height_per_explain))
    for i in range(num_plot):
        ax = plt.subplot(num_plot, 1, i+1)
        saliency_df = pd.DataFrame(explains[i].transpose([1,0]), columns=["A","C","G","T"])
        saliency_df.index.name = "pos"
        tfomics.impress.plot_attribution_map(saliency_df, ax, figsize=(num_plot,1))
        plt.ylabel(sort[i])

def plot_weights(
    array, 
    pwm=False, 
    figsize=(10,3), 
    save=None,
    **kwargs
):
    if pwm:
        ic = compute_per_position_ic(array, background=[0.25, 0.25, 0.25, 0.25], pseudocount=0.001)
        array = array*ic[:, None]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    df = pd.DataFrame(array, columns=['A', 'C', 'G', 'T'])
    df.index.name = 'pos'
    crp_logo = logomaker.Logo(df, ax=ax, font_name='Arial Rounded', **kwargs)
    crp_logo.style_spines(visible=False)
    plt.ylim(min(df.sum(axis=1).min(), 0), df.sum(axis=1).max())
    plt.show()

    if save:
        dir = save.split("/")[:-1]
        _make_dirs("/".join(dir))
        plt.savefig(save, dpi=300, bbox_inches='tight')

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

def plot_nucleotide_freq(seqs, title="", ax=None, figsize=(10, 5)):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    nuc_across_seq = nucleotide_content_seqs(seqs, axis=0, ohe=True, normalize=True)
    ax.plot(nuc_across_seq.T)
    ax.legend(["A", "C", "G", "T"])
    ax.set_xlabel("Position")
    ax.set_ylabel("Frequency")
    