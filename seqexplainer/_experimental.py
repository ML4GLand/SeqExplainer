import torch
from torch.optim import Adam

def generate_maximally_activating_input(model, neuron_index, input_size, lr=0.1, steps=100):
    # Create a random input tensor with the specified size
    input_tensor = torch.randn(input_size)
    input_tensor.requiresGrad = True
    
    # Define the optimizer
    optimizer = Adam([input_tensor], lr=lr)
    
    # Perform gradient ascent
    for _ in range(steps):
        optimizer.zero_grad()
        output = model(input_tensor)
        # get the output of the intermediate layer
        intermediate_output = model.intermediate_layer(input_tensor)
        # select the output of the neuron of interest
        neuron_output = intermediate_output[0, neuron_index]
        neuron_output.backward()
        optimizer.step()
    
    return input_tensor

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

def gc_bias_gia(
    model,
    null_sequences,
    pattern,
    position
):
    """Test the GC content of a pattern using GIA
    
    Embed a pattern in a sequence at a specific position and embed a GC motif at the same position
    """
    pass

def positional_bias_gia(
    model,
    null_sequences,
    pattern
):
    pass

def flanking_patterns_gia(
    model,
    null_sequences,
    pattern,
    position,
    flank_size,
):
    """Test the flanks of an embedded pattern using GIA
    
    Embed a pattern in a sequence at a specific position
    """
    pass

def pattern_interaction_gia(
    model,
    null_sequences,
    pattern1,
    position,
    pattern2,
):
    """Test the interaction between two patterns using GIA
    
    Embed a pattern in a sequence at a specific position and then vary the position of the second pattern
    """
    pass

def pattern_cooperativity_gia(
    model,
    null_sequence,
    pattern1,
    position1,
    pattern2,
    position2
):
    """Test the cooperativity between two patterns using GIA
    
    Embed two patterns at specific positions. First test the importance of the first pattern, then the second pattern, 
    then the interaction between the two patterns.
    """
    pass

def pattern_occlusion_gia(
    model,
    null_sequences,
    pattern,
    max_occluded,
    num_random_patterns,
):
    """Test the occlusion of a pattern using GIA
    
    Find a pattern in a sequence and replace with a random pattern a set number of times
    """
    pass

def position_occlusion_gia(
    model,
    null_sequences,
    positions,
    occlusion_length,
):
    """Test the occlusion of a position using GIA
    
    Randomly replace a position in a sequence with a random pattern. Can either be a passed in set of positions with
    a fixed or variable set of lengths, or a set of ranges.
    """
    pass

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm
from ..preprocess._utils import _get_vocab
from ._utils import _k_largest_index_argsort
from ..utils import track
from .._settings import settings


def _get_first_conv_layer(
    model: nn.Module, 
    device: str = "cpu"
):
    """
    Get the first convolutional layer of a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to get the first convolutional layer from.
    device : str, optional
        The device to move the layer to, by default "cpu"

    Returns
    -------
    torch.nn.Module
        The first convolutional layer in the model.
    """
    if model.__class__.__name__ == "Jores21CNN":
        layer_shape = model.biconv.kernels[0].shape
        kernels = model.biconv.kernels[0]
        biases = model.biconv.biases[0]
        layer = nn.Conv1d(
            in_channels=layer_shape[1],
            out_channels=layer_shape[0],
            kernel_size=layer_shape[2],
            padding="same",
        )
        layer.weight = nn.Parameter(kernels)
        layer.bias = nn.Parameter(biases)
        return layer.to(device)
    elif model.__class__.__name__ == "Kopp21CNN":
        return model.conv
    for layer in model.convnet.module:
        name = layer.__class__.__name__
        if name == "Conv1d":
            first_layer = model.convnet.module[0]
            return first_layer.to(device)
    print("No Conv1d layer found, returning None")
    return None


def _get_activations_from_layer(
    layer: nn.Module, 
    sdataloader: DataLoader,
    device: str = "cpu", 
    vocab: str = "DNA"
):
    """
    Get the values of activations using a passed in layer and sequence inputs in a dataloader
    TODO: We currently generate the sequences for all activators which involves decoding
    all of them. We only need to do this for the maximal activating seqlets

    Parameters
    ----------
    layer : torch.nn.Module
        The layer to get activations from.
    sdataloader : DataLoader
        The dataloader to get sequences from.
    device : str, optional
        The device to move the layer to, by default "cpu"
    vocab : str, optional
        The vocabulary to use, by default "DNA"

    Returns
    -------
    np.ndarray
        The activations from the layer.

    Note
    ----
    We currently only use forward sequences for computing activations of the layer,
    we do not currenlty include reverse complements
    """
    from ..preprocess import decode_seqs
    activations = []
    sequences = []
    dataset_len = len(sdataloader.dataset)
    batch_size = sdataloader.batch_size
    for i_batch, batch in tqdm(
        enumerate(sdataloader),
        total=int(dataset_len / batch_size),
        desc="Getting maximial activating seqlets",
    ):
        ID, x, x_rev_comp, y = batch
        sequences.append(decode_seqs(x.detach().cpu().numpy(), vocab=vocab, verbose=False))
        x = x.to(device)
        layer = layer.to(device)
        activations.append(F.relu(layer(x)).detach().cpu().numpy())
        np_act = np.concatenate(activations)
        np_seq = np.concatenate(sequences)
    return np_act, np_seq


def _get_filter_activators(
    activations: np.ndarray,
    sequences: np.ndarray,
    kernel_size: int,
    num_filters: int = None,
    method: str = "Alipanahi15",
    threshold: float = 0.5,
    num_seqlets: int = 100,
):
    """
    Get the sequences that activate a filter the most using a passed in method.
    We currently implement two methods, Alipanahi15 and Minnoye20.
    
    Parameters
    ----------
    activations : np.ndarray
        The activations from the layer.
    sequences : np.ndarray
        The sequences corresponding to the activations in a numpy array.
    kernel_size : int
        The kernel size of the layer.
    num_filters : int, optional
        The number of filters to get seqlets for, by default None
    method : str, optional
        The method to use, by default "Alipanahi15"
    threshold : float, optional
        The threshold for filtering activations, by default 0.5
    num_seqlets : int, optional
        The number of seqlets to get, by default 100

    Returns
    -------
    np.ndarray
        The sequences that activate the filter the most.

    Note
    ----
    We currently only use forward sequences for computing activations of the layer,
    we do not currenlty include reverse complements
    """
    num_filters = num_filters if num_filters is not None else activations.shape[1]
    if method == "Alipanahi15":
        assert (threshold is not None), "Threshold must be specified for Alipanahi15 method."
        filter_activators = []
        for i, filt in tqdm(
            enumerate(range(num_filters)),
            desc=f"Getting filter activators for {num_filters} filters",
            total=num_filters,
        ):
            single_filter = activations[:, filt, :]
            max_val = np.max(single_filter)
            activators = []
            for i in range(len(single_filter)):
                starts = np.where(single_filter[i] >= max_val * threshold)[0]
                for start in starts:
                    activators.append(sequences[i][start : start + kernel_size])
            filter_activators.append(activators)
    elif method == "Minnoye20":
        assert (num_seqlets is not None), "num_seqlets must be specified for Minnoye20 method."
        filter_activators = []
        for i, filt in tqdm(
            enumerate(range(num_filters)),
            desc=f"Getting filter activators for {num_filters} filters",
            total=num_filters,
        ):
            single_filter = activations[:, filt, :]
            inds = _k_largest_index_argsort(single_filter, num_seqlets)
            filter_activators.append(
                [
                    seq[inds[i][1] : inds[i][1] + kernel_size]
                    for i, seq in enumerate(sequences[inds[:, 0]])
                ]
            )
    return filter_activators


def _get_pfms(
    filter_activators: np.ndarray,
    kernel_size: int,
    vocab: str = "DNA",
):  
    """
    Generate position frequency matrices for the maximal activating seqlets in filter_activators 

    Parameters
    ----------
    filter_activators : np.ndarray
        The sequences that activate the filter the most.
    kernel_size : int
        The kernel size of the layer.
    vocab : str, optional
        The vocabulary to use, by default "DNA"

    Returns
    -------
    np.ndarray
        The position frequency matrices for the maximal activating seqlets in filter_activators
    """
    filter_pfms = {}
    vocab = _get_vocab(vocab)
    for i, activators in tqdm(
        enumerate(filter_activators),
        total=len(filter_activators),
        desc="Getting PFMs from filters",
    ):
        pfm = {
            vocab[0]: np.zeros(kernel_size),
            vocab[1]: np.zeros(kernel_size),
            vocab[2]: np.zeros(kernel_size),
            vocab[3]: np.zeros(kernel_size),
            "N": np.zeros(kernel_size),
        }
        for seq in activators:
            for j, nt in enumerate(seq):
                pfm[nt][j] += 1
        filter_pfm = pd.DataFrame(pfm)
        filter_pfm = filter_pfm.drop("N", axis=1)
        filter_pfms[i] = filter_pfm
        filter_pfms[i] = filter_pfms[i].div(filter_pfms[i].sum(axis=1), axis=0)
    return filter_pfms


@track
def generate_pfms_sdata(
    model: nn.Module,
    sdata,
    method: str = "Alipanahi15",
    vocab: str = "DNA",
    num_filters: int = None,
    threshold: float = 0.5,
    num_seqlets: int = 100,
    batch_size: int = None,
    num_workers: int = None,
    device: str = "cpu",
    transform_kwargs: dict = {},
    key_name: str = "pfms",
    prefix: str = "",
    suffix: str = "",
    copy: bool = False,
    **kwargs
):
    """
    Generate position frequency matrices for the maximal activating seqlets in the first 
    convolutional layer of a model. This involves computing the activations of the layer
    and then getting the sequences that activate the filter the most. We currently implement
    two methods, Alipanahi15 and Minnoye20. Using the maximally activating seqlets we then
    generate position frequency matrices for each filter and store those in the uns variable
    of the sdata object.

    Parameters
    ----------
    model
        The model to generate the PFMs with
    sdata
        The SeqData object holding sequences and to store the PFMs in
    method : str, optional
        The method to use, by default "Alipanahi15". This takes the all
        seqlets that activate the filter by more than half its maximum activation
    vocab : str, optional
        The vocabulary to use when decoding the sequences to create the PFM, by default "DNA"
    num_filters : int, optional
        The number of filters to get seqlets for, by default None. If not none will take the first
        num_filters filters in the model
    threshold : float, optional
        For Alipanahi15 method, the threshold defining maximally activating seqlets, by default 0.5
    num_seqlets : int, optional
        For Minnoye20 method, the number of seqlets to get, by default 100
    batch_size : int, optional
        The batch size to use when computing activations, by default None
    num_workers : int, optional
        The number of workers to use when computing activations, by default None
    device : str, optional
        The device to use when computing activations, by default "cpu" but will use gpu automatically if
        available
    transform_kwargs : dict, optional
        The kwargs to use when transforming the sequences when dataloading, by default ({})
        no arguments are passed (i.e. sequences are assumed to be ready for dataloading)
    key_name : str, optional
        The key to use when storing the PFMs in the uns variable of the sdata object, by default "pfms"
    prefix : str, optional
        The prefix to use when storing the PFMs in the uns variable of the sdata object, by default ""
    suffix : str, optional
        The suffix to use when storing the PFMs in the uns variable of the sdata object, by default ""
    copy : bool, optional
        Whether to return a copy the sdata object, by default False and sdata is modified in place
    """
    sdata = sdata.copy() if copy else sdata
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = batch_size if batch_size is not None else settings.batch_size
    num_workers = num_workers if num_workers is not None else settings.dl_num_workers
    sdataset = sdata.to_dataset(target_keys=None, transform_kwargs=transform_kwargs)
    sdataloader = DataLoader(
        sdataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False
    )
    first_layer = _get_first_conv_layer(
        model, 
        device=device
    )
    activations, sequences = _get_activations_from_layer(
        first_layer, 
        sdataloader, 
        device=device, 
        vocab=vocab
    )
    filter_activators = _get_filter_activators(
        activations,
        sequences,
        first_layer.kernel_size[0],
        num_filters=num_filters,
        method=method,
        threshold=threshold,
        num_seqlets=num_seqlets,
    )
    filter_pfms = _get_pfms(
        filter_activators, 
        first_layer.kernel_size[0], 
        vocab=vocab
    )
    sdata.uns[f"{prefix}{key_name}{suffix}"] = filter_pfms
    return sdata if copy else None

#https://github.com/MedChaabane/deepRAM/blob/master/extract_motifs.py
def plot_target_corr(filter_outs, seq_targets, filter_names, target_names, out_pdf, seq_op='mean'):
    num_seqs = filter_outs.shape[0]
    num_targets = len(target_names)

    if seq_op == 'mean':
        filter_outs_seq = filter_outs.mean(axis=2)
    else:
        filter_outs_seq = filter_outs.max(axis=2)

    # std is sequence by filter.
    filter_seqs_std = filter_outs_seq.std(axis=0)
    filter_outs_seq = filter_outs_seq[:,filter_seqs_std > 0]
    filter_names_live = filter_names[filter_seqs_std > 0]

    filter_target_cors = np.zeros((len(filter_names_live),num_targets))
    for fi in range(len(filter_names_live)):
        for ti in range(num_targets):
            cor, p = spearmanr(filter_outs_seq[:,fi], seq_targets[:num_seqs,ti])
            filter_target_cors[fi,ti] = cor

    cor_df = pd.DataFrame(filter_target_cors, index=filter_names_live, columns=target_names)

    sns.set(font_scale=0.3)
    plt.figure()
    sns.clustermap(cor_df, cmap='BrBG', center=0, figsize=(8,10))
    plt.savefig(out_pdf)
    plt.close()

#https://github.com/p-koo/tfomics/blob/master/tfomics/impress.py
def plot_filters(W, fig, num_cols=8, alphabet='ACGT', names=None, fontsize=12):
  """plot 1st layer convolutional filters"""

  num_filter, filter_len, A = W.shape
  num_rows = np.ceil(num_filter/num_cols).astype(int)

  fig.subplots_adjust(hspace=0.2, wspace=0.2)
  for n, w in enumerate(W):
    ax = fig.add_subplot(num_rows,num_cols,n+1)
    
    # Calculate sequence logo heights -- information
    I = np.log2(4) + np.sum(w * np.log2(w+1e-7), axis=1, keepdims=True)
    logo = I*w

    # Create DataFrame for logomaker
    counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(filter_len)))
    for a in range(A):
      for l in range(filter_len):
        counts_df.iloc[l,a] = logo[l,a]

    logomaker.Logo(counts_df, ax=ax)
    ax = plt.gca()
    ax.set_ylim(0,2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    plt.xticks([])
    plt.yticks([])
    if names is not None:
      plt.ylabel(names[n], fontsize=fontsize)


import torch
import numpy as np
from tqdm.auto import tqdm
from ._utils import _k_largest_index_argsort, _naive_ism
from ..preprocess import ohe_seqs, feature_implant_across_seq
from ..utils import track
from .. import settings


def best_k_muts(
    model: torch.nn.Module, 
    X: np.ndarray, 
    k: int = 1, 
    device: str = None
) -> np.ndarray:
    """
    Find and return the k highest scoring sequence from referenece sequence X.

    Using ISM, this function calculates all the scores of all possible mutations
    of the reference sequence using a trained model. It then returns the k highest
    scoring sequences, along with delta scores from the reference sequence and the indeces
    along the lengths of the sequences where the mutations can be found

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    X: np.ndarray
        The one-hot encoded sequence to calculate find mutations for.
    k: int, optional
        The number of mutated seqences to return
    device: str, optional
        Whether to use a 'cpu' or 'cuda'.

    Returns
    -------
    mut_X: np.ndarray
        The k highest scoring one-hot-encoded sequences
    maxs: np.ndarray
        The k highest delta scores corresponding to the k highest scoring sequences
    locs: np.ndarray
        The indeces along the length of the sequences where the mutations can be found
    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval().to(device)
    X = np.expand_dims(X, axis=0) if X.ndim == 2 else X
    X = X.transpose(0, 2, 1) if X.shape[2] == 4 else X
    X = torch.Tensor(X).float().numpy()
    X_ism = _naive_ism(model, X, device=device, batch_size=1)
    X_ism = X_ism.squeeze(axis=0)
    inds = _k_largest_index_argsort(X_ism, k)
    locs = inds[:, 1]
    maxs = np.max(X_ism, axis=0)[locs]
    mut_Xs = np.zeros((k, X.shape[1], X.shape[2]))
    for i in range(k):
        mut_X = X.copy().squeeze(axis=0)
        mut_X[:, inds[i][1]] = np.zeros(mut_X.shape[0])
        mut_X[:, inds[i][1]][inds[i][0]] = 1
        mut_Xs[i] = mut_X
    return mut_Xs, maxs, locs


def best_mut_seqs(
    model: torch.nn.Module,
    X: np.ndarray, 
    batch_size: int = None, 
    device: str = None
) -> np.ndarray:
    """Find and return the highest scoring sequence for each sequence from a set reference sequences X. 
    
    X should contain one-hot-encoded sequences
    and should be of shape (n, 4, l). n is the number of sequences, 4 is the number of
    nucleotides, and l is the length of the sequence.

    Using ISM, this function calculates all the scores of all possible mutations
    of each reference sequence using a trained model. It then returns the highest
    scoring sequence, along with delta scores from the reference sequence and the indeces
    along the lengths of the sequences where the mutations can be found.

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    X: np.ndarray
        The one-hot encoded sequences to calculate find mutations for.
    batch_size: int, optional
        The number of sequences to score at once.
    device: str, optional
        Whether to use a 'cpu' or 'cuda'.

    Returns
    -------
    mut_X: np.ndarray
        The highest scoring one-hot-encoded sequences
    maxs: np.ndarray
        The highest delta scores corresponding to the highest scoring sequences
    locs: np.ndarray
        The indeces along the length of the sequences where the mutations can be found

    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = settings.batch_size if batch_size is None else batch_size
    model.eval().to(device)
    X = X.transpose(0, 2, 1) if X.shape[2] == 4 else X
    X = torch.Tensor(X).float().numpy()
    X_ism = _naive_ism(model, X, device=device, batch_size=batch_size)
    maxs, inds, mut_X = [], [], X.copy()
    for i in range(len(mut_X)):
        maxs.append(np.max(X_ism[i]))
        ind = np.unravel_index(X_ism[i].argmax(), X_ism[i].shape)
        inds.append(ind[1])
        mut_X[i][:, ind[1]] = np.zeros(mut_X.shape[1])
        mut_X[i][:, ind[1]][ind[0]] = 1
    return mut_X, np.array(maxs), np.array(inds)


def evolution(
    model: torch.nn.Module,
    X: np.ndarray,
    rounds: int = 10,
    k: int = 10,
    force_different: bool = True,
    batch_size: int = None,
    device: str = "cpu",
) -> np.ndarray:
    """Perform rounds rounds of in-silico evolution on a single sequence X.

    Using ISM, this function calculates all the scores of all possible mutations
    on a starting sequence X. It then mutates the sequence and repeats the process
    for rounds rounds. In the end, it returns new "evolved" sequence after rounds mutations
    and the delta scores from the reference sequence and the indeces along the lengths of the sequences
    with which the mutations occured

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    X: np.ndarray
        The one-hot encoded reference sequence to start the evolution with
    rounds: int, optional
        The number of rounds of evolution to perform
    force_different: bool, optional
        Whether to force the mutations to occur at different locations in the reference sequence
    k: int, optional
        The number of mutated sequences to consider at each round. This is in case
        the same position scores highest multiple times.
    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    batch_size = settings.batch_size if batch_size is None else batch_size
    model.eval().to(device)
    curr_X = X.copy()
    mutated_positions, mutated_scores = [], []
    for r in range(rounds):
        mut_X, score, positions = best_k_muts(model, curr_X, k=k, device=device)
        if force_different:
            for i, p in enumerate(positions):
                if p not in mutated_positions:
                    curr_X = mut_X[i]
                    mutated_positions.append(p)
                    mutated_scores.append(score[i])
                    break
        else:
            curr_X = mut_X[0]
            mutated_positions.append(positions[0])
            mutated_scores.append(score[0])
    return curr_X, mutated_scores, mutated_positions


@track
def evolve_seqs_sdata(
    model: torch.nn.Module, 
    sdata, 
    rounds: int, 
    return_seqs: bool = False, 
    device: str = "cpu", 
    copy: bool = False, 
    **kwargs
):
    """
    In silico evolve a set of sequences that are stored in a SeqData object.

    Parameters
    ----------
    model: torch.nn.Module  
        The model to score the sequences with
    sdata: SeqData  
        The SeqData object containing the sequences to evolve
    rounds: int
        The number of rounds of evolution to perform
    return_seqs: bool, optional
        Whether to return the evolved sequences
    device: str, optional
        Whether to use a 'cpu' or 'cuda'.
    copy: bool, optional
        Whether to copy the SeqData object before mutating it
    kwargs: dict, optional
        Additional arguments to pass to the evolution function
    
    Returns
    -------
    sdata: SeqData
        The SeqData object containing the evolved sequences
    """
    sdata = sdata.copy() if copy else sdata
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval().to(device)
    evolved_seqs = np.zeros(sdata.ohe_seqs.shape)
    deltas = np.zeros((sdata.n_obs, rounds))
    for i, ohe_seq in tqdm(enumerate(sdata.ohe_seqs), total=len(sdata.ohe_seqs), desc="Evolving seqs"):
        evolved_seq, delta, _ = evolution(model, ohe_seq, rounds=rounds, device=device)
        evolved_seqs[i] = evolved_seq
        deltas[i, :] = deltas[i, :] + delta
    orig_seqs = torch.Tensor(sdata.ohe_seqs).to(device)
    original_scores = model(orig_seqs).detach().cpu().numpy().squeeze()
    sdata["original_score"] = original_scores
    sdata["evolved_1_score"] = original_scores + deltas[:, 0]
    for i in range(2, rounds + 1):
        sdata.seqs_annot[f"evolved_{i}_score"] = (
            sdata.seqs_annot[f"evolved_{i-1}_score"] + deltas[:, i - 1]
        )
    print(return_seqs)
    if return_seqs:
        evolved_seqs = torch.Tensor(evolved_seqs).to(device)
        return evolved_seqs
    return sdata if copy else None


def feature_implant_seq_sdata(
    model: torch.nn.Module,
    sdata,
    seq_id: str,
    feature: np.ndarray,
    feature_name: str = "feature",
    encoding: str = "str",
    onehot: bool = False,
    store: bool = True,
    device: str = "cpu",
):
    """
    Score a set of sequences with a feature inserted at every position of each sequence in sdata.
    For double stranded models, the feature is inserted in both strands, with the reverse complement
    of the feature in the reverse strand

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    sdata: SeqData
        The SeqData object containing the sequences to score
    seq_id: str
        The id of the sequence to score
    feature: np.ndarray
        The feature to insert into the sequences
    feature_name: str, optional
        The name of the feature
    encoding: str, optional
        The encoding of the feature. One of 'str', 'ohe', 'int'
    onehot: bool, optional
        Whether the feature is one-hot encoded
    store: bool, optional
        Whether to store the scores in the SeqData object
    device: str, optional
        Whether to use a 'cpu' or 'cuda'.

    Returns
    -------
    np.ndarray
        The scores of the sequences with the feature inserted if store is False
    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval().to(device)
    seq_idx = np.where(sdata.seqs_annot.index == seq_id)[0][0]
    if encoding == "str":
        seq = sdata.seqs[seq_idx]
        implanted_seqs = feature_implant_across_seq(seq, feature, encoding=encoding)
        implanted_seqs = ohe_seqs(implanted_seqs, vocab="DNA", verbose=False)
        X = torch.from_numpy(implanted_seqs).float()
    elif encoding == "onehot":
        seq = sdata.ohe_seqs[seq_idx]
        implanted_seqs = feature_implant_across_seq(
            seq, 
            feature, 
            encoding=encoding, 
            onehot=onehot
        )
        X = torch.from_numpy(implanted_seqs).float()
    else:
        raise ValueError("Encoding not recognized.")
    if model.strand == "ss":
        X = X.to(device)
        X_rev = X
    else:
        X = X.to(device)
        X_rev = torch.flip(X, [1, 2]).to(device)
    preds = model(X, X_rev).cpu().detach().numpy().squeeze()
    if store:
        sdata.seqsm[f"{seq_id}_{feature_name}_slide"] = preds
    return preds


def feature_implant_seqs_sdata(
    model: torch.nn.Module,
    sdata,
    feature: np.ndarray,
    seqsm_key: str = None, 
    device: str = "cpu",
    **kwargs
):
    """
    Score a set of sequences with a feature inserted at every position of each sequence in sdata

    Parameters
    ----------
    model: torch.nn.Module
        The model to score the sequences with
    sdata: SeqData
        The SeqData object containing the sequences to score
    feature: np.ndarray
        The feature to insert into the sequences
    seqsm_key: str, optional
        The key to store the scores in the SeqData object
    kwargs: dict, optional
        Additional arguments to pass to the feature_implant_seq_sdata function
    
    Returns
    -------
    np.ndarray
        The scores of the sequences with the feature inserted if seqsm_key is None
    """
    device = "cuda" if settings.gpus > 0 else "cpu" if device is None else device
    model.eval().to(device)
    predictions = []
    for i, seq_id in tqdm(
        enumerate(sdata.seqs_annot.index),
        desc="Implanting feature in all seqs of sdata",
        total=len(sdata.seqs_annot),
    ):
        predictions.append(
            feature_implant_seq_sdata(
                model, 
                sdata, 
                seq_id, 
                feature, 
                store=False, 
                **kwargs
            )
        )
    if seqsm_key is not None:
        sdata.seqsm[seqsm_key] = np.array(predictions)
    else:
        return np.array(predictions)

import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt


# MOANA (MOtif ANAlysis)

def activation_pwm(fmap, X, threshold=0.5, window=20):
    # Set the left and right window sizes
    window_left = int(window/2)
    window_right = window - window_left

    N, L, A = X.shape # assume this ordering (i.e., TensorFlow ordering) of channels in X
    num_filters = fmap.shape[-1]

    W = []
    for filter_index in range(num_filters):

        # Find regions above threshold
        coords = np.where(fmap[:,:,filter_index] > np.max(fmap[:,:,filter_index])*threshold)
        x, y = coords

        # Sort score
        index = np.argsort(fmap[x,y,filter_index])[::-1]
        data_index = x[index].astype(int)
        pos_index = y[index].astype(int)

        # Make a sequence alignment centered about each activation (above threshold)
        seq_align = []
        for i in range(len(pos_index)):

            # Determine position of window about each filter activation
            start_window = pos_index[i] - window_left
            end_window = pos_index[i] + window_right

            # Check to make sure positions are valid
            if (start_window > 0) & (end_window < L):
                seq = X[data_index[i], start_window:end_window, :]
                seq_align.append(seq)

        # Calculate position probability matrix
        if len(seq_align) > 0:
            W.append(np.mean(seq_align, axis=0))
        else:
            W.append(np.zeros((window, A)))
    W = np.array(W)

    return W


def clip_filters(W, threshold=0.5, pad=3):
    W_clipped = []
    for w in W:
        L, A = w.shape
        entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=1)
        index = np.where(entropy > threshold)[0]
        if index.any():
            start = np.maximum(np.min(index)-pad, 0)
            end = np.minimum(np.max(index)+pad+1, L)
            W_clipped.append(w[start:end,:])
        else:
            W_clipped.append(w)

    return W_clipped


def generate_meme(W, output_file='meme.txt', prefix='Filter'):
    # background frequency
    nt_freqs = [1./4 for i in range(4)]

    # open file for writing
    f = open(output_file, 'w')

    # print intro material
    f.write('MEME version 4\n')
    f.write('\n')
    f.write('ALPHABET= ACGT\n')
    f.write('\n')
    f.write('Background letter frequencies:\n')
    f.write('A %.4f C  %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
    f.write('\n')

    for j, pwm in enumerate(W):
        if np.count_nonzero(pwm) > 0:
            L, A = pwm.shape
            f.write('MOTIF %s%d \n' % (prefix, j))
            f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (L, L))
            for i in range(L):
                f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[i,:]))
            f.write('\n')
    
    f.close()


def plot_filters(W, fig, num_cols=8, alphabet="ACGT", names=None, fontsize=12):
    """plot first-layer convolutional filters from PWM"""
    
    if alphabet == "ATCG":
        W = W[:,:,[0,2,3,1]]
    
    num_filter, filter_len, A = W.shape
    num_rows = np.ceil(num_filter/num_cols).astype(int)

    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    for n, w in enumerate(W):
        ax = fig.add_subplot(num_rows,num_cols,n+1)

        # Calculate sequence logo heights -- information
        I = np.log2(4) + np.sum(w * np.log2(w+1e-7), axis=1, keepdims=True)
        logo = I*w
        
        # Create DataFrame for logomaker
        counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(filter_len)))
        for a in range(A):
            for l in range(filter_len):
                counts_df.iloc[l,a] = logo[l,a]
        
        logomaker.Logo(counts_df, ax=ax)
        ax = plt.gca()
        ax.set_ylim(0, 2) # set y-axis of all sequence logos to run from 0 to 2 bits
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])

        if names:
            plt.ylabel(names[n], fontsize=fontsize)
        
    return fig

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


def convert_tfr_to_np(tfr_dataset):
    """
    convert tfr dataset to a list of numpy arrays
    :param tfr_dataset: tfr dataset format
    :return:
    """
    all_data = [[] for i in range(len(next(iter(tfr_dataset))))]
    for i, (data) in enumerate(tfr_dataset):
        for j, data_type in enumerate(data):
            all_data[j].append(data_type)
    return [np.concatenate(d) for d in all_data]


def batch_np(whole_dataset, batch_size):
    """
    batch a np array for passing to a model without running out of memory
    :param whole_dataset: np array dataset
    :param batch_size: batch size
    :return: generator of np batches
    """
    for i in range(0, whole_dataset.shape[0], batch_size):
        yield whole_dataset[i:i + batch_size]