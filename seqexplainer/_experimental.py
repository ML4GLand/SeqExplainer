import torch
from torch.optim import Adam

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from yuzu.utils import perturbations


# my own
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

# MOANA (MOtif ANAlysis)
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

# MOANA (MOtif ANAlysis)
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

# MOANA (MOtif ANAlysis)
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

# peter koo
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

# peter koo
def batch_np(whole_dataset, batch_size):
    """
    batch a np array for passing to a model without running out of memory
    :param whole_dataset: np array dataset
    :param batch_size: batch size
    :return: generator of np batches
    """
    for i in range(0, whole_dataset.shape[0], batch_size):
        yield whole_dataset[i:i + batch_size]

# generative stuff
def seqs_from_tensor(tensor : torch.tensor, num_seqs : int = 1) -> np.ndarray:
    """
    Decodes sequences represented by tensors into their string values.

    Parameters
    ----------
    tensor : torch.tensor
        Tensor to be decoded.
    num_seqs : int
        Number of sequences to decode.
        Default is 1.

    Returns
    -------
    seqs : np.ndarray
        Numpy array of decoded sequences.
    """
    tokens = np.argmax(tensor.detach().numpy(), axis=1).reshape(num_seqs, -1)
    seqs = np.array([decode_seq(_token2one_hot(token)) for token in tokens])
    return seqs

# EUGENe for generative models
def latent_interpolation(latent_dim : int, samples : int, num_seqs : int = 1, generator = None, normal : bool = False, inclusive : bool = True) -> list:
    """
    Linearly interpolates between two random latent points. Useful for visualizing generative generators.

    Parameters
    ----------
    latent_dim : int
        Latent dimension of random latent space points.
    samples : int
        Number of samples to make between the two latent points. Higher numbers represent more sequences and should show smoother results.
    num_seqs : int
        Number of sequence channels to interpolate.
        Defaults to 1.
    generator : Basgeneratore
        If provided, interpolated values will be passed through the given generator and be returned in place of latent points.
        Default is None.
    normal : bool
        Whether to randomly sample points from a standard normal distibution rather than from 0 to 1.
        Defaults to False.
    inclusive : bool
        Whether to returm random latent points along with their interpolated samples.
        Defaults to True.

    Returns
    -------
    z_list : list
        List of latent space points represented as tensors.
    gen_seqs : list
        List of tokenized sequences expressed as strings.
        Returns in place of z_list when a generator is provided.
    """
    if normal:
        z1 = torch.normal(0, 1, (num_seqs, latent_dim))
        z2 = torch.normal(0, 1, (num_seqs, latent_dim))
    else:
        z1 = torch.rand(num_seqs, latent_dim)
        z2 = torch.rand(num_seqs, latent_dim)

    z_list = []
    for n in range(samples):
        weight = (n + 1)/(samples + 1)
        z_interp = torch.lerp(z1, z2, weight)
        z_list.append(z_interp)
    if inclusive:
        z_list.insert(0, z1)
        z_list.append(z2)

    if generator is None:
        return z_list

    gen_seqs = []
    for z in z_list:
        gen_seq = seqs_from_tensor(generator(z))
        gen_seqs.append(gen_seq)
    return gen_seqs

# EUGENe for generative models
def generate_seqs_from_generator(generator, num_seqs : int = 1, normal : bool = False, device : str = "cpu"):
    """
    Generates random sequences from a generative generator.

    Parameters
    ----------
    generator : Basgeneratore
        Generative generator used for sequence generation.
    num_seqs : int
        Number of sequences to decode.
        Default is 1.
    normal : bool
        Whether to sample from a normal distribution instead of a uniform distribution.
        Default is false.

    Returns
    -------
    seqs : np.ndarray
        Numpy array of decoded sequences.
    """
    if normal:
        z = torch.Tensor(np.random.normal(0, 1, (num_seqs, generator.latent_dim)))
    else: 
        z = torch.rand(num_seqs, generator.latent_dim)
    z = z.to(device)
    fake = generator(z)
    return seqs_from_tensor(fake.cpu(), num_seqs)