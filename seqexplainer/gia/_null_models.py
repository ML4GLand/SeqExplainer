import numpy as np
from ..preprocess._preprocess import dinuc_shuffle_seqs


def generate_profile_set(base_sequence, num_sample):
    """
    create a subset of sequences as background by matching nucleotide profiles
    :param base_sequence: sequences to use for matching
    :param num_sample: sample size
    :return: background set of onehot sequences
    """
    seq_model = np.mean(np.squeeze(base_sequence), axis=0)
    seq_model /= np.sum(seq_model, axis=0, keepdims=True)

    # sequence length
    L = seq_model.shape[1]

    x_null = np.zeros((num_sample, 4, L))
    for n in range(num_sample):

        # generate uniform random number for each nucleotide in sequence
        Z = np.random.uniform(0, 1, L)

        # calculate cumulative sum of the probabilities
        cum_prob = seq_model.cumsum(axis=0)

        # find bin that matches random number for each position
        for l in range(L):
            index = [j for j in range(4) if Z[l] < cum_prob[j, l]][0]
            x_null[n, index, l] = 1

    return x_null

def generate_shuffled_set(base_sequence, num_sample):
    """
    Funciton for creating a shuffled set of sequences based on an input set
    :param base_sequence: sequences to shuffle
    :param num_sample: sample size
    :return: background set of onehot sequences
    """
    # take a random subset of base_sequence
    shuffle = np.random.permutation(len(base_sequence))
    x_null = base_sequence[shuffle[:num_sample]]

    # shuffle nucleotides
    [np.random.shuffle(x) for x in x_null]
    return x_null

def generate_dinucleotide_shuffled_set(base_sequence, num_sample):
    """
    Function for dinuc shuffling provided sequences
    :param base_sequence: set of sequences
    :param num_sample: sample size
    :return: background set of onehot sequences
    """
    # take a random subset of base_sequence
    shuffle = np.random.permutation(len(base_sequence))
    x_null = base_sequence[shuffle[:num_sample]]

    # shuffle dinucleotides
    x_null = dinuc_shuffle_seqs(x_null)
    return x_null

def generate_subset_set(base_sequence, num_sample):
    """
    Function for creating a subset of sequences
    :param base_sequence: set of sequences
    :param num_sample: sample size
    :return: background set of onehot sequences
    """
    # take a random subset of base_sequence
    shuffle = np.random.permutation(len(base_sequence))
    x_null = base_sequence[shuffle[:num_sample]]
    return x_null
