
import torch
import numpy as np
from ._seq import _get_vocab, ohe_seq, decode_seq, decode_seqs, reverse_complement_seqs

def perturb_seq(seq):
    """Numpy version of perturbations"""
    n_choices, seq_len = seq.shape
    idxs = seq.argmax(axis=0)
    n = seq_len * (n_choices - 1)
    X = np.tile(seq, (n, 1))
    X = X.reshape(n, n_choices, seq_len)
    for k in range(1, n_choices):
        i = np.arange(seq_len) * (n_choices - 1) + (k - 1)
        X[i, idxs, np.arange(seq_len)] = 0
        X[i, (idxs + k) % n_choices, np.arange(seq_len)] = 1
    return X

def perturb_seq_torch(seq):
    """Torch version of perturbations"""
    n_choices, seq_len = seq.shape
    idxs = seq.argmax(axis=0)
    n = seq_len * (n_choices - 1)
    X = torch.tile(seq, (n, 1))
    X = X.reshape(n, n_choices, seq_len)
    for k in range(1, n_choices):
        i = torch.arange(seq_len) * (n_choices - 1) + (k - 1)
        X[i, idxs, torch.arange(seq_len)] = 0
        X[i, (idxs + k) % n_choices, torch.arange(seq_len)] = 1
    return X

def perturb_seqs(seqs):
    n_seqs, n_choices, seq_len = seqs.shape
    idxs = seqs.argmax(axis=1)
    n = seq_len * (n_choices - 1)
    X = np.tile(seqs, (n, 1, 1))
    X = X.reshape(n, n_seqs, n_choices, seq_len).transpose(1, 0, 2, 3)
    for i in range(n_seqs):
        for k in range(1, n_choices):
            idx = np.arange(seq_len) * (n_choices - 1) + (k - 1)

            X[i, idx, idxs[i], np.arange(seq_len)] = 0
            X[i, idx, (idxs[i] + k) % n_choices, np.arange(seq_len)] = 1
    return X

def perturb_seqs_torch(seqs):
    n_seqs, n_choices, seq_len = seqs.shape
    idxs = seqs.argmax(axis=1)
    n = seq_len * (n_choices - 1)
    X = torch.tile(seqs, (n, 1, 1))
    X = X.reshape(n, n_seqs, n_choices, seq_len).permute(1, 0, 2, 3)
    for i in range(n_seqs):
        for k in range(1, n_choices):
            idx = torch.arange(seq_len) * (n_choices - 1) + (k - 1)

            X[i, idx, idxs[i], torch.arange(seq_len)] = 0
            X[i, idx, (idxs[i] + k) % n_choices, torch.arange(seq_len)] = 1
    return X

def embed_pattern_seq(
    seq, 
    pattern, 
    position, 
    ohe=True,
    pattern_encoding="str",
    vocab="DNA",
):
    """
    Insert a pattern at a given position in a single sequence.
    """
    # If the sequence is one-hot encoded
    if ohe:
        # If the pattern is a string, we need to one-hot encode it
        if pattern_encoding == "str":
            pattern = ohe_seq(pattern, vocab=vocab)
        
        # If the pattern is one-hot encoded but has the wrong shape, we need to transpose it
        if pattern.shape[0] != seq.shape[0]:
            pattern = pattern.transpose()

        # Insert the pattern and return
        return np.concatenate((seq[:, :position], pattern, seq[:, position + pattern.shape[-1] :]), axis=1)
    
    # If the sequence is string encoded
    else:
        # If the pattern is one-hot encoded, we need to decode it
        if pattern_encoding == "ohe":
            if pattern.shape[0] != len(_get_vocab(vocab)):
                pattern = pattern.transpose()
            pattern = decode_seq(pattern, vocab=vocab)
        
        return seq[:position] + pattern + seq[position + len(pattern) :]
    
def embed_pattern_seqs(
    seqs,
    pattern,
    positions,
    ohe=True,
    pattern_encoding="str",
    vocab="DNA",
):
    """
    Insert a pattern at a given position in a batch of sequences.
    """
    # make a copy of the sequences
    embed_seqs = np.copy()
    if isinstance(positions, int):
        positions = [positions] * len(seqs)
    for i, seq in enumerate(seqs):
        embed_seqs[i] = embed_pattern_seq(seq, pattern, positions[i], ohe, pattern_encoding, vocab)
    return embed_seqs

def embed_patterns_seq(
    seq,
    patterns,
    positions,
    ohe=True,
    pattern_encoding="str",
    vocab="DNA",
):
    """
    Insert a batch of patterns at a batch of positions in a single sequence.
    """
    # make a copy of the sequences
    if isinstance(positions, int):
        positions = [positions] * len(patterns)
    embed_seq = seq.copy()
    for i, pattern in enumerate(patterns):
        embed_seq = embed_pattern_seq(embed_seq, pattern, positions[i], ohe, pattern_encoding, vocab)
    return embed_seq

def embed_patterns_seqs(
    seqs,
    patterns,
    positions,
    ohe=True,
    pattern_encoding="str",
    vocab="DNA",
):
    """
    Insert a batch of patterns at a batch of positions in a batch of sequences.
    """
    # make a copy of the sequences
    if isinstance(positions, int):
        positions = [positions] * len(patterns)
    embed_seqs = seqs.copy()
    for i, seq in enumerate(seqs):
        for j, pattern in enumerate(patterns):
            embed_seqs[i] = embed_pattern_seq(seq, pattern, positions[j], ohe, pattern_encoding, vocab)
    return embed_seqs

def tile_pattern_seq(seq, pattern, starting_pos=0, step=1, vocab="DNA", **kwargs):
    """
    Insert a pattern at every position for a single sequence.
    """
    if isinstance(seq, str):
        seq_len = len(seq)
    else:
        seq_len = seq.shape[-1]
    if isinstance(pattern, str):
        pattern_len = len(pattern)
    else:
        if pattern.shape[0] != len(_get_vocab(vocab)):
            pattern = pattern.transpose()
        pattern_len = pattern.shape[-1]
    implanted_seqs = []
    for pos in range(starting_pos, seq_len - pattern_len + 1, step):
        seq_implanted = embed_pattern_seq(seq, pattern, pos, **kwargs)
        implanted_seqs.append(seq_implanted)
    return np.array(implanted_seqs)

def find_patterns_seq(seq, patterns, pattern_names=None, starting_pos=0, check_rev_comp=True):
    """Function to find patterns and annotate the position and orientation of patterns in sequences
    
    Users should be able to specify an exact pattern or pass in pattern to search for.
    Can make use of the JASPAR database to find patterns. 
    """
    if isinstance(seq, np.ndarray):
        seq = decode_seq(seq, vocab="DNA") 

    if isinstance(patterns, dict):
        pattern_names = list(patterns.keys())
        patterns = list(patterns.values())
    
    if check_rev_comp:
        rev_patterns = list(reverse_complement_seqs(patterns, verbose=False))
        all_patterns = patterns + rev_patterns
        orientations = ["F"] * len(patterns) + ["R"] * len(rev_patterns)
        if pattern_names is not None:
            all_pattern_names = pattern_names + pattern_names
        else:
            all_pattern_names = [f"pattern{i}" for i in range(len(patterns))] + [f"pattern{i}" for i in range(len(patterns))]
    
    else:
        all_patterns =patterns 
        orientations = ["F"] * len(patterns)
        if pattern_names is not None:
            all_pattern_names = pattern_names
        else:
            all_pattern_names = [f"pattern{i}" for i in range(len(patterns))]
    
    longest_pattern = max(patterns, key=len)
    shortest_pattern = min(patterns, key=len)
    pattern_name_dict = dict(zip(all_patterns, all_pattern_names))
    pattern_orient_dict = dict(zip(all_patterns, orientations))
    pattern_hits_dict = {}
    for i in range(starting_pos, len(seq)-len(shortest_pattern)+1):
        for j in range(len(longest_pattern), len(shortest_pattern)-1, -1):
            pattern = seq[i:i+j]
            if pattern in all_patterns:
                pattern_hits_dict.setdefault(i, []).append(pattern_name_dict[pattern])
                pattern_hits_dict.setdefault(i, []).append(pattern_orient_dict[pattern])

    return pattern_hits_dict

def find_patterns_seqs(seqs, patterns, pattern_names=None, starting_pos=0, check_rev_comp=True):
    """Function to find patterns and annotate the position and orientation of patterns in sequences
    
    Users should be able to specify an exact pattern or pass in pattern to search for.
    Can make use of the JASPAR database to find patterns. 
    """
    if isinstance(seqs, np.ndarray):
        seqs = decode_seqs(seqs, vocab="DNA") 

    all_hits = []
    for seq in seqs:
        hits = find_patterns_seq(seq, patterns, pattern_names=pattern_names, starting_pos=starting_pos, check_rev_comp=check_rev_comp)
        all_hits.append(hits)
    return all_hits

