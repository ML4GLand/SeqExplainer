import numpy as np
import torch
from ..preprocess._preprocess import decode_seq, decode_seqs, ohe_seq, reverse_complement_seqs
from ..preprocess._helpers import _get_vocab


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

    Parameters
    ----------
    seq : str or np.ndarray
        The sequence to embed the pattern in. Can by a single string or a one-hot encoded numpy array.
    pattern : str or np.ndarray
        The pattern to embed. Can by a single string or a one-hot encoded numpy array.
    position : int
        The position to embed the pattern at. One position only for this function!
    ohe : bool, optional
        Whether the input seq is one-hot encoded or not, by default True
    pattern_encoding : str, optional
        Whether the pattern is one-hot encoded or not, by default "str"

    Returns
    -------
    np.ndarray
        The original sequence with the pattern embedded at the given position.
    """
    # If the sequence is one-hot encoded
    if ohe:
        embed_seq = seq.copy()
        # If the pattern is a string, we need to one-hot encode it
        if pattern_encoding == "str":
            pattern = ohe_seq(pattern, vocab=vocab)
        
        # If the pattern is one-hot encoded but has the wrong shape, we need to transpose it
        if pattern.shape[0] != embed_seq.shape[0]:
            pattern = pattern.transpose()

        # Insert the pattern and return
        return np.concatenate((embed_seq[:, :position], pattern, embed_seq[:, position + pattern.shape[-1] :]), axis=1)
    
    # If the sequence is string encoded
    else:
        embed_seq = seq + ""
        # If the pattern is one-hot encoded, we need to decode it
        if pattern_encoding == "ohe":
            if pattern.shape[0] != len(_get_vocab(vocab)):
                pattern = pattern.transpose()
            pattern = decode_seq(pattern, vocab=vocab)
        
        return embed_seq[:position] + pattern + embed_seq[position + len(pattern) :]
    
def embed_pattern_seqs(
    seqs,
    pattern,
    positions,
    ohe=True,
    pattern_encoding="str",
    vocab="DNA",
):
    """
    Insert a single pattern at same or different positions in a batch of sequences.
    If a single position is given, the pattern will be inserted at the same position in all sequences.
    If a list of positions is given, the pattern will be inserted at the corresponding position in each sequence.

    Parameters
    ----------
    seqs : list of str or np.ndarray
        The sequences to embed the pattern in. Can by a list of strings or a numpy array of one-hot encoded sequences.
    pattern : str or np.ndarray
        The pattern to embed. Can by a single string or a one-hot encoded numpy array.
    positions : int or list of int
        The position to embed the pattern at. One position for all sequences or one position for each sequence.
    ohe : bool, optional
        Whether the input seqs are one-hot encoded or not, by default True
    pattern_encoding : str, optional
        Whether the pattern is one-hot encoded or not, by default "str"

    Returns
    -------
    np.ndarray
        The original sequences with the pattern embedded at the given positions.
    """
    embed_seqs = seqs.copy()
    
    # If given a single position, use it for all sequences
    if isinstance(positions, int):
        positions = [[positions]] * len(seqs)
    
    # If given a list of positions, use them for all sequences 
    elif isinstance(positions, list):
        positions = positions * len(seqs)
    
    # Else raise an error
    elif len(positions) != len(seqs):
        raise ValueError("The number of passed in positions must match the number of sequences.")
    
    positions = np.array(positions)
    for i in range(len(positions)):
        #print(len(positions[i]))
        for j in range(len(positions[i])):
            if positions[i][j] < 0:
                continue
            embed_seqs[i] = embed_pattern_seq(seqs[i], pattern, positions[i][j], ohe, pattern_encoding, vocab)
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

    Parameters
    ----------
    seq : str or np.ndarray
        The sequence to embed the patterns in. Can by a single string or a one-hot encoded numpy array.
    patterns : list of str or np.ndarray
        The patterns to embed. Can by a list of strings or a numpy array of one-hot encoded patterns.
    positions : list of int
        The positions to embed the patterns at. Must be one position for each pattern.
    ohe : bool, optional
        Whether the input seq is one-hot encoded or not, by default True
    pattern_encoding : str, optional
        Whether the patterns are one-hot encoded or not, by default "str"
    
    Returns
    -------
    np.ndarray
        The original sequence with the patterns embedded at the given positions.
    """
    # make a copy of the sequences
    if len(patterns) != len(positions):
        raise ValueError("The number of patterns and positions must be the same.")
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
    If you want to insert the patterns in the same position in all sequences, patterns should be a list 
    that matches positions in length
    If you want to insert the patterns in different positions in all sequences, patterns and positions 
    should be list of lists (or 2D array). The length should match the number of sequences, and the width 
    should match the number of patterns to insert in each sequence.
    
    Parameters
    ----------
    seqs : list of str or np.ndarray
        A list of sequences to embed the patterns in.
    patterns : list of str or np.ndarray
        A list of patterns to embed in the sequences. The first dimension corresponds to the sequences, the second to the patterns.
    positions : list of list of int
        A list of lists of positions to embed the patterns at. The first dimension corresponds to the sequences, the second to the patterns.
    ohe : bool, optional
        Whether the sequences are one-hot encoded, by default True
    pattern_encoding : str, optional
        Whether the patterns are one-hot encoded or string encoded, by default "str"
    vocab : str, optional
        The vocabulary to use for one-hot encoding, by default "DNA"
    """
    # make a copy of the sequences
    embed_seqs = seqs.copy()

    # If the positions are a single list, it needs to match the number of patterns and be tiled across the sequences
    if isinstance(positions[0], int):
        if len(positions) == len(patterns) != len(seqs):
            positions = [positions] * len(seqs)
            patterns = [patterns] * len(seqs)
        else:
            raise ValueError("The number of positions must match the number of patterns if using same positions across sequences.")
 
    # If positions are a list of lists, we need to check that dimensions match the number of sequences and patterns
    elif isinstance(positions[0], list):
        if len(positions) != len(seqs) or len(patterns) != len(seqs):
            raise ValueError("The number of sequences must match the number of lists of positions.")
        if len(positions[0]) != len(patterns[0]):
            raise ValueError("The number of patterns must match the number of positions in each list of positions.")                

    positions = np.array(positions)
    patterns = np.array(patterns)
    for i in range(len(positions)):
        for j in range(len(positions[i])):
            if positions[i][j] < 0:
                continue
            embed_seqs[i] = embed_pattern_seq(embed_seqs[i], patterns[i][j], positions[i][j], ohe, pattern_encoding, vocab)
    return embed_seqs

def tile_pattern_seq(
    seq, 
    pattern, 
    starting_pos=0, 
    step=1, 
    vocab="DNA", 
    **kwargs
):
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

def tile_pattern_seqs(
    seqs,
    pattern,
    starting_pos=0,
    step=1,
    vocab="DNA",
    **kwargs
):
    """
    Insert a pattern at every position for a batch of sequences.
    """

    implanted_seqs = []
    for seq in seqs:
        implanted_seqs.append(tile_pattern_seq(seq, pattern, starting_pos, step, vocab, **kwargs))
    return np.array(implanted_seqs)
        
def find_patterns_seq(
    seq, 
    patterns, 
    pattern_names=None, 
    starting_pos=0, 
    check_rev_comp=True
):
    """Function to find patterns and annotate the position and orientation of patterns in sequences
    
    Users should be able to specify an exact pattern or pass in pattern to search for.
    Can make use of the JASPAR database to find patterns. 
    """
    if isinstance(seq, np.ndarray):
        seq = decode_seq(seq, vocab="DNA") 

    if isinstance(patterns, dict):
        pattern_names = list(patterns.keys())
        patterns = list(patterns.values())

    if isinstance(patterns, str):
        patterns = [patterns]
    
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
                pattern_hits_dict.setdefault(i, []).append(pattern)
    return pattern_hits_dict

def find_patterns_seqs(
    seqs, 
    patterns, 
    pattern_names=None, 
    starting_pos=0, 
    check_rev_comp=True
):
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

def occlude_patterns_seq(
    seq,
    patterns,
    occlusion_pattern=None,
    starting_pos=0,
    check_rev_comp=True,
    max_occluded=None,
    max_iters=1000,
):
    # make a copy of the sequence
    embed_seq = seq.copy()
    it = 0

    #while there are still motif hits
    hits = find_patterns_seq(seq, patterns, starting_pos=starting_pos, check_rev_comp=check_rev_comp)
    while len(hits) > 0 and it < max_iters:
        if max_occluded is None:
            max_occluded = len(hits)
        
        # occlude motifs in seq
        for pos, motif in hits.items():    
            # Get random pattern of same size as pattern
            if occlusion_pattern is None:
                occlusion_pattern = ohe_seq("".join(np.random.choice(["A", "C", "G", "T"], size=len(motif[-1]))))
            embed_seq = embed_pattern_seq(embed_seq, occlusion_pattern, pos, ohe=True, pattern_encoding="ohe")

        # check if there are still motif hits
        hits = find_patterns_seq(seq, patterns, starting_pos=starting_pos, check_rev_comp=check_rev_comp)
        it += 1
        return embed_seq

def occlude_patterns_seqs(
    seqs,
    patterns,
    occlusion_pattern=None,
    starting_pos=0,
    check_rev_comp=True,
    max_occluded=None,
    max_iters=1000,
):
    embed_seqs = seqs.copy()
    it = 0
    # find motifs in seq
    hits = find_patterns_seqs(embed_seqs, patterns, starting_pos=starting_pos, check_rev_comp=check_rev_comp)
    # get number of sum of total hits across all sequences
    total_hits = 0
    for hit in hits:
        total_hits += len(hit)
    while total_hits > 0 and it < max_iters:
        if max_occluded is None:
            max_occluded = len(hits)
        
        # occlude motifs in seq
        for i, hit in enumerate(hits):
            for pos, motif in hit.items():
                # Get random pattern of same size as pattern
                if occlusion_pattern is None:
                    occlusion_pattern = ohe_seq("".join(np.random.choice(["A", "C", "G", "T"], size=len(motif[-1]))))
                embed_seqs[i] = embed_pattern_seq(embed_seqs[i], occlusion_pattern, pos, ohe=True, pattern_encoding="ohe")
        hits = find_patterns_seqs(embed_seqs, patterns, starting_pos=starting_pos, check_rev_comp=check_rev_comp)
        total_hits = 0
        for hit in hits:
            total_hits += len(hit)
        it += 1
    return embed_seqs
