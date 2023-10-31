import numpy as np
from ._perturb import embed_pattern_seqs, tile_pattern_seqs

def embed_deepstarr_distance_cooperativity(
    null_sequences,
    motif_a,
    motif_b,
    step=1,
    allow_overlap=False
):
    """
    Generates sequences for a motif cooperativity analysis similar to that performed in the DeepSTARR paper.
    
    Embeds two motifs in a set of provided null sequences. The first motif (MotifA) is embedded in the center of each sequence,
    while the second motif (MotifB) is tiled across the sequence at a range of distances from MotifA, both upstream and downstream. 
    
    This function returns three sets of sequences: 
        (1) the null sequences with only MotifA in the center
        (2) the null sequences with only MotifB at a range of distances from the center
        (3) the null sequences with both MotifA and MotifB at a range of distances from the center
        
    The generated sequences can then be fed to a model to determine the effect of the distance between the motifs on the model's predictions.

    A few notes for this function. Using step sizes other than 1 will result in different behaviors depending on the length of the motifs and the
    null sequences. If null sequences are odd length and motif a is also odd, motif a can be inserted symetrically in the center of the sequence. If null sequences
    are even length and motif a is odd, motif a will be inserted one position to the left of center.

    Parameters
    ----------
    null_sequences : np.array
        A numpy array of sequences to embed the motifs in. The sequences should not be one-hot encoded. Must also be longer than the motifs.
    motif_a : str
        The sequence of the first motif to embed in the center of the sequences.
    motif_b : str
        The sequence of the second motif to tile across the sequences.
    step : int, optional
        The step size for tiling MotifB across the sequences. The default is 1.
    allow_overlap : bool, optional
        Whether to allow MotifB to overlap with MotifA. The default is False.

    Returns
    -------
    A_seqs : np.array
        The null sequences with only MotifA in the center.
    B_seqs : np.array
        The null sequences with only MotifB at a range of distances from the center.
    AB_seqs : np.array
        The null sequences with both MotifA and MotifB at a range of distances from the center.
    motif_b_pos : np.array
        The positions that MotifB was tiled across the sequences.
    motif_b_distances : np.array
        The distances between MotifA and MotifB for each sequence. A '+' indicates that MotifB is downstream of MotifA, while a '-' indicates that MotifB is upstream of MotifA.
    
    Examples
    --------
    Coming soon...
    """

    # Get the length of the motifs and sequences
    motif_a_len = len(motif_a)
    motif_b_len = len(motif_b)
    seq_len = len(null_sequences[0])
    
    # Grab the middle of the sequence and offset by the motif length so that the motif is centered if possible
    motif_a_start = int(np.floor(seq_len/2) - np.ceil(motif_a_len/2))
    
    # Embed motif A in the sequence at the center
    A_seqs = embed_pattern_seqs(
        seqs=null_sequences,
        pattern=motif_a,
        positions=motif_a_start,
        ohe=False
    )

    # Get the positions that motif B will get tiled across
    motif_b_pos = np.arange(0, seq_len - motif_b_len + 1, step=step)
    
    # Remove any positions that overlap with motif A
    if not allow_overlap:
        
        # Remove any positions that overlap with motif A
        motif_b_pos = motif_b_pos[~((motif_b_pos >= (motif_a_start - motif_b_len)) & (motif_b_pos <= (motif_a_start + motif_a_len)))]
        
        # Tile B across the background
        B_seqs = []
        for pos in motif_b_pos:
            curr_B_seqs = embed_pattern_seqs(
                seqs=null_sequences,
                pattern=motif_b,
                positions=int(pos),
                ohe=False
            )
            B_seqs.append(curr_B_seqs)
        B_seqs = np.array(B_seqs)

        # Tile B across the A sequences
        AB_seqs = []
        for pos in motif_b_pos:
            curr_AB_seqs = embed_pattern_seqs(
                seqs=A_seqs,
                pattern=motif_b,
                positions=int(pos),
                ohe=False
            )
            AB_seqs.append(curr_AB_seqs)
        AB_seqs = np.array(AB_seqs)

    # Allow overlap
    else:
        
        # Tile the motif B across the sequence
        B_seqs = tile_pattern_seqs(
            seqs=null_sequences,
            pattern=motif_b,
            ohe=False,
            step=step
        )

        # Tile B across the A sequences
        AB_seqs = tile_pattern_seqs(
            seqs=A_seqs,
            pattern=motif_b,
            ohe=False,
            step=step
        )

    # Grab distances, turn into a string and add + for positive and - for negative
    motif_b_distances = (motif_b_pos - motif_a_start).astype(str)
    motif_b_distances = np.array(["+" + dist if int(dist) >= 0 else dist for dist in motif_b_distances])
    return A_seqs, B_seqs, AB_seqs, motif_b_pos, motif_b_distances