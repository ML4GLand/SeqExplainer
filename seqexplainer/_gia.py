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