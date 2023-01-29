import torch
import numpy as np
from tqdm.auto import tqdm
np.random.seed(13)

# Vocabularies
DNA = ["A", "C", "G", "T"]
RNA = ["A", "C", "G", "U"]
COMPLEMENT_DNA = {"A": "T", "C": "G", "G": "C", "T": "A"}
COMPLEMENT_RNA = {"A": "U", "C": "G", "G": "C", "U": "A"}

def _get_vocab(vocab):
    if vocab == "DNA":
        return DNA
    elif vocab == "RNA":
        return RNA
    else:
        raise ValueError("Invalid vocab, only DNA or RNA are currently supported")

# exact concise
def _get_vocab_dict(vocab):
    """
    Returns a dictionary mapping each token to its index in the vocabulary.
    Used in `_tokenize`.
    """
    return {l: i for i, l in enumerate(vocab)}

# exact concise
def _get_index_dict(vocab):
    """
    Returns a dictionary mapping each token to its index in the vocabulary.
    """
    return {i: l for i, l in enumerate(vocab)}

# modified concise
def _tokenize(seq, vocab="DNA", neutral_vocab=["N"]):
    """
    Convert sequence to integers based on a vocab

    Parameters
    ----------
    seq: 
        sequence to encode
    vocab: 
        vocabulary to use
    neutral_vocab: 
        neutral vocabulary -> assign those values to -1
    
    Returns
    -------
        List of length `len(seq)` with integers from `-1` to `len(vocab) - 1`
    """
    vocab = _get_vocab(vocab)
    if isinstance(neutral_vocab, str):
        neutral_vocab = [neutral_vocab]

    nchar = len(vocab[0])
    for l in vocab + neutral_vocab:
        assert len(l) == nchar
    assert len(seq) % nchar == 0  # since we are using striding

    vocab_dict = _get_vocab_dict(vocab)
    for l in neutral_vocab:
        vocab_dict[l] = -1

    # current performance bottleneck
    return [
        vocab_dict[seq[(i * nchar) : ((i + 1) * nchar)]]
        for i in range(len(seq) // nchar)
    ]

# my own
def _sequencize(tvec, vocab="DNA", neutral_value=-1, neutral_char="N"):
    """
    Converts a token vector into a sequence of symbols of a vocab.
    """
    vocab = _get_vocab(vocab) 
    index_dict = _get_index_dict(vocab)
    index_dict[neutral_value] = neutral_char
    return "".join([index_dict[i] for i in tvec])

# modified concise
def _token2one_hot(tvec, vocab="DNA", fill_value=None):
    """
    Converts an L-vector of integers in the range [0, D] into an L x D one-hot
    encoding. If fill_value is not None, then the one-hot encoding is filled
    with this value instead of 0.

    Parameters
    ----------
    tvec : np.array
        L-vector of integers in the range [0, D]
    vocab_size : int
        D
    fill_value : float, optional
        Value to fill the one-hot encoding with. If None, then the one-hot
    """
    vocab = _get_vocab(vocab)
    vocab_size = len(vocab)
    arr = np.zeros((vocab_size, len(tvec)))
    tvec_range = np.arange(len(tvec))
    tvec = np.asarray(tvec)
    arr[tvec[tvec >= 0], tvec_range[tvec >= 0]] = 1
    if fill_value is not None:
        arr[:, tvec_range[tvec < 0]] = fill_value
    return arr.astype(np.int8) if fill_value is None else arr.astype(np.float16)

# modified dinuc_shuffle
def _one_hot2token(one_hot, neutral_value=-1, consensus=False):
    """
    Converts a one-hot encoding into a vector of integers in the range [0, D]
    where D is the number of classes in the one-hot encoding.

    Parameters
    ----------
    one_hot : np.array
        L x D one-hot encoding
    neutral_value : int, optional
        Value to use for neutral values.
    
    Returns
    -------
    np.array
        L-vector of integers in the range [0, D]
    """
    if consensus:
        return np.argmax(one_hot, axis=0)
    tokens = np.tile(neutral_value, one_hot.shape[1])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot.transpose()==1)
    tokens[seq_inds] = dim_inds
    return tokens

# pad and subset, exact concise
def _pad(seq, max_seq_len, value="N", align="end"):
    seq_len = len(seq)
    assert max_seq_len >= seq_len
    if align == "end":
        n_left = max_seq_len - seq_len
        n_right = 0
    elif align == "start":
        n_right = max_seq_len - seq_len
        n_left = 0
    elif align == "center":
        n_left = (max_seq_len - seq_len) // 2 + (max_seq_len - seq_len) % 2
        n_right = (max_seq_len - seq_len) // 2
    else:
        raise ValueError("align can be of: end, start or center")

    # normalize for the length
    n_left = n_left // len(value)
    n_right = n_right // len(value)

    return value * n_left + seq + value * n_right

# exact concise
def _trim(seq, maxlen, align="end"):
    seq_len = len(seq)

    assert maxlen <= seq_len
    if align == "end":
        return seq[-maxlen:]
    elif align == "start":
        return seq[0:maxlen]
    elif align == "center":
        dl = seq_len - maxlen
        n_left = dl // 2 + dl % 2
        n_right = seq_len - dl // 2
        return seq[n_left:n_right]
    else:
        raise ValueError("align can be of: end, start or center")

# modified concise
def _pad_sequences(
    seqs, 
    maxlen=None, 
    align="end", 
    value="N"
):
    """
    Pads sequences to the same length.

    Parameters
    ----------
    seqs : list of str
        Sequences to pad
    maxlen : int, optional
        Length to pad to. If None, then pad to the length of the longest sequence.
    align : str, optional
        Alignment of the sequences. One of "start", "end", "center"
    value : str, optional
        Value to pad with

    Returns
    -------
    np.array
        Array of padded sequences
    """

    # neutral element type checking
    assert isinstance(value, list) or isinstance(value, str)
    assert isinstance(value, type(seqs[0])) or type(seqs[0]) is np.str_
    assert not isinstance(seqs, str)
    assert isinstance(seqs[0], list) or isinstance(seqs[0], str)

    max_seq_len = max([len(seq) for seq in seqs])

    if maxlen is None:
        maxlen = max_seq_len
    else:
        maxlen = int(maxlen)

    if max_seq_len < maxlen:
        import warnings
        warnings.warn(
            f"Maximum sequence length ({max_seq_len}) is smaller than maxlen ({maxlen})."
        )
        max_seq_len = maxlen

    # check the case when len > 1
    for seq in seqs:
        if not len(seq) % len(value) == 0:
            raise ValueError("All sequences need to be dividable by len(value)")
    if not maxlen % len(value) == 0:
        raise ValueError("maxlen needs to be dividable by len(value)")

    padded_seqs = [
        _trim(_pad(seq, max(max_seq_len, maxlen), value=value, align=align), maxlen, align=align)
        for seq in seqs 
    ]
    return padded_seqs

def _is_overlapping(a, b):
    """Returns True if two intervals overlap"""
    if b[0] >= a[0] and b[0] <= a[1]:
        return True
    else:
        return False

def _merge_intervals(intervals):
    """Merges a list of overlapping intervals"""
    if len(intervals) == 0:
        return None
    merged_list = []
    merged_list.append(intervals[0])
    for i in range(1, len(intervals)):
        pop_element = merged_list.pop()
        if _is_overlapping(pop_element, intervals[i]):
            new_element = pop_element[0], max(pop_element[1], intervals[i][1])
            merged_list.append(new_element)
        else:
            merged_list.append(pop_element)
            merged_list.append(intervals[i])
    return merged_list

def _hamming_distance(string1, string2):
    """Find hamming distance between two strings. Returns inf if they are different lengths"""
    distance = 0
    L = len(string1)
    if L != len(string2):
        return np.inf
    for i in range(L):
        if string1[i] != string2[i]:
            distance += 1
    return distance

def _collapse_pos(positions):
    """Collapse neighbor positions of array to ranges"""
    ranges = []
    start = positions[0]
    for i in range(1, len(positions)):
        if positions[i - 1] == positions[i] - 1:
            continue
        else:
            ranges.append((start, positions[i - 1] + 2))
            start = positions[i]
    ranges.append((start, positions[-1] + 2))
    return ranges

# next 4 are from dinuc shuffle in DeepLift package
def _string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)

def _char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tostring().decode("ascii")

def _one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens

def _tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]

def remove_only_N_seqs(seqs):
    return [seq for seq in seqs if not all([x == "N" for x in seq])]  

def sanitize_seq(seq):
    """Capitalizes and removes whitespace for single seq."""
    return seq.strip().upper()

def sanitize_seqs(seqs):
    """Capitalizes and removes whitespace for a set of sequences."""
    return np.array([seq.strip().upper() for seq in seqs])

def ascii_encode_seq(seq, pad=0):
    """
    Converts a string of characters to a NumPy array of byte-long ASCII codes.
    """
    encode_seq = np.array([ord(letter) for letter in seq], dtype=int)
    if pad > 0:
        encode_seq = np.pad(encode_seq, pad_width=(0, pad), mode="constant", constant_values=36)
    return encode_seq

def ascii_encode_seqs(seqs, pad=0):
    """
    Converts a set of sequences to a NumPy array of byte-long ASCII codes.
    """
    encode_seqs = np.array(
        [ascii_encode_seq(seq, pad=pad) for seq in seqs], dtype=int
    )
    return encode_seqs

def ascii_decode_seq(seq):
    """
    Converts a NumPy array of byte-long ASCII codes to a string of characters.
    """
    return "".join([chr(int(letter)) for letter in seq]).replace("$", "")

def ascii_decode_seqs(seqs):
    """Convert a set of one-hot encoded arrays back to strings"""
    return np.array([ascii_decode_seq(seq) for seq in seqs], dtype=object)

def reverse_complement_seq(seq, vocab="DNA"):
    """Reverse complement a single sequence."""
    if isinstance(seq, str):
        if vocab == "DNA":
            return "".join(COMPLEMENT_DNA.get(base, base) for base in reversed(seq))
        elif vocab == "RNA":
            return "".join(COMPLEMENT_RNA.get(base, base) for base in reversed(seq))
        else:
            raise ValueError("Invalid vocab, only DNA or RNA are currently supported")
    elif isinstance(seq, np.ndarray):
        return torch.from_numpy(np.flip(seq, axis=(0, 1)).copy()).numpy()

def reverse_complement_seqs(seqs, vocab="DNA", verbose=True):
    """Reverse complement set of sequences."""
    if isinstance(seqs[0], str):
        return np.array(
            [
                reverse_complement_seq(seq, vocab)
                for i, seq in tqdm(
                    enumerate(seqs),
                    total=len(seqs),
                    desc="Reverse complementing sequences",
                    disable=not verbose,
                )
            ]
        )
    elif isinstance(seqs[0], np.ndarray):
        return torch.from_numpy(np.flip(seqs, axis=(1, 2)).copy()).numpy()

def gc_content_seq(seq, ohe=False):
    if ohe:
        return np.sum(seq[1:3, :])/seq.shape[1]
    else:
        return (seq.count("G") + seq.count("C"))/len(seq)
    
def gc_content_seqs(seqs, ohe=False):
    if ohe:
        seq_len = seqs[0].shape[1]
        return np.sum(seqs[:, 1:3, :], axis=1).sum(axis=1)/seq_len
    else:
        return np.array([gc_content_seq(seq) for seq in seqs])

def ohe_seq(
    seq, 
    vocab="DNA", 
    neutral_vocab="N", 
    fill_value=0
):
    """Convert a sequence into one-hot-encoded array."""
    seq = seq.strip().upper()
    return _token2one_hot(_tokenize(seq, vocab, neutral_vocab), vocab, fill_value=fill_value)

def ohe_seqs(
    seqs,
    vocab="DNA",
    neutral_vocab="N",
    maxlen=None,
    pad=True,
    pad_value="N",
    fill_value=None,
    seq_align="start",
    verbose=True,
):
    """Convert a set of sequences into one-hot-encoded array."""
    if isinstance(neutral_vocab, str):
        neutral_vocab = [neutral_vocab]
    if isinstance(seqs, str):
        raise ValueError("seq_vec should be an iterable not a string itself")
    assert len(vocab[0]) == len(pad_value)
    assert pad_value in neutral_vocab
    if pad:
        seqs_vec = _pad_sequences(seqs, maxlen=maxlen, align=seq_align, value=pad_value)
    arr_list = [
        ohe_seq(
            seq=seqs_vec[i],
            vocab=vocab,
            neutral_vocab=neutral_vocab,
            fill_value=fill_value,
        )
        for i in tqdm(
            range(len(seqs_vec)),
            total=len(seqs_vec),
            desc="One-hot encoding sequences",
            disable=not verbose,
        )
    ]
    if pad:
        return np.stack(arr_list)
    else:
        return np.array(arr_list, dtype=object)

def decode_seq(arr, vocab="DNA", neutral_value=-1, neutral_char="N"):
    """Convert a single one-hot encoded array back to string"""
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    return _sequencize(
        tvec=_one_hot2token(arr, neutral_value),
        vocab=vocab,
        neutral_value=neutral_value,
        neutral_char=neutral_char,
    )

def decode_seqs(arr, vocab="DNA", neutral_char="N", neutral_value=-1, verbose=True):
    """Convert a one-hot encoded array back to set of sequences"""
    arr_list = [
        decode_seq(
            arr=arr[i],
            vocab=vocab,
            neutral_value=neutral_value,
            neutral_char=neutral_char,
        )
        for i in tqdm(
            range(len(arr)),
            total=len(arr),
            desc="Decoding sequences",
            disable=not verbose,
        )
    ]
    return np.array(arr_list)

def shuffle_seq():
    #TODO
    pass

def shuffle_seqs():
    #TODO
    pass

def dinuc_shuffle_seq(
    seq, 
    num_shufs=None, 
    rng=None
):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.

    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D np array, then the
    result is an N x L x D np array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).

    Parameters
    ----------
    seq : str
        The sequence to shuffle.
    num_shufs : int, optional
        The number of shuffles to create. If None, only one shuffle is created.
    rng : np.random.RandomState, optional
        The random number generator to use. If None, a new one is created.

    Returns
    -------
    list of str or np.array
        The shuffled sequences.

    Note
    ----
    This function comes from DeepLIFT's dinuc_shuffle.py.
    """
    if type(seq) is str or type(seq) is np.str_:
        arr = _string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = _one_hot2token(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")
    if not rng:
        rng = np.random.RandomState(rng)

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    if type(seq) is str or type(seq) is np.str_:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim), dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str or type(seq) is np.str_:
            all_results.append(_char_array_to_string(chars[result]))
        else:
            all_results[i] = _token2one_hot(chars[result])
    return all_results if num_shufs else all_results[0]

def dinuc_shuffle_seqs(seqs, num_shufs=None, rng=None):
    """
    Shuffle the sequences in `seqs` in the same way as `dinuc_shuffle_seq`.
    If `num_shufs` is not specified, then the first dimension of N will not be
    present (i.e. a single string will be returned, or an L x D array).

    Parameters
    ----------
    seqs : np.ndarray
        Array of sequences to shuffle
    num_shufs : int, optional
        Number of shuffles to create, by default None
    rng : np.random.RandomState, optional
        Random state to use for shuffling, by default None

    Returns
    -------
    np.ndarray
        Array of shuffled sequences

    Note
    -------
    This is taken from DeepLIFT
    """
    if not rng:
        rng = np.random.RandomState(rng)

    if type(seqs) is str or type(seqs) is np.str_:
        seqs = [seqs]

    all_results = []
    for i in range(len(seqs)):
        all_results.append(dinuc_shuffle_seq(seqs[i], num_shufs=num_shufs, rng=rng))
    return np.array(all_results)

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

def feature_implant_seq(
    seq, feature, position, vocab="DNA", encoding="str", onehot=False
):
    """
    Insert a feature at a given position in a single sequence.
    """
    if encoding == "str":
        return seq[:position] + feature + seq[position + len(feature) :]
    elif encoding == "onehot":
        if onehot:
            feature = _token2one_hot(feature.argmax(axis=1), vocab=vocab, fill_value=0)
        if feature.shape[0] != seq.shape[0]:
            feature = feature.transpose()
        return np.concatenate(
            (seq[:, :position], feature, seq[:, position + feature.shape[-1] :]), axis=1
        )
    else:
        raise ValueError("Encoding not recognized.")

def feature_implant_across_seq(seq, feature, **kwargs):
    """
    Insert a feature at every position for a single sequence.
    """
    if isinstance(seq, str):
        assert isinstance(feature, str)
        seq_len = len(seq)
        feature_len = len(feature)
    elif isinstance(seq, np.ndarray):
        assert isinstance(feature, np.ndarray)
        seq_len = seq.shape[-1]
        if feature.shape[0] != seq.shape[0]:
            feature_len = feature.shape[0]
        else:
            feature_len = feature.shape[-1]
    implanted_seqs = []
    for pos in range(seq_len - feature_len + 1):
        seq_implanted = feature_implant_seq(seq, feature, pos, **kwargs)
        implanted_seqs.append(seq_implanted)
    return np.array(implanted_seqs)
