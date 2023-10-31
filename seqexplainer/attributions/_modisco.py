import os
import h5py
import numpy as np
import pandas as pd
from modiscolite.io import save_hdf5
from modiscolite.tfmodisco import TFMoDISco
from .._utils import _make_dirs, _path_to_image_html
from modiscolite.report import run_tomtom, create_modisco_logos, report_motifs

def _trim_cwm(
    cwm,
    trim_threshold=0.3
):
    """Trim the cwm to remove low scoring positions"""
    # Trim the cwm
    score = np.sum(np.abs(cwm), axis=1)
    trim_thresh = np.max(score) * trim_threshold
    pass_inds = np.where(score >= trim_thresh)[0]
    start, end = max(np.min(pass_inds) - 4, 0), min(np.max(pass_inds) + 4 + 1, len(score) + 1)
    trimmed_cwm = cwm[start:end]
    return trimmed_cwm

def modisco(
    one_hot,
    hypothetical_contribs,
    output_dir="./",
    output_name="modisco.h5",
    max_seqlets_per_metacluster=1000,
    min_metacluster_size=50,
    sliding_window_size=15,
    flank_size=5,
    target_seqlet_fdr=0.1,
    **kwargs,
):
    """Run TFMoDISco on a given set of hypothetical contributions."""
    if one_hot.shape[1] == 4:
        one_hot = one_hot.transpose(0, 2, 1)
    if hypothetical_contribs.shape[1] == 4:
        hypothetical_contribs = hypothetical_contribs.transpose(0, 2, 1)

    pos_patterns, neg_patterns = TFMoDISco(
        one_hot=one_hot,
        hypothetical_contribs=hypothetical_contribs,
        max_seqlets_per_metacluster=max_seqlets_per_metacluster,
        min_metacluster_size=min_metacluster_size,
        sliding_window_size=sliding_window_size,
        flank_size=flank_size,
        target_seqlet_fdr=target_seqlet_fdr,
        **kwargs,
    )
    _make_dirs(output_dir)
    save_hdf5(os.path.join(output_dir, output_name), pos_patterns, neg_patterns)
    return pos_patterns, neg_patterns

def modisco_logos(
    modisco_h5_file,
    output_dir="./",
    trim_threshold=0.3,
    **kwargs
):
    """Create logos from the motifs found by TFMoDISco."""
    _make_dirs(output_dir)
    names = create_modisco_logos(
        modisco_file=modisco_h5_file,
        modisco_logo_dir=output_dir,
        trim_threshold=trim_threshold,
    )
    return names

def modisco_tomtom(
    modisco_h5_file,
    meme_db_file,
    output_dir="./",
    top_n_matches=10,
    tomtom_exec="tomtom",
    trim_threshold=0.3,
    trim_min_length=3,
):
    """Run TomTom on the motifs found by TFMoDISco."""
    _make_dirs(output_dir)
    tomtom_df = run_tomtom(
        modisco_h5py=modisco_h5_file,
        output_prefix=output_dir,
        meme_motif_db=meme_db_file,
        top_n_matches=top_n_matches,
        tomtom_exec=tomtom_exec,
        trim_threshold=trim_threshold,
        trim_min_length=trim_min_length,
    )
    tomtom_df.to_csv(os.path.join(output_dir, "tomtom.tsv"), sep="\t", index=False)
    return tomtom_df

def modisco_report(
    modisco_h5_file,
    meme_db_file,
    output_dir="./",
    suffix="./",
    top_n_matches=10,
    trim_threshold=0.3,
    trim_min_length=3,
):
    """Create a report of the motifs found by TFMoDISco."""
    _make_dirs(output_dir)
    report_motifs(
        modisco_h5py=modisco_h5_file,
        output_dir=output_dir,
        meme_motif_db=meme_db_file,
        suffix=suffix,
        top_n_matches=top_n_matches,
        trim_threshold=trim_threshold,
        trim_min_length=trim_min_length,
    )
    return

def modisco_load_report(
    modisco_out,
    n_hits=3
):
    """Display the report of the motifs found by TFMoDISco."""
    report_df = pd.read_html(io=open(os.path.join(modisco_out, "motifs.html"), "r"))[0]
    report_df["modisco_cwm_fwd"] = [os.path.join(modisco_out, "trimmed_logos", pattern + ".cwm.fwd.png") for pattern in report_df["pattern"].values]
    report_df["modisco_cwm_rev"] = [os.path.join(modisco_out, "trimmed_logos", pattern + ".cwm.rev.png") for pattern in report_df["pattern"].values]
    tomtom_match_cols = report_df.columns[report_df.columns.str.contains("match\d_logo")]
    tomtom_match_cols = tomtom_match_cols[:n_hits]
    #get index of last match col within n_hits
    last_match_col = report_df.columns.get_loc(tomtom_match_cols[-1])
    # remove all columns after last match col
    report_df = report_df.iloc[:, :last_match_col + 2]
    for i, match_col in enumerate(tomtom_match_cols[:n_hits]):
        if report_df[f"match{i}"].isnull().all() or report_df[f"match{i}"].values[0] == "None":
            report_df[match_col] = ""
        else:
            report_df[match_col] = [os.path.join(modisco_out, pattern + ".png") for pattern in report_df[f"match{i}"].values]
    return report_df

def modisco_display_report(
    modisco_report_df,
):
    from IPython.core.display import HTML, display
    formatter_dict = {col: _path_to_image_html for col in modisco_report_df.columns if col.endswith("logo") or col.endswith("fwd") or col.endswith("rev")}
    display(HTML(modisco_report_df.to_html(escape=False, formatters=formatter_dict)))

def modisco_load_h5(
    modisco_h5_file,
    trim_threshold = 0.3
):
    """Load the motifs found by TFMoDISco."""

    results = h5py.File(modisco_h5_file, 'r')
    modisco_dict = {}
    for name in ["pos_patterns", "neg_patterns"]:
        if name not in results.keys():
            continue
        metacluster = results[name]
        key = lambda x: int(x[0].split("_")[-1])
        modisco_dict[name] = {}
        for pattern_name, pattern in sorted(metacluster.items(), key=key):
            modisco_dict[name][pattern_name] = {}

            ppm = pattern["sequence"][:]
            seqlet_set = pattern["seqlets"]["sequence"][:]
            modisco_dict[name][pattern_name]["ppm"] = ppm
            modisco_dict[name][pattern_name]["seqlet_set"] = seqlet_set
            
            cwm_fwd = pattern["contrib_scores"][:]
            cwm_rev = cwm_fwd[::-1, ::-1]
            modisco_dict[name][pattern_name]["cwm_fwd"] = cwm_fwd
            modisco_dict[name][pattern_name]["cwm_rev"] = cwm_rev
            
            trimmed_cwm_fwd = _trim_cwm(cwm_fwd, trim_threshold=trim_threshold)
            trimmed_cwm_rev = _trim_cwm(cwm_rev, trim_threshold=trim_threshold)
            modisco_dict[name][pattern_name]["trimmed_cwm_fwd"] = trimmed_cwm_fwd
            modisco_dict[name][pattern_name]["trimmed_cwm_rev"] = trimmed_cwm_rev  
    results.close()  
    return modisco_dict

def modisco_extract(
    pos_patterns,
    neg_patterns,
    trim_threshold = 0.3
):
    """Used to match output from modisco_load_h5 on results from running modisco."""
    modisco_dict = {}
    for name, patterns in zip(["pos_patterns", "neg_patterns"], [pos_patterns, neg_patterns]):
        modisco_dict[name] = {}
        for i, pattern in enumerate(patterns):
            pattern_name = f"pattern_{i}"
            modisco_dict[name][pattern_name] = {}

            ppm = pattern.sequence
            seqlet_set = np.array([seqlet.sequence for seqlet in pattern.seqlets])
            modisco_dict[name][pattern_name]["ppm"] = ppm
            modisco_dict[name][pattern_name]["seqlet_set"] = seqlet_set
            
            cwm_fwd = pattern.contrib_scores
            cwm_rev = cwm_fwd[::-1, ::-1]
            modisco_dict[name][pattern_name]["cwm_fwd"] = cwm_fwd
            modisco_dict[name][pattern_name]["cwm_rev"] = cwm_rev
            
            trimmed_cwm_fwd = _trim_cwm(cwm_fwd, trim_threshold=trim_threshold)
            trimmed_cwm_rev = _trim_cwm(cwm_rev, trim_threshold=trim_threshold)
            modisco_dict[name][pattern_name]["trimmed_cwm_fwd"] = trimmed_cwm_fwd
            modisco_dict[name][pattern_name]["trimmed_cwm_rev"] = trimmed_cwm_rev
    return modisco_dict
 