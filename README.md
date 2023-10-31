[![PyPI version](https://badge.fury.io/py/seqexplainer.svg)](https://badge.fury.io/py/seqexplainer)
![PyPI - Downloads](https://img.shields.io/pypi/dm/seqexplainer)

<img src="docs/_static/SeqExplainer_logo.png" alt="SeqExplainer Logo" width=350>

# SeqExplainer (Sequence explainability tools)
SeqExplainer is a Python package for interpreting sequence-to-function machine learning models. Most of the core functionality is for post-hoc analysis of a trained model. SeqExplainer currently supports:

- [Filter interpretation](https://github.com/ML4GLand/tutorials/blob/main/seqexplainer/filter_interpretation.ipynb)
- [Attribution analysis](https://github.com/ML4GLand/tutorials/blob/main/seqexplainer/attribution_analysis.ipynb)
- [Sequence evolution](https://github.com/ML4GLand/tutorials/blob/main/seqexplainer/sequence_evolution.ipynb)
- [In silico experiments with a trained model (aka GIA)](https://github.com/ML4GLand/use_cases/blob/main/DeepSTARR/evoaug/distance_cooperativity_analysis.ipynb)


# Requirements

The main dependencies of SeqExplainer are:

```bash
python
torch
captum
numpy
matplotlib
logomaker
sklearn
shap
```

# Contributing
This section was modified from https://github.com/pachterlab/kallisto.

All contributions, including bug reports, documentation improvements, and enhancement suggestions are welcome. Everyone within the community is expected to abide by our [code of conduct](https://github.com/ML4GLand/EUGENe/blob/main/CODE_OF_CONDUCT.md)

As we work towards a stable v1.0.0 release, and we typically develop on branches. These are merged into `dev` once sufficiently tested. `dev` is the latest, stable, development branch. 

`main` is used only for official releases and is considered to be stable. If you submit a pull request, please make sure to request to merge into `dev` and NOT `main`.

# References
1. Novakovsky, G., Dexter, N., Libbrecht, M. W., Wasserman, W. W. & Mostafavi, S. Obtaining genetics insights from deep learning via explainable artificial intelligence. Nat. Rev. Genet. 1–13 (2022) doi:10.1038/s41576-022-00532-2
