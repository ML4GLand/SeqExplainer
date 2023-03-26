[![PyPI version](https://badge.fury.io/py/seqexplainer.svg)](https://badge.fury.io/py/seqexplainer)
![PyPI - Downloads](https://img.shields.io/pypi/dm/seqexplainer)

<img src="docs/_static/SeqExplainer_logo.png" alt="SeqExplainer Logo" width=350>

# SeqExplainer -- Interpreting sequence-to-function machine learning models

A huge goal of applying machine learniing to genomics data is to [obtain novel geneetic insights [1]](https://www.nature.com/articles/s41576-022-00532-2). This can be challenging when models are complex (such as with neural networks). There are [many interpretability methods](https://github.com/ML4GLand/awesome-dl4g/blob/main/README.md#interpretability) specifically designed for such complex sequence-to-function preditctors, but they can be difficult to use and often are not interoperable.

The goal of SeqExplainer is to bring all these methods under one roof. We have designed a workflow that can take in any PyTorch model trained to predict labels from DNA sequence and expose it to many of the most popular explainability methods available in the field. We also offer some wrappers for explaining "shallow" sklearn models.

## What is the scope of SeqExplainer?

Most of the core functionality is for post-hoc analysis of a trained model. 

## Common workflows

### Feature attribution analysis (coming soon)

### Identifying motifs in attributions (coming soon)

### Testing feature dependencies (coming soon)

## Tutorials

###  Extracting motif syntax rules from a CNN (coming soon)

### Explaining predictions for shallow models on synthetic MPRAs using SHAP (coming soon)

## Requirements

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

## References
1. Novakovsky, G., Dexter, N., Libbrecht, M. W., Wasserman, W. W. & Mostafavi, S. Obtaining genetics insights from deep learning via explainable artificial intelligence. Nat. Rev. Genet. 1–13 (2022) doi:10.1038/s41576-022-00532-2
