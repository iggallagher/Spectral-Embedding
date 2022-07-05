# Spectral-Embedding

## Introduction

This package includes a variety of tools for spectral embedding of graphs, using the theory of the generalised random dot product graph and its many extensions to analyse networks. Details of these approaches can be found in the following papers:
- Gallagher, I., Jones, A., Bertiger, A., Priebe, C., & Rubin-Delanchy, P. (2019). Spectral embedding of weighted graphs. [*arXiv preprint arXiv:1910.05534*](https://arxiv.org/abs/1910.05534).
- Gallagher, I., Jones, A., & Rubin-Delanchy, P. (2021). Spectral embedding for dynamic networks with stability guarantees. *Advances in Neural Information Processing Systems, 34*. [*arXiv preprint arXiv:2106.01282*](https://arxiv.org/abs/2106.01282)
- Jones, A., & Rubin-Delanchy, P. (2020). The multilayer random dot product graph. [*arXiv preprint arXiv:2007.10455*](https://arxiv.org/abs/2007.10455).
- Modell, A., Gallagher, I., Cape, J., & Rubin-Delanchy, P. (2022). Spectral embedding and the latent geometry of multipartite networks. [*arXiv preprint arXiv:2202.03945*](https://arxiv.org/abs/2202.03945).
- Rubin-Delanchy, P., Cape, J., Tang, M., & Priebe, C. E. (2022). A statistical interpretation of spectral embedding: the generalised random dot product graph. *Journal of the Royal Statistical Society: Series B*. [*arXiv preprint arXiv:1709.05506*](https://arxiv.org/abs/1709.05506).

Currently, the package has functionality that can be divided into four sections:
- Network generation,
- Spectral embedding algorithms,
- Calculation of the embedding asymptotic distribution,
- Chernoff information calculations.

There are also a number of example Python notebooks which demonstrate different functionality from the four sections of this package. The rest of this README is structured as follows:
- **Installation** describes how to install this Python package using pip.
- **Examples** goes through the example Python notebooks is detail, describe which functions of the package are being used.
- **Functions** gives a full description of the functions in the four different sections, explaining input, output and function parameters.

Note that this package is still in development, so please check back here for information regarding updates, new functionality and examples added to this package. If there is anything you would like to see included in this package, or to report bugs and error, please email me at ian.gallagher@bristol.ac.uk.

## Installation

The package can be installed via pip using the command:
`pip install git+https://github.com/iggallagher/Spectral-Embedding`.

## Examples

### Stochastic Block Model Embedding.ipynb
TO DO

### Weighted Stochastic Block Models.ipynb
TO DO

### Dynamic Network Embedding.ipynb
TO DO

### Sparse Matrix Embedding.ipynb
TO DO

## Functions

### network_generation.py
This collection of functions generates random graphs based on the stochastic block model and its variations.

#### `generate_SBM(n, B, pi)`

### embedding.py
TO DO

### distribution.py
TO DO

### chernoff.py
TO DO
