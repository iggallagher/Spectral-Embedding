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
- `generate_SBM(n, B, pi)`: Generate a stochastic block model with n nodes using block mean matrix B and community role assignment pi.
- `generate_MMSBM(n, B, alpha)`: Generate a mixed membership stochastic block model with n nodes using block mean matrix B where the community distributions are generated using a Dirichlet distribution using parameter alpha.
- `generate_DCSBM(n, B, pi, a=1, b=1)`: Generate a degree-corrected stochastic block model with n nodes using block mean matrix B and community role assignment pi. Node-specific weights are generated using a beta distribution using parameters a and b.

These all have weighted versions where edge weights can be generated from a beta, exponential, gamma, Gaussian or Poisson.
- `generate_WSBM(n, pi, params, distbn)`: Generate a weighted stochastic block model with n nodes and community role assignment pi. Given the distribution distbn, the list of matrices params give the necessary parameters for the distributions for each of the different communities.
- `generate_WSBM_zero(n, pi, params, distbn, rho)`: Same as the above but with a sparsity parameter rho that dictates how many edges are present in the graph.
- `generate_WMMSBM(n, alpha, params, distbn)`: Generate a mixed membership stochastic block model with n nodes where the community distributions are generated using a Dirichlet distribution using parameter alpha. Given the distribution distbn, the list of matrices params give the necessary parameters for the distributions for each of the different communities.
- `generate_WMMSBM_zero(n, alpha, params, distbn, rho)`: Same as the above but with a sparsity parameter rho that dictates how many edges are present in the graph.
- `generate_ZISBM(n, pi, params, distbn, a=1, b=1)`: Generate a zero-inflated stochastic block model (the weighted version of the degree-corrected stochastic block model) with n nodes using community role assignment pi. Node-specific weights are generated using a beta distribution using parameters a and b. Given the distribution distbn, the list of matrices params give the necessary parameters for the distributions for each of the different communities.

There exists dynamic versions of the stochastic block model and its variations generation a time series of graphs. In each case, the number of networks is given by the length of the array of block mean matrices Bs.
- `generate_SBM_dynamic(n, Bs, pi)`: Generate a sequence of stochastic block models with n nodes using block mean matrices Bs and community role assignment pi.
- `generate_MMSBM_dynamic(n, Bs, alpha)`: Generate a mixed membership stochastic block model with n nodes using block mean matrices Bs where the community distributions are generated using a Dirichlet distribution using parameter alpha. The number of networks is given by the length of Bs.
- `generate_DCSBM_dynamic(n, Bs, pi)`: Generate a degree-corrected stochastic block model with n nodes using block mean matrices Bs and community role assignment pi. Node-specific weights are generated using a beta distribution using parameters a and b.

Finally, there are some utility functions that may be useful for generate random graphs.
- `symmetrises(A, diag=False)`: Return a symmetric version of the matrix A using the lower right triangle of the matrix. The parameter diag determines whether the diagonal of the matrix A should be kept or set to zero.
- `generate_B(K, rho=1)`: Generate a random block mean matrix for K communities with maximum value given by rho.

### embedding.py
This collection of functions computes different spectral embeddings of a network. All of the techniques involved rely on the left and right spectral embedding from a singular value decomposition of a matrix. All of these algorithms allow for the matrix A to be a sparse or dense matrix.
- `left_embed(A, d)`: Compute the d-dimensional spectral embedding using the left singular values and vectors of the d-truncated singular value decomposition of the matrix A.
- `right_embed(A, d)`: Compute the d-dimensional spectral embedding using the right singular values and vectors of the d-truncated singular value decomposition of the matrix A.
- `both_embed(A, d)`: Compute both the left and right d-dimensional spectral embeddings given above.

These embedding techniques are then used for different represenations of a graph adjacency matrix.
- `ASE(A, d)`: Compute the d-dimensional adjacency spectral embedding of an adjacency matrix A.
- `LSE(A, d)`: Compute the d-dimensional Laplacian spectral embedding of an adjacency matrix A.
- `RWSE(A, d)`: Compute the d-dimensional random walk spectral embedding of an adjacency matrix A.

The embedding techniques are also used to embed a time series of adjacency matrices.
- `UASE(As, d)`: Compute the d-dimensional left and right unfolded adjacency spectral embedding for a sequence of adjacency matrices As.
- `omnibus(As, d)`: Compute the d-dimensional omnibus spectral embedding for a sequence of adjacency matrices As. For more details see Levin, K., Athreya, A., Tang, M., Lyzinski, V., Park, Y., and Priebe, C. E. (2017). A central limit theorem for an omnibus embedding of multiple random graphs and implications for multiscale network inference. [*arXiv preprint arXiv:1705.09355*](https://arxiv.org/abs/1705.09355).

This section also includes functionality to choose the dimensionality for an adjacency spectral embeddding. For more details see Zhu, M. and Ghodsi, A. (2006). Automatic dimensionality selection from the scree plot via the use of profile likelihood.
- `dim_select(A, max_dim=100)`: Compute the profile likelihood for the singular values of the singular value decomposition of an adjacency matrix A considering the first max_dim singular values.
- `plot_dim_select(lq_best, lq, S, max_plot=50)`: Produce a visualation of the output of the `dim_select` function; lq_best, lq, S.

### distribution.py
TO DO

### chernoff.py
TO DO
