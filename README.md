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

Note that this package is still in development, so please check back here for information regarding updates, new functionality and examples added to this package. If there is anything you would like to see included in this package, or to report bugs and errors, please email me at ian.gallagher@bristol.ac.uk.

## Installation

The package can be installed via pip using the command:
`pip install git+https://github.com/iggallagher/Spectral-Embedding`.

## Examples

Below is a brief introduction to each of the examples included in the package outlining some of the key functionality.

### Stochastic Block Model Embedding.ipynb
This example introduces the different network generation and embedding functions. Three small example networks are randomly generated; a two-community stochastic block model, mixed membership stochastic block model and a degree-corrected stochastic block model. These are embedded into two dimensions using the adjacency spectral embedding and Laplacian spectral embedding. We expain the structure of the resulting embeddings compared to the community structure in the different variations of the stochastic block model.

See this notebook for examples of the following functions:
- Network generation: `generate_SBM`, `generate_MMSBM` and `generate_DCSBM`,
- Spectral embedding: `ASE` and `LSE`.

### Weighted Stochastic Block Models.ipynb
This example introduces weighted network generation, asymptotic distribution and size-adjusted Chernoff information functions. A weighted two-community stochastic block model is generated where edges have a Poisson distribution which is embedded into two dimensions using adjacency spectral embedding. The asymptotic distribution of the spectral embedding for both communities is computed and shown as a Gaussian mixture model, and the size-adjusted Chernoff information is calculated. This is compared to an entry-wise transformation of the weighted network which records whether an edge has a non-zero weight.

See this notebook for examples of the following functions:
- Weighted network generation: `generate_WSBM`,
- Spectral embedding: `ASE`,
- Asymptotic distribution calculation: `WSBM_distbn` and `gaussian_ellipse`,
- Size-adjusted Chernoff information calculation: `chernoff`.

### Mixed Membership and Degree-Corrected SBM Distributions.ipynb ###
This example introduces asymptotic distribution functions for the mixed membership stochastic block model and degree-corrected stochastic block model. A two-community mixed membership stochastic block model and a degree-corrected stochastic block model are generated and both are embedded into two dimensions using adjacency spectral embedding. The asymptotic distribution of the spectral embedding is computed for different community assignment probabilities for the mixed membership stochastic block model and for different weights for the degree-corrected stochastic block model.

See this notebook for examples of the following functions:
- Network generation: `generate_MMSBM` and `generate_DCSBM`,
- Spectral embedding: `ASE`,
- Asymptotic distribution calculation: `MMSBM_distbn`, `DCSBM_distbn` and `gaussian_ellipse`.

### Dynamic Network Embedding.ipynb
This example introduces adjacency spectral embedding for a sequence of networks, known as a dynamic network. A dynamic network consisting of four-community stochastic block models is generated and embedded into two dimensions using unfolded adjacency spectral embedding along with the asymptotic distribution for the unfolded adjacency spectral embedding for each community at each time period. The output is compared to the embeddings generated by computing separate adjacency spectral embeddings for each time period, and the omnibus embedding. Some discussion about the stability properties of the three different approaches and why unfolded adjacency spectral embedding is to be preferred.

See this notebook for examples of the following functions:
- Dynamic network generation: `generate_SBM_dynamic`,
- Spectral embedding: `UASE`, `ASE` and `omnibus`,
- Asymptotic distribution calculation: `SBM_dynamic_distbn`, `SBM_distbn` and `gaussian_ellipse`.

### Sparse Matrix Embedding.ipynb
This example introduces adjacency spectral embedding for sparse networks. There is currently no functionality to generate random sparse networks, but the example covers adjacency spectral embedding, Laplacian spectral embedding and unfolded adjacency spectral embedding for sparse networks.

See this notebook for examples of the following functions:
- Spectral embedding: `ASE`, `LSE` and `UASE`.

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
- `left_embed(A, d, version='sqrt')`: Compute the d-dimensional spectral embedding using the left singular values and vectors of the d-truncated singular value decomposition of the matrix A.
- `right_embed(A, d, version='sqrt')`: Compute the d-dimensional spectral embedding using the right singular values and vectors of the d-truncated singular value decomposition of the matrix A.
- `both_embed(A, d, version='sqrt')`: Compute both the left and right d-dimensional spectral embeddings given above.

These embeddings have a `version` parameter whcih determines which factorisation of the matrix A should be used. In all cases, the default option is `version='sqrt'`. Given the singular value decomposition $A \approx U S V^\top$, where the left and right embeddings are scaled by the square root of the eigenvalues, the left embedding $\hat{X} = US^{1/2}$ and the right embedding $\hat{Y} = VS^{1/2}$. For the left and right embedding functions, the other options are `version='full'`, where the left and right embeddings are scaled by the eigenvalues, $\hat{X} = US$ and the right embedding by $\hat{Y} = VS$, and `version='none'`, where the left and right embeddings are not scaled by the eigenvalues, $\hat{X} = U$ and the right embedding by $\hat{Y} = V$.

For the both embedding function, if not using the option `version='sqrt'`, you must either specific how to scale the left embedding (`version='fullleft' or `version='noneleft') or how to scale the left embedding (`version='fullright' or `version='noneright'). Note that `version='fullleft' is identical to `version='noneright' and similarly, `version='noneleft' is identical to `version='fullright'. Whatever option used $A \approx \hat{X} \hat{Y}^\top$.

These embedding techniques are then used for different represenations of a graph adjacency matrix.
- `ASE(A, d, version='sqrt')`: Compute the d-dimensional adjacency spectral embedding of an adjacency matrix A.
- `LSE(A, d, version='sqrt')`: Compute the d-dimensional Laplacian spectral embedding of an adjacency matrix A.
- `RWSE(A, d, version='sqrt')`: Compute the d-dimensional random walk spectral embedding of an adjacency matrix A.

The embedding techniques are also used to embed a time series of adjacency matrices.
- `UASE(As, d, version='sqrt')`: Compute the d-dimensional left and right unfolded adjacency spectral embedding for a sequence of adjacency matrices As.
- `omnibus(As, d)`: Compute the d-dimensional omnibus spectral embedding for a sequence of adjacency matrices As. For more details see Levin, K., Athreya, A., Tang, M., Lyzinski, V., Park, Y., and Priebe, C. E. (2017). A central limit theorem for an omnibus embedding of multiple random graphs and implications for multiscale network inference. [*arXiv preprint arXiv:1705.09355*](https://arxiv.org/abs/1705.09355). These functions inherit the same options for `version` as their underlying embedding functions; `left_embed`, `right_embed` and `both_embed`.
 
This section also includes functionality to choose the dimensionality for an adjacency spectral embeddding. For more details see Zhu, M. and Ghodsi, A. (2006). Automatic dimensionality selection from the scree plot via the use of profile likelihood.
- `dim_select(A, max_dim=100)`: Compute the profile likelihood for the singular values of the singular value decomposition of an adjacency matrix A considering the first max_dim singular values.
- `plot_dim_select(lq_best, lq, S, max_plot=50)`: Produce a visualation of the output of the `dim_select` function; lq_best, lq, S.

### distribution.py
This collection of functions computes the asymptotic distributions for the adjacency spectral embedding of the stochastic block model and its variations using the central limit theorems from the papers discussed in the Introduction. Note that these functions assume that the adjacency spectral embedding for the parameter `version='sqrt'`.
- `SBM_distbn(A, B, Z, pi, d)`: Compute the asymptotic distribuion for the adjacency spectral embedding of the stochastic block model with adjacency matrix A, block mean matrix B, community assignment Z, community assignment probabilities pi, and embedding dimension d. The output is the asymptotic means and covariances for each of the K communities.
- `MMSBM_distbn(A, B, Z, alpha, d, zs)`: Compute the asymptotic distribuion for the adjacency spectral embedding of the mixed membership stochastic block model with adjacency matrix A, block mean matrix B, community assignment distributions Z, Dirichlet distribution parameter alpha, and embedding dimension d. The output is the asymptotic means and covariances for each of the community assignment distributions given by zs.
- `DCSBM_distbn(A, B, Z, pi, d, ws, a=2, b=2)`: Compute the asymptotic distribuion for the adjacency spectral embedding of the degree-corrected stochastic block model with adjacency matrix A, block mean matrix B, community assignment Z, community assignment probabilities pi, and embedding dimension d, where the node-specific weights are generated using a beta distribution using parameters a and b. The output is the asymptotic means and covariances for each of the K communities with weights given by ws.

These all have weighted versions which work in the same way except the block variance matrix C must also be provided.
- `WSBM_distbn(A, B, C, Z, pi, d)`.
- `WMMSBM_distbn(A, B, C, Z, alpha, d, zs)`.
- `WDCSBM_distbn(A, B, C, Z, pi, d, ws, a=2, b=2)`.

There also exists functions to compute the asymptotic distributions for the right unfolded adjacency spectral embedding for a sequence of (weighted) stochastic block models.
- `SBM_dynamic_distbn(As, Bs, Z, pi, d)`: Compute the asymptotic distribuion for the right unfolded adjacency spectral embedding for a sequence of stochastic block models with adjacency matrices As, block mean matrices Bs, community assignment Z, community assignment probabilities pi, and embedding dimension d. The output is the asymptotic means and covariances for each of the K communities for each adjacency matrix.
- `WSBM_dynamic_distbn(As, Bs, Cs, Z, pi, d)`: A weighted version of the previous function also using the sequence of block variance matrices Cs.

Finally, this section also includes a function to aid plotting the resulting asymptotic distributions produced by the other functions.
- `gaussian_ellipse(mean, cov)`: Produce a curve corresponding to the 95% confidence ellipses of a Gaussian distribuion with a specific mean and covariance.

### chernoff.py
This collection of functions computes the size-adjusted Chernoff information for the adjacency spectral embedding of a stochastic block model.
- `chernoff(X, SigmaXs)`: Compute the size-adjusted Chernoff information of an embedding with means X and covariances SigmaXs. If there are more than two communities, the worst size-adjusted Chernoff information is outputted.
- `chernoff_full(X, SigmaXs)`: Compute all the size-adjusted Chernoff informations of an embedding with means X and covariances SigmaXs. If there are more than two communities, the size-adjusted Chernoff information for every pair of communities is outputted.

The section also includes some utility functions for the optimisation of the size-adjusted Chernoff information calculations.
