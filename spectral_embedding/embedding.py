import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.sparse as sparse


def left_embed(A, d):
    if sparse.issparse(A):
        UA, SA, VAt = sparse.linalg.svds(A, d)
    else:
        UA, SA, VAt = np.linalg.svd(A)
        
    XA = UA[:,0:d] @ np.diag(np.sqrt(SA[0:d]))  
    return XA


def right_embed(A, d):
    if sparse.issparse(A):
        UA, SA, VAt = sparse.linalg.svds(A, d)
    else:
        UA, SA, VAt = np.linalg.svd(A)
        
    VA = VAt.T
    YA = VA[:,0:d] @ np.diag(np.sqrt(SA[0:d]))  
    return YA


def both_embed(A, d):
    if sparse.issparse(A):
        UA, SA, VAt = sparse.linalg.svds(A, d)
    else:
        UA, SA, VAt = np.linalg.svd(A)
        
    VA = VAt.T
    XA = UA[:,0:d] @ np.diag(np.sqrt(SA[0:d]))
    YA = VA[:,0:d] @ np.diag(np.sqrt(SA[0:d]))  
    return (XA, YA)


def safe_inv_sqrt(x):
    if x < 0:
        raise ValueError('x must be non-negative')
    if x == 0:
        return 0
    if x > 0:
        return np.power(x, -0.5)


def ASE(A, d):
    return left_embed(A, d)


def LSE(A, d):
    if sparse.issparse(A):
        L = sparse.csgraph.laplacian(A, normed=True)
    else:
        E = np.diag([safe_inv_sqrt(d) for d in np.sum(A, axis=0)])
        L = E @ A @ E
        
    return left_embed(L, d)


def RWSE(A, d):
    E = np.diag([safe_inv_sqrt(d) for d in np.sum(A, axis=0)])
    XL = LSE(A, d)
    return (E @ XL)[:,1:]


def UASE(As, d):
    T = len(As)
    n = As[0].shape[0]
    
    if np.all([sparse.issparse(A) for A in As]):
        A = sparse.hstack(As)
    else:
        A = np.block([A for A in As])
        
    XA, YA = both_embed(A, d)
    YAs = YA.reshape((T, n, d))
        
    return (XA, YAs)


def omnibus(As, d):
    T = len(As)
    n = As[0].shape[0]
    
    A = np.zeros((T*n,T*n))
    for t1 in range(T):
        for t2 in range(T):
            A[t1*n:(t1+1)*n,t2*n:(t2+1)*n] = (As[t1] + As[t2])/2
            
    YA = left_embed(A, d)
    YAs = np.zeros((T,n,d))
    for t in range(T):
        YAs[t] = YA[t*n:(t+1)*n]
        
    return YAs
    

def dim_select(A, max_dim=100):        
    SA = np.linalg.svd(A, compute_uv=False)
    
    # If no max dimension chosen or above number of nodes, set to the number of nodes
    if max_dim == 0 or max_dim > len(SA):
        max_dim = len(SA)
    
    # Compute likelihood profile
    lq = np.zeros(max_dim-1)
    for q in range(max_dim-1):
        # Break between qth and (q+1)th singular values
        theta_0 = np.mean(SA[:q+1])
        theta_1 = np.mean(SA[q+1:max_dim])
        sigma = np.sqrt(((q-1)*np.var(SA[:q+1]) + (max_dim-q-1)*np.var(SA[q+1:max_dim])) / (max_dim-2))
        lq_0 = np.sum(stats.norm.logpdf(SA[:q+1], theta_0, sigma))
        lq_1 = np.sum(stats.norm.logpdf(SA[q+1:max_dim], theta_1, sigma))
        lq[q] = lq_0 +lq_1
       
    # Return best number of dimensions
    lq_best = np.nanargmax(lq)+1
    
    return lq_best, lq, SA[:max_dim]


def plot_dim_select(lq_best, lq, S, max_plot=50):
    max_dim = len(S)
    
    # If no max plot chosen or above number of nodes, set to maximum dimension
    if max_plot == 0 or max_plot > max_dim:
        max_plot = max_dim
    
    fig, axs = plt.subplots(2, 1, figsize=(8.0,6.0), sharex=True)
        
    axs[0].plot(range(1,max_dim+1), S, '.')
    axs[0].set_title('Singular values')
    axs[0].set_xlim([0,max_plot])
    axs[0].axvline(x=lq_best+0.5, ls='--', c='k')
    
    axs[1].plot(range(1,max_dim), lq, '.')
    axs[1].set_title('Log likelihood')
    axs[1].set_xlabel('Number of dimensions')
    axs[1].set_xlim([0,max_plot])
    axs[1].axvline(x=lq_best+0.5, ls='--', c='k')
        
    return