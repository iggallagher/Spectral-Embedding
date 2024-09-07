import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.sparse as sparse


def left_embed(A, d, version='sqrt'):
    if version not in ['full', 'none', 'sqrt']:
        raise ValueError('version must be full, none or sqrt (default)')
        
    if sparse.issparse(A):
        UA, SA, VAt = sparse.linalg.svds(A, d)
        if version == 'sqrt':
            XA = np.fliplr(UA[:,0:d]) @ np.diag(np.sqrt(np.flip(SA[0:d])))
        if version == 'full':
            XA = np.fliplr(UA[:,0:d]) @ np.diag(np.flip(SA[0:d]))
        if version == 'none':
            XA = np.fliplr(UA[:,0:d])
    else:
        UA, SA, VAt = np.linalg.svd(A)
        if version == 'sqrt':
            XA = UA[:,0:d] @ np.diag(np.sqrt(SA[0:d]))
        if version == 'full':
            XA = UA[:,0:d] @ np.diag(SA[0:d])
        if version == 'none':
            XA = UA[:,0:d]
            
    return XA


def right_embed(A, d, version='sqrt'):
    if version not in ['full', 'none', 'sqrt']:
        raise ValueError('version must be full, none or sqrt (default)')
        
    if sparse.issparse(A):
        UA, SA, VAt = sparse.linalg.svds(A, d)
        VA = VAt.T
        if version == 'sqrt':
            YA = np.fliplr(VA[:,0:d]) @ np.diag(np.sqrt(np.flip(SA[0:d])))
        if version == 'full':
            YA = np.fliplr(VA[:,0:d]) @ np.diag(np.flip(SA[0:d]))
        if version == 'none':
            YA = np.fliplr(VA[:,0:d])   
    else:
        UA, SA, VAt = np.linalg.svd(A)
        VA = VAt.T
        if version == 'sqrt':
            YA = VA[:,0:d] @ np.diag(np.sqrt(SA[0:d]))  
        if version == 'full':
            YA = VA[:,0:d] @ np.diag(SA[0:d])
        if version == 'none':
            YA = VA[:,0:d]
            
    return YA


def both_embed(A, d, version='sqrt'):
    if version not in ['fullleft', 'fullright', 'noneleft', 'noneright', 'sqrt']:
        raise ValueError('version must be fullleft (noneright), fullright (noneleft) or sqrt (default)')
        
    if sparse.issparse(A):
        UA, SA, VAt = sparse.linalg.svds(A, d)
        VA = VAt.T
        if version == 'sqrt':
            XA = np.fliplr(UA[:,0:d]) @ np.diag(np.sqrt(np.flip(SA[0:d])))
            YA = np.fliplr(VA[:,0:d]) @ np.diag(np.sqrt(np.flip(SA[0:d])))
        if version == 'fullleft' or version == 'noneright':
            XA = np.fliplr(UA[:,0:d]) @ np.diag(np.flip(SA[0:d]))
            YA = np.fliplr(VA[:,0:d])
        if version == 'noneleft' or version == 'fullright':
            XA = np.fliplr(UA[:,0:d])
            YA = np.fliplr(VA[:,0:d]) @ np.diag(np.flip(SA[0:d])) 
    else:
        UA, SA, VAt = np.linalg.svd(A)
        VA = VAt.T
        if version == 'sqrt':
            XA = UA[:,0:d] @ np.diag(np.sqrt(SA[0:d]))
            YA = VA[:,0:d] @ np.diag(np.sqrt(SA[0:d]))  
        if version == 'fullleft' or version == 'noneright':
            XA = UA[:,0:d] @ np.diag(SA[0:d])
            YA = VA[:,0:d]
        if version == 'noneleft' or version == 'fullright':
            XA = UA[:,0:d]
            YA = VA[:,0:d] @ np.diag(SA[0:d])
            
    return (XA, YA)


def safe_inv_sqrt(x):
    if x < 0:
        raise ValueError('x must be non-negative')
    if x == 0:
        return 0
    if x > 0:
        return np.power(x, -0.5)


def ASE(A, d, version='sqrt'):
    if version not in ['full', 'none', 'sqrt']:
        raise ValueError('version must be full, none or sqrt (default)')
    
    return left_embed(A, d, version)


def LSE(A, d, version='sqrt'):
    if version not in ['full', 'none', 'sqrt']:
        raise ValueError('version must be full, none or sqrt (default)')
        
    if sparse.issparse(A):
        L = sparse.csgraph.laplacian(A, normed=True)
    else:
        E = np.diag([safe_inv_sqrt(d) for d in np.sum(A, axis=0)])
        L = E @ A @ E
        
    return left_embed(L, d, version)


def RWSE(A, d, version='sqrt'):
    if version not in ['full', 'none', 'sqrt']:
        raise ValueError('version must be full, none or sqrt (default)')
    
    if sparse.issparse(A):
        E = sparse.diags(np.array([safe_inv_sqrt(d + gamma) for d in np.sum(A, axis=1)]).reshape(-1))
    else:
        E = np.diag([safe_inv_sqrt(d + gamma) for d in np.sum(A, axis=1)])
    XL = LSE(A, d, version)
    return (E @ XL)[:,1:]


def RLSE(A, d, gamma=None, version='sqrt'):
    if version not in ['full', 'none', 'sqrt']:
        raise ValueError('version must be full, none or sqrt (default)')

    if gamma == None:
        gamma = np.mean(np.sum(A, axis=1))
    
    if sparse.issparse(A):
        E = sparse.diags(np.array([safe_inv_sqrt(d) for d in np.sum(A, axis=1)]).reshape(-1))
    else:
        E = np.diag([safe_inv_sqrt(d + gamma) for d in np.sum(A, axis=1)])
    L = E @ A @ E
        
    return left_embed(L, d, version)
    
    
def UASE(As, d, version='sqrt'):
    if version not in ['fullleft', 'fullright', 'noneleft', 'noneright', 'sqrt']:
        raise ValueError('version must be fullleft (noneright), fullright (noneleft) or sqrt (default)')
    
    T = len(As)
    n = As[0].shape[0]
    
    if np.all([sparse.issparse(A) for A in As]):
        A = sparse.hstack(As)
    else:
        A = np.block([A for A in As])
        
    XA, YA = both_embed(A, d, version)
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
    # If no max dimension chosen or above number of nodes, set to the number of nodes
    if max_dim == 0 or max_dim > A.shape[0]:
        max_dim = A.shape[0]
    
    if sparse.issparse(A):
        _, SA, _ = sparse.linalg.svds(A, max_dim)
        SA = np.flip(SA)
    else:
        SA = np.linalg.svd(A, compute_uv=False)
    
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
