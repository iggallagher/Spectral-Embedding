import numpy as np


def left_embed(A, d):
    UA, SA, VAt = np.linalg.svd(A)
    XA = UA[:,0:d] @ np.diag(np.sqrt(SA[0:d]))  
    return XA


def right_embed(A, d):
    UA, SA, VAt = np.linalg.svd(A)
    VA = VAt.T
    YA = VA[:,0:d] @ np.diag(np.sqrt(SA[0:d]))  
    return YA


def both_embed(A, d):
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
    E = np.diag([safe_inv_sqrt(d) for d in np.sum(A, axis=0)])
    L = E @ A @ E
    return left_embed(L, d)


def UASE(As, d):
    T = len(As)
    n = As[0].shape[0]
    
    A = np.block([A for A in As])
    XA, YA = both_embed(A, d)
    
    YAs = np.zeros((T,n,d))
    for t in range(T):
        YAs[t] = YA[t*n:(t+1)*n]
        
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


def dim_select(A, plot=True, max_dim=100):        
    SA = np.linalg.svd(A, compute_uv=False)
    
    # If no max dimension chosen, set to the number of nodes
    if max_dim == 0:
        max_dim = len(SA)
    
    # Compute likelihood profile
    lq = np.zeros(max_dim); lq[0] = 'nan'
    for q in range(1,max_dim):
        theta_0 = np.mean(SA[:q])
        theta_1 = np.mean(SA[q:max_dim])
        sigma = np.sqrt(((q-1)*np.var(SA[:q]) + (max_dim-q-1)*np.var(SA[q:max_dim])) / (max_dim-2))
        lq_0 = np.sum(np.log(stats.norm.pdf(SA[:q], theta_0, sigma)))
        lq_1 = np.sum(np.log(stats.norm.pdf(SA[q:max_dim], theta_1, sigma)))
        lq[q] = lq_0 +lq_1    
    lq_best = np.nanargmax(lq)

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(8.0,6.0), sharex=True)
        
        axs[0].plot(range(1,max_dim+1), SA[:max_dim], '.-')
        axs[0].set_title('Singular values')
        axs[0].axvline(x=lq_best, ls='--', c='k')

        axs[1].plot(range(max_dim), lq[:max_dim], '.-')
        axs[1].set_title('Log likelihood')
        axs[1].set_xlabel('Number of dimensions')
        axs[1].axvline(x=lq_best, ls='--', c='k');
        
    return lq_best