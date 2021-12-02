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