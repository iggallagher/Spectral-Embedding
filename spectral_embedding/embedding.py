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


def safe_inv(x):
    if x == 0:
        return 0
    else:
        return 1/x
    
    
def ASE(A, d):
    return left_embed(A, d)


def LSE(A, d):
    E = np.diag([safe_inv_sqrt(d) for d in np.sum(A, axis=0)])
    L = E @ A @ E
    return left_embed(L, d)


def RWSE(A, d):
    E = np.diag([safe_inv(d) for d in np.sum(A, axis=0)])
    L = E @ A
    return left_embed(L, d)
