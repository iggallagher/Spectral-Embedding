import numpy as np
import scipy.stats as stats


def symmetrises(A, diag=False):
    if diag:
        return np.tril(A,0) + np.transpose(np.tril(A,-1))
    else:
        return np.tril(A,-1) + np.transpose(np.tril(A,-1))

    
def generate_B(K, rho=1):
    return symmetrises(rho * stats.uniform.rvs(size=(K,K)), diag=True)


def generate_SBM(n, B, pi):
    if B.shape[0] != B.shape[1]:
        raise ValueError('B must be a square matrix size K-by-K')
    if len(pi) != B.shape[0]:
        raise ValueError('pi must be an array length K')
    
    K = len(pi)
    Z = np.random.choice(range(K), p=pi, size=n)
    A = symmetrises(stats.bernoulli.rvs(B[Z,:][:,Z]))
    
    return (A, Z)


def generate_MMSBM(n, B, alpha):
    if B.shape[0] != B.shape[1]:
        raise ValueError('B must be a square matrix size K-by-K')
    if len(alpha) != B.shape[0]:
        raise ValueError('alpha must be an array length K')
    
    K = len(alpha)
    Z = stats.dirichlet.rvs(alpha, size=n)
    Zij = np.array([np.random.choice(range(K), p=Zi, size=n) for Zi in Z])
    A = symmetrises(stats.bernoulli.rvs(B[Zij,Zij.T]))
    
    return (A, Z)


def generate_DCSBM(n, B, pi):
    if B.shape[0] != B.shape[1]:
        raise ValueError('B must be a square matrix size K-by-K')
    if len(pi) != B.shape[0]:
        raise ValueError('pi must be an array length K')
    
    K = len(pi)
    W = stats.uniform.rvs(size=n)
    Z = np.random.choice(range(K), p=pi, size=n)
    A = symmetrises(stats.bernoulli.rvs(np.outer(W,W) * B[Z,:][:,Z]))
    
    return (A, Z, W)