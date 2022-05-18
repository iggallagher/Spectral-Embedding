import numpy as np
import scipy.stats as stats


def symmetrises(A, diag=False):
    if diag:
        return np.tril(A,0) + np.tril(A,-1).T
    else:
        return np.tril(A,-1) + np.tril(A,-1).T

    
def generate_B(K, rho=1):
    return symmetrises(rho * stats.uniform.rvs(size=(K,K)), diag=True)


def generate_SBM(n, B, pi):
    K = len(pi)
    if B.shape[0] != K or B.shape[1] != K:
        raise ValueError('B must be a square matrix size K-by-K')
    
    Z = np.random.choice(range(K), p=pi, size=n)
    A = symmetrises(stats.bernoulli.rvs(B[Z,:][:,Z]))
    
    return (A, Z)


def generate_MMSBM(n, B, alpha):
    K = len(alpha)
    if B.shape[0] != K or B.shape[1] != K:
        raise ValueError('B must be a square matrix size K-by-K')
    
    Z = stats.dirichlet.rvs(alpha, size=n)
    Zij = np.array([np.random.choice(range(K), p=Zi, size=n) for Zi in Z])
    A = symmetrises(stats.bernoulli.rvs(B[Zij,Zij.T]))
    
    return (A, Z)


def generate_DCSBM(n, B, pi, a=1, b=1):
    K = len(pi)
    if B.shape[0] != K or B.shape[1] != K:
        raise ValueError('B must be a square matrix size K-by-K')
    
    W = stats.beta.rvs(size=n, a=a, b=b)
    Z = np.random.choice(range(K), p=pi, size=n)
    A = symmetrises(stats.bernoulli.rvs(np.outer(W,W) * B[Z,:][:,Z]))
    
    return (A, Z, W)


def generate_WSBM(n, pi, params, distbn):
    K = len(pi)
    
    if distbn not in ['beta', 'exponential', 'gamma', 'gaussian', 'poisson']:
        raise ValueError('distbn must be beta, exponential, gamma, gaussian or poisson')
    
    if distbn == 'beta':
        if len(params) != 2 or params[0].shape != (K,K) or params[1].shape != (K,K):
            raise ValueError('params must be two square matrices size K-by-K [alphas, betas]')
            
        Z = np.random.choice(range(K), p=pi, size=n)
        A = symmetrises(stats.beta.rvs(a = params[0][Z,:][:,Z], b = params[1][Z,:][:,Z]))
        
    if distbn == 'exponential':
        if len(params) != 1 or params[0].shape != (K,K):
            raise ValueError('params must be one square matrix size K-by-K [lambdas]')
        
        Z = np.random.choice(range(K), p=pi, size=n)
        A = symmetrises(stats.expon.rvs(scale = 1/params[0][Z,:][:,Z]))
        
    if distbn == 'gamma':
        if len(params) != 2 or params[0].shape != (K,K) or params[1].shape != (K,K):
            raise ValueError('params must be two square matrices size K-by-K [alphas, betas]')
            
        Z = np.random.choice(range(K), p=pi, size=n)
        A = symmetrises(stats.gamma.rvs(a = params[0][Z,:][:,Z], scale = 1/params[1][Z,:][:,Z]))
        
    if distbn == 'gaussian':
        if len(params) != 2 or params[0].shape != (K,K) or params[1].shape != (K,K):
            raise ValueError('params must be two square matrices size K-by-K [means, variances]')
            
        Z = np.random.choice(range(K), p=pi, size=n)
        A = symmetrises(stats.norm.rvs(loc = params[0][Z,:][:,Z], scale = np.sqrt(params[1][Z,:][:,Z])))
        
    if distbn == 'poisson':
        if len(params) != 1 or params[0].shape != (K,K):
            raise ValueError('params must be one square matrix size K-by-K [lambdas]')
        
        Z = np.random.choice(range(K), p=pi, size=n)
        A = symmetrises(stats.poisson.rvs(mu = params[0][Z,:][:,Z]))
    
    return (A, Z)


def generate_WMMSBM(n, alpha, params, distbn):
    K = len(alpha)
    
    if distbn not in ['beta', 'exponential', 'gamma', 'gaussian', 'poisson']:
        raise ValueError('distbn must be beta, exponential, gamma, gaussian or poisson')
    
    if distbn == 'beta':
        if len(params) != 2 or params[0].shape != (K,K) or params[1].shape != (K,K):
            raise ValueError('params must be two square matrices size K-by-K [alphas, betas]')
            
        Z = stats.dirichlet.rvs(alpha, size=n)
        Zij = np.array([np.random.choice(range(K), p=Zi, size=n) for Zi in Z])
        A = symmetrises(stats.beta.rvs(a = params[0][Zij,Zij.T], b = params[1][Zij,Zij.T]))
        
    if distbn == 'exponential':
        if len(params) != 1 or params[0].shape != (K,K):
            raise ValueError('params must be one square matrix size K-by-K [lambdas]')
        
        Z = stats.dirichlet.rvs(alpha, size=n)
        Zij = np.array([np.random.choice(range(K), p=Zi, size=n) for Zi in Z])
        A = symmetrises(stats.expon.rvs(scale = 1/params[0][Zij,Zij.T]))
        
    if distbn == 'gamma':
        if len(params) != 2 or params[0].shape != (K,K) or params[1].shape != (K,K):
            raise ValueError('params must be two square matrices size K-by-K [alphas, betas]')
            
        Z = stats.dirichlet.rvs(alpha, size=n)
        Zij = np.array([np.random.choice(range(K), p=Zi, size=n) for Zi in Z])
        A = symmetrises(stats.gamma.rvs(a = params[0][Zij,Zij.T], scale = 1/params[1][Zij,Zij.T]))
        
    if distbn == 'gaussian':
        if len(params) != 2 or params[0].shape != (K,K) or params[1].shape != (K,K):
            raise ValueError('params must be two square matrices size K-by-K [means, variances]')
            
        Z = stats.dirichlet.rvs(alpha, size=n)
        Zij = np.array([np.random.choice(range(K), p=Zi, size=n) for Zi in Z])
        A = symmetrises(stats.norm.rvs(loc = params[0][Zij,Zij.T], scale = np.sqrt(params[1][Zij,Zij.T])))
        
    if distbn == 'poisson':
        if len(params) != 1 or params[0].shape != (K,K):
            raise ValueError('params must be one square matrix size K-by-K [lambdas]')
        
        Z = stats.dirichlet.rvs(alpha, size=n)
        Zij = np.array([np.random.choice(range(K), p=Zi, size=n) for Zi in Z])
        A = symmetrises(stats.poisson.rvs(mu = params[0][Zij,Zij.T]))
    
    return (A, Z)


def generate_WSBM_zero(n, pi, params, distbn, rho):
    (A, Z) = generate_WSBM(n, pi, params, distbn)
    W = symmetrises(stats.bernoulli.rvs(rho, size=(n,n)))
    return (W*A, Z)
                    
                    
def generate_WMMSBM_zero(n, alpha, params, distbn, rho):
    (A, Z) = generate_WMMSBM(n, pi, params, distbn)
    W = symmetrises(stats.bernoulli.rvs(rho, size=(n,n)))
    return (W*A, Z)


def generate_SBM_dynamic(n, Bs, pi):
    K = len(pi)
    T = Bs.shape[0]
    if Bs.shape[1] != K or Bs.shape[2] != K :
        raise ValueError('Bs must be array of T square matrices size K-by-K')
    
    Z = np.random.choice(range(K), p=pi, size=n)
    As = np.zeros((T,n,n))
    for t in range(T):
        As[t] = symmetrises(stats.bernoulli.rvs(Bs[t][Z,:][:,Z]))
    
    return (As, Z)


def generate_MMSBM_dynamic(n, Bs, alpha):
    K = len(alpha)
    T = Bs.shape[0]
    if Bs.shape[1] != K or Bs.shape[2] != K :
        raise ValueError('Bs must be array of T square matrices size K-by-K')
    
    Z = np.random.choice(range(K), p=pi, size=n)
    As = np.zeros((T,n,n))
    for t in range(T):
        Zij = np.array([np.random.choice(range(K), p=Zi, size=n) for Zi in Z])
        As[t] = symmetrises(stats.bernoulli.rvs(Bs[t][Zij,Zij.T]))
    
    return (As, Z)


def generate_DCSBM_dynamic(n, Bs, pi):
    K = len(pi)
    T = Bs.shape[0]
    if Bs.shape[1] != K or Bs.shape[2] != K :
        raise ValueError('Bs must be array of T square matrices size K-by-K')
    
    W = stats.uniform.rvs(size=n)
    Z = np.random.choice(range(K), p=pi, size=n)
    As = np.zeros((T,n,n))
    for t in range(T):
        As[t] = symmetrises(stats.bernoulli.rvs(np.outer(W,W) * Bs[t][Z,:][:,Z]))
    
    return (As, Z, W)