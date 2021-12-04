import numpy as np


def chernoff_full(X, SigmaXs):
    CIs = chernoff_full(X, SigmaXs)
    return CI
    

def chernoff_full(X, SigmaXs):
    K = X.shape[0]
    CIs = np.zeros((K,K))
    
    for i in range(K-1):
        for j in range(i,K):
            CIs[i,j] = -minimize(f_RD, 0.0, (X[i], X[j], SigmaXs[i], SigmaXs[j]), method='TNC').fun
            CIs[j,i] = CIs[i,j]
            
    return CIs


def renyi(t, X0, X1, SigmaX0, SigmaX1):
    return 0.5*t*(1-t) * (X0-X1).T @ np.linalg.inv(t*SigmaX0 + (1-t)*SigmaX1) @ (X0-X1)


def logit(x):
    return 1/(1 + np.exp(-x))


def f_RD(x, X0, X1, SigmaX0, SigmaX1):
    return -renyi(logit(x), X0, X1, SigmaX0, SigmaX1)