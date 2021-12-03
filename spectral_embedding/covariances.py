import numpy as np


def SBM_covariance(A, B, Z, pi, d):
    return WSBM_covariance(A, B, B*(1-B), Z, pi, d):


def WSBM_covariance(A, B, C, Z, pi, d):
    P = B[Z,:][:,Z]
    K = B.shape[0]
            
    # Spectral embeddings
    UA, SA, VAt = np.linalg.svd(A); VA = VAt.T
    UB, SB, VBt = np.linalg.svd(B); VB = VBt.T
    UP, SP, VPt = np.linalg.svd(P); VP = VPt.T

    XA = UA[:,0:d] @ np.diag(np.sqrt(SA[0:d]))
    XB = UB[:,0:d] @ np.diag(np.sqrt(SB[0:d]))
    XP = UP[:,0:d] @ np.diag(np.sqrt(SP[0:d]))

    XZ = XB[Z,:]
    
    # Find spectral embedding map to latent positions
    UW, _, VWt = np.linalg.svd(UP[:,0:d].T @ UA[:,0:d] + VP[:,0:d].T @ VA[:,0:d])
    W = UW.dot(VWt)
    L = np.linalg.inv(XZ.T @ XZ) @ XZ.T @ XP
    
    X = XB @ L @ W
    
    # Covariance matrices
    XBinv = np.linalg.pinv(XB)
    Lambda = XBinv @ B @ XBinv.T
        
    Sigmas = np.zeros((K,d,d))
    for i in range(K):
        for j in range(K):
            Sigmas[i] += pi[j]*C[i,j]*np.outer(XB[j],XB[j])
        
    Delta = np.zeros((d,d))
    for i in range(K):
        Delta += pi[i]*np.outer(XB[i],XB[i])
   
    D = np.linalg.inv(Lambda @ Delta @ Lambda.T) @ Lambda
    SigmaXs = np.zeros((K,d,d))
    for i in range(K):
        SigmaXs[i] = W.T @ L.T @ D @ Sigmas[i] @ D.T @ L @ W
   
    return (X, SigmaXs)