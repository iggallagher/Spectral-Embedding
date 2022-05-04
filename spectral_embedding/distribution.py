import numpy as np
from scipy.special import gamma


def SBM_distbn(A, B, Z, pi, d):
    return WSBM_distbn(A, B, B*(1-B), Z, pi, d)


def WSBM_distbn(A, B, C, Z, pi, d):
    P = B[Z,:][:,Z]
    K = len(pi)
            
    # Spectral embeddings
    UA, SA, VAt = np.linalg.svd(A); VA = VAt.T
    UB, SB, VBt = np.linalg.svd(B); VB = VBt.T
    UP, SP, VPt = np.linalg.svd(P); VP = VPt.T

    XB = UB[:,0:d] @ np.diag(np.sqrt(SB[0:d]))
    XP = UP[:,0:d] @ np.diag(np.sqrt(SP[0:d]))
    XZ = XB[Z,:]
    
    # Find spectral embedding map to latent positions
    UW, _, VWt = np.linalg.svd(UP[:,0:d].T @ UA[:,0:d] + VP[:,0:d].T @ VA[:,0:d])
    W = UW @ VWt
    L = np.linalg.inv(XZ.T @ XZ) @ XZ.T @ XP
    
    X = XB @ L @ W
    
    # Covariance matrices
    XBinv = np.linalg.pinv(XB)
    Lambda = XBinv @ B @ XBinv.T
        
    Sigma = np.zeros((K,d,d))
    for i in range(K):
        for j in range(K):
            Sigma[i] += pi[j]*C[i,j]*np.outer(XB[j],XB[j])
        
    Delta = np.zeros((d,d))
    for i in range(K):
        Delta += pi[i]*np.outer(XB[i],XB[i])
   
    D = np.linalg.inv(Lambda @ Delta @ Lambda.T) @ Lambda
    SigmaX = np.zeros((K,d,d))
    for i in range(K):
        SigmaX[i] = W.T @ L.T @ D @ Sigma[i] @ D.T @ L @ W
   
    return (X, SigmaX)


def dirichlet_moment(alpha, beta):
    alpha0 = np.sum(alpha)
    beta0  = np.sum(beta)
    K = len(alpha)
    
    moment = gamma(alpha0) / gamma(alpha0 + beta0)
    for k in range(K):
        moment *= gamma(alpha[k] + beta[k]) / gamma(alpha[k])
        
    return moment


def MMSBM_distbn(A, B, C, Z, alpha, d):
    return WMMSBM_distbn(A, B, B*(1-B), Z, alpha, d)


def WMMSBM_distbn(A, B, C, Z, alpha, d):
    P = Z @ B @ Z.T
    K = len(alpha)
            
    # Spectral embeddings
    UA, SA, VAt = np.linalg.svd(A); VA = VAt.T
    UB, SB, VBt = np.linalg.svd(B); VB = VBt.T
    UP, SP, VPt = np.linalg.svd(P); VP = VPt.T

    XB = UB[:,0:d] @ np.diag(np.sqrt(SB[0:d]))
    XP = UP[:,0:d] @ np.diag(np.sqrt(SP[0:d]))
    XZ = Z @ XB
    
    # Find spectral embedding map to latent positions
    UW, _, VWt = np.linalg.svd(UP[:,0:d].T @ UA[:,0:d] + VP[:,0:d].T @ VA[:,0:d])
    W = UW @ VWt
    L = np.linalg.inv(XZ.T @ XZ) @ XZ.T @ XP
    
    X = XB @ L @ W
    
    # Covariance matrices
    XBinv = np.linalg.pinv(XB)
    Lambda = XBinv @ B @ XBinv.T
    
    Sigma = np.zeros((K,d,d))
    for i in range(K):
        for j in range(K):
            for k in range(K):
                for l in range(K):
                    beta_C = np.zeros((K))
                    beta_C[j] += 1; beta_C[k] += 1; beta_C[l] += 1
                    beta_B = np.zeros((K))
                    beta_B[j] += 2; beta_B[k] += 1; beta_B[l] += 1
                    Sigma[i] += (dirichlet_moment(alpha, beta_C)*C[i,j] - 
                                 dirichlet_moment(alpha, beta_B)*B[i,j]**2) * np.outer(XB[k],XB[l])
        
    Delta = np.zeros((d,d))
    for i in range(K):
        for j in range(K):
            beta = np.zeros((K))
            beta[i] += 1; beta[j] += 1
            Delta += dirichlet_moment(alpha, beta) * np.outer(XB[i],XB[j])
   
    D = np.linalg.inv(Lambda @ Delta @ Lambda.T) @ Lambda
    SigmaX = np.zeros((K,d,d))
    for i in range(K):
        SigmaX[i] = W.T @ L.T @ D @ Sigma[i] @ D.T @ L @ W
   
    return (X, SigmaX)


def SBM_dynamic_distbn(As, Bs, Z, pi, d):
    return WSBM_dynamic_distbn(As, Bs, Bs*(1-Bs), Z, pi, d)


def WSBM_dynamic_distbn(As, Bs, Cs, Z, pi, d):
    T = As.shape[0]
    K = len(pi)
    
    A = np.block([A for A in As])
    B = np.block([B for B in Bs])
    P = np.block([B[Z,:][:,Z] for B in Bs])
    
    # Spectral embeddings
    UA, SA, VAt = np.linalg.svd(A); VA = VAt.T
    UB, SB, VBt = np.linalg.svd(B); VB = VBt.T
    UP, SP, VPt = np.linalg.svd(P); VP = VPt.T

    XB = UB[:,0:d] @ np.diag(np.sqrt(SB[0:d]))
    XP = UP[:,0:d] @ np.diag(np.sqrt(SP[0:d]))
    XZ = XB[Z,:]
        
    # Map spectral embeddings to latent positions
    UW, _, VWt = np.linalg.svd(UP[:,0:d].T @ UA[:,0:d] + VP[:,0:d].T @ VA[:,0:d])
    W = UW @ VWt
    L = np.linalg.inv(XZ.T @ XZ) @ XZ.T @ XP
    R = np.linalg.inv(L.T) @ W
    
    # Covariance matrices    
    Sigmas = np.zeros((T,K,d,d))
    for t in range(T):
        for i in range(K):
            for j in range(K):
                Sigmas[t,i] += pi[j]*Cs[t,i,j]*np.outer(XB[j],XB[j])
          
    Delta = np.zeros((d,d))
    for i in range(K):
        Delta += pi[i]*np.outer(XB[i],XB[i])
    DeltaInv = np.linalg.inv(Delta)

    Ys = np.zeros((T,d,d))
    
    for t in range(T):    
        Ys[t] = VB[t*d:(t+1)*d,0:d] @ np.diag(np.sqrt(SB[0:d])) @ R       
    
    SigmaYs = np.zeros((T,K,d,d))
    for t in range(T):
        for i in range(K):
            SigmaYs[t,i] = R.T @ DeltaInv @ Sigmas[t,i] @ DeltaInv @ R
    
    return (Ys, SigmaYs)


def gaussian_ellipse(mean, cov):
    if mean.shape == (1,2):
        mean = np.array(mean)[0]
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    rtheta = np.arctan2(u[1], u[0])
    v = 3. * np.sqrt(2.) * np.sqrt(v)
    width = v[0]; height = v[1]
    
    R = np.array([
        [np.cos(rtheta), -np.sin(rtheta)],
        [np.sin(rtheta),  np.cos(rtheta)],
        ])    
    theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
    x = 0.5 * width * np.cos(theta)
    y = 0.5 * height * np.sin(theta)    
    x, y = np.dot(R,np.array([x, y]))
    x += mean[0]
    y += mean[1]
    
    return [x,y]