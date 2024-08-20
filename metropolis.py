# Metropolis (MCMC) algorithm

# Samples from function func. The function func() has to return an array, the logPosterior (log base 10)

# thetaInit: initial guess
# Nburnin : Number of Burn-In's
# Nsamp: Samplenumber
# sampleCov: Covariance of samples
# returns a Nsamp * (1+Ntheta) array, where the columns are
# 1:  log10 posterior PDF
# 2+: Ntheta parameters

import numpy as np

def metrop(func,thetaInit,Nburnin,Nsamp,sampleCov,seed,**kwargs):
    
    np.random.seed(seed)
    if not np.isscalar(thetaInit): 
        Ntheta = len(thetaInit)
    else:
        Ntheta = 1
        
    thetaCur = thetaInit
    funcCur = func(thetaInit,**kwargs)
    funcSamp = np.empty((Nsamp,1+Ntheta))
    funcSamp[:] = np.nan
    
    nAccept = 0
    acceptRate = 0
    
    for n in np.arange(1,Nburnin+Nsamp+1):
        
        if np.isscalar(sampleCov):
            thetaProp = np.random.normal(loc=thetaCur, scale=np.sqrt(sampleCov), size=1)
        else:
            thetaProp = np.random.multivariate_normal(mean=thetaCur, cov=sampleCov, size = 1)
            
        funcProp = func(thetaProp,**kwargs)
        logMR = funcProp - funcCur 
        
        if logMR >= 0 or logMR > np.log10(np.random.uniform(low=0, high=1, size=1)):
            thetaCur = thetaProp
            funcCur = funcProp
            nAccept = nAccept + 1
            acceptRate = nAccept/n
        
        if n > Nburnin:
            funcSamp[n-Nburnin-1,0:1] = funcCur
            funcSamp[n-Nburnin-1,1:(1+Ntheta)] = thetaCur
    
    return funcSamp