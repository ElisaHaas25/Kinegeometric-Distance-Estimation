# File contains all functions for kinegeometric distance and velocity estimation

import packages
from functions import *



# 2D Correlation vector between parallax uncertainty (sigma_w) and proper motion uncertainties in ra/dec (sigma_mu_alpha/delta delta) (rho_w_mu_alpha/delta : correlation)

def f_Sigma_mu_w(sigma_w,sigma_mu_alpha,sigma_mu_delta,rho_w_mu_alpha,rho_w_mu_delta):
    
    return np.array([sigma_w*sigma_mu_alpha*rho_w_mu_alpha, sigma_w*sigma_mu_delta*rho_w_mu_delta])

#variance of parallax 

def f_Sigma_w_w(sigma_w):
    return sigma_w**2

# 2x2 covariance matrix of proper motion containing variance of proper motions (sigma_mu_alpha/delta), correlation between the two (rho_mu_alpha_delta):

def f_Sigma_mu_mu(sigma_mu_alpha,sigma_mu_delta,rho_mu_alpha_delta):
    return np.array([[sigma_mu_alpha**2, sigma_mu_alpha*sigma_mu_delta*rho_mu_alpha_delta], [sigma_mu_alpha*sigma_mu_delta*rho_mu_alpha_delta, sigma_mu_delta**2]])

# mean m_2 and covariance Sigma_2 of 2D likelihood in proper motion conditioned on the parallax, for given v and r. 

#Input: m_mu, Sigma_mu_w, Sigma_w_w, m_w

def f_m_2(m_mu, Sigma_mu_w, Sigma_w_w, w, m_w):
    
    
    m_2 = m_mu + Sigma_mu_w*Sigma_w_w**(-1)*(w - m_w)
    
    return  m_2

def f_Sigma_2(Sigma_mu_mu, Sigma_mu_w,Sigma_w_w):
    
    return Sigma_mu_mu - np.dot(Sigma_mu_w,Sigma_mu_w) * Sigma_w_w**(-1)

# mean m_v and covariance Sigma_v of 2D Gaussian PDF in velocity 
# m_tau, Sigma_tau: mean and covariance of 2D Gaussian prior in velocity conditional on the distance and direction

def f_m_v(r,Sigma_2,m_2,Sigma_tau,m_tau):
    k = 4.740471
    M_v1 = np.linalg.inv(np.linalg.inv(k**2 * r**2 * Sigma_2) + np.linalg.inv(Sigma_tau))
    M_v2 = np.dot(np.linalg.inv(k**2 * r**2 * Sigma_2) , k*r*m_2) + np.dot(np.linalg.inv(Sigma_tau),m_tau)
    
    return np.dot(M_v1,M_v2)

def f_Sigma_v(r,Sigma_2,Sigma_tau):
    k = 4.740471
    return np.linalg.inv(np.linalg.inv(k**2 * r**2 * Sigma_2) + np.linalg.inv(Sigma_tau))

#get the required r function for the kinegeometric prior statistics from the R-code

r_source2 = robjects.r['source']
r_source2('./Rfiles/call_velocity_prior.R')

kinegeo_prior_stats = robjects.globalenv['eval.prior.healpix']

# Function to calculate the velocity quantiles
#
# Input: 
#
# rSamp: array containing all distance estimation samples from the distance posterior (in kpc)
# n: number of samples drawn for each distance
# hp: healpixel at which to evaluate the prior
# probs: array containing the quantiles to be evaluated
# w: parallax (in mas)
# wsd: parallax error (in mas)
# mu_ra, mu_dec: proper motion in ra, dec (in mas/yr)
# sd_mu_ra, sd_mu_dec: proper motion error in ra, dec (in mas/yr)
# corr_w_mu_ra,corr_w_mu_dec: correlations between parallax and proper motion in ra/dec 
# corr_mu_ra_dec: between proper motion in ra and dec
#
# Function first calculates the mean m_v and covariance Sigma_v of 2D Gaussian PDF in velocity for each distance value in rSamp. Then it draws n velocity samples from this distribution (2D-vectors) and calculates the mean velocity for each of the distance samples. Then for the mean velocity samples, the quantiles are calculated. The output is then a tuple containing:
# v_ra_res, v_dec_res: arrays containing the in probs specified quantiles of the velocity in ra/dec
# v_corr_res: correlation between array of v_ra and v_dec samples
# Sigma_rv_ra, Sigma_rv_dec: covariance between each velocity in ra/dec and distance
# velocitySamples_allr_mean: array containing all velocity samples


def quantile_velpost(rSamp ,n ,hp , probs, w ,wsd, mu_ra, mu_dec,sd_mu_ra, sd_mu_dec,corr_w_mu_ra,corr_w_mu_dec,corr_mu_ra_dec):
    
    Nsamp = len(rSamp)
    
    if not (Nsamp > 0 and Nsamp != np.inf):
        return 'Nsamp not (finite and positive)'
    if not (n > 0 and n != np.inf):
        return 'n not (finite and positive)'
    if np.isnan(np.array([ mu_ra, mu_dec,sd_mu_ra, sd_mu_dec,corr_w_mu_ra,corr_w_mu_dec,corr_mu_ra_dec])).any():
        return 'some inputs NA'
    if np.any(probs<0) or np.any(probs>1):
        return 'probs not in range 0-1'
    if np.isnan(rSamp).any():
        return 'some values in rSamp NA'
    
    
    # list which will later contain all n 2D velocity samples for each of the N_g distances r in rSamp:
    velocitySamples_allr = [] 
    # list which will later contain the means of the n 2D velocity samples for each of the N_g distances r in rSamp
    velocitySamples_allr_mean = []
    # list with m_v for all N_g distances
    m_v_all = []
    
    for i in range(len(rSamp)):
        
        #mean and covariance of 2D velocity posterior conditional on distance
        
        vramean_tau, vrasd_tau, vdecmean_tau, vdecsd_tau, cor_tau = kinegeo_prior_stats(p=hp,r=rSamp[i])  
        
        m_tau = np.array([vramean_tau,vdecmean_tau])
        Sigma_tau = np.array([[vrasd_tau**2 ,vrasd_tau * vdecsd_tau *cor_tau], [vrasd_tau*vdecsd_tau*cor_tau,vdecsd_tau**2]])
        
        #mean and covariance of product 2D Gaussian:[ 2D velocity posterior conditioned on distance * 2D proper motion posterior (convert mean and covariance into velocity using distance) ]
        
        m_w = 1/rSamp[i] 
        m_mu = np.array([mu_ra,mu_dec])
        
        Sigma_mu_w = f_Sigma_mu_w(sigma_w = wsd ,sigma_mu_alpha = sd_mu_ra ,sigma_mu_delta = sd_mu_dec, rho_w_mu_alpha = corr_w_mu_ra, rho_w_mu_delta = corr_w_mu_dec)
        Sigma_w_w = f_Sigma_w_w(sigma_w = wsd)
        Sigma_mu_mu = f_Sigma_mu_mu(sigma_mu_alpha = sd_mu_ra, sigma_mu_delta = sd_mu_dec, rho_mu_alpha_delta = corr_mu_ra_dec)
        
        m_2 = f_m_2(m_mu=m_mu, Sigma_mu_w=Sigma_mu_w, Sigma_w_w=Sigma_w_w, w=w, m_w = m_w) 
        Sigma_2 = f_Sigma_2(Sigma_mu_mu=Sigma_mu_mu, Sigma_mu_w=Sigma_mu_w,Sigma_w_w=Sigma_w_w)
        
        m_v = f_m_v(r=rSamp[i],Sigma_2=Sigma_2,m_2=m_2,Sigma_tau=Sigma_tau,m_tau=m_tau)
        Sigma_v = f_Sigma_v(r=rSamp[i],Sigma_2=Sigma_2,Sigma_tau=Sigma_tau)
        
        velocitySamples_singler = np.random.multivariate_normal(mean = m_v, cov = Sigma_v,size = n)
        velocitySamples_singler_mean = np.mean(velocitySamples_singler, axis=0)
        
        m_v_all.append(m_v)
        velocitySamples_allr.append(velocitySamples_singler) 
        velocitySamples_allr_mean.append(velocitySamples_singler_mean) 
    
    #convert list to arrays
    
    velocitySamples_allr = np.array(velocitySamples_allr)
    velocitySamples_allr_mean = np.array(velocitySamples_allr_mean)
    m_v_all = np.array(m_v_all)
    
    #get results: mean and quantiles of v_ra and v_dec + correlation between array of v_ra and v_dec samples
    
    v_ra_res = np.quantile(velocitySamples_allr_mean[:,0],probs)
    v_dec_res = np.quantile(velocitySamples_allr_mean[:,1],probs)
    v_corr_res = np.corrcoef(velocitySamples_allr_mean[:,0],velocitySamples_allr_mean[:,1])
    
    # Expectation value of v
    
    E_v = 1/Nsamp * sum(m_v_all)
    
    #covariance between each velocity and distance
    
    r_mean = 1/Nsamp * sum(rSamp)
    
    Sigma_rv_ra = 1/Nsamp * sum((rSamp-r_mean)*(m_v_all[:,0]-E_v[0]))
    Sigma_rv_dec = 1/Nsamp * sum((rSamp-r_mean)*(m_v_all[:,1]-E_v[1]))
    
    return v_ra_res, v_dec_res, v_corr_res, Sigma_rv_ra, Sigma_rv_dec, velocitySamples_allr_mean  