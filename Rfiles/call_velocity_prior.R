# Code to compute velocity prior (i.e. the mean and covariance of a 2D
# Gaussian) at a given HEALpixel and distance,
# as developed and used in GDR3 distances and velocities paper VI
# https://doi.org/10.3847/1538-3881/ad08bb
# CBJ 2023-11-27
# Code extracted from est_distvel.R and functions.R
# sfit is a smooth.spline{stats} model

############ Functions

# Evaluate parameters of velocity prior by evaluating all Ncomp components of
# smoothing spline (sfit) at specified distance r, and return as
# Ncomp-element named vector with the names as in sfit. 
# Currently these are two mean, two SDs, on correlation:
# vramean, vrasd, vdecmean, vdecsd, cor 
# As spline has no natural bounds (e.g. to >0) impose some limits.
eval.sfit <- function(sfit, r) {
  Ncomp <- length(sfit)
  velprior <- rep(NA, Ncomp)
  names(velprior) <- names(sfit)
  rmax <- attr(sfit, "rmax")
  r <- ifelse(r>rmax, rmax, r)
  for(i in 1:Ncomp) {
    velprior[i] <- predict(sfit[[i]], r)$y
  }
  # Enforce minimum values of SD (km/s), in particular >0.
  velprior["vrasd"]  <- ifelse(velprior["vrasd"]  < +2, +2, velprior["vrasd"])
  velprior["vdecsd"] <- ifelse(velprior["vdecsd"] < +2, +2, velprior["vdecsd"])
  # Enforce minimum and maximum values of correlation
  velprior["cor"] <- ifelse(velprior["cor"] < -0.95, -0.95, velprior["cor"])
  velprior["cor"] <- ifelse(velprior["cor"] > +0.95, +0.95, velprior["cor"])
  #
  return(velprior)
}

# prior10: Product of two 1D Gaussians in vra and vdec
# Return normalized prior density. If log=TRUE, return as natural log.
# vel = (vra, vdec). Not named.
# velprior = (vramean, vrasd, vdecmean, vdecsd). Named.

d.prior10 <- function(vel, velprior, log=FALSE) {
  if(log) {
    return(as.numeric(dnorm(x=vel[1], mean=velprior["vramean"],  sd=velprior["vrasd"], log=TRUE) +
                        dnorm(x=vel[2], mean=velprior["vdecmean"], sd=velprior["vdecsd"], log=TRUE) ))
    
  } else {
    return(as.numeric(dnorm(x=vel[1], mean=velprior["vramean"],  sd=velprior["vrasd"], log=FALSE) *
                        dnorm(x=vel[2], mean=velprior["vdecmean"], sd=velprior["vdecsd"], log=FALSE) ))
  }
}


# Function which evaluates prior at given Healpixel p, distance r: 

eval.prior.healpix <- function(p,r) {
    
    tempEnv <- new.env()
    
    #velpriorfnamestem <- "/jupyter/ad/no233/linux/R/interactive_distance_estimation/Rfiles/velocity_prior_fits/models/" #velocity_prior_fits/models/" 
    # Load velocity prior for HEALpixel number p (level 5)
    
    #velpriorfname <- paste0(velpriorfnamestem, p, ".Robj") #, sep=""
   #if(!file.exists(velpriorfname)) {
   #  cat(p, "velocity prior model file", velpriorfname, "does not exist.\n")
     #clean.end(writedistvelfile=FALSE)
   #}
    #load(velpriorfname, envir=tempEnv)
    
    load(url(paste0("https://www.mpia.de/homes/calj/gedr3_distances_velocities/velocity_prior_fits/models/",p,".Robj")), envir=tempEnv)
    
    sfit <- tempEnv$sfit
    # Evaluate velocity prior at specified distance r (scalar, in pc)
    velprior <- eval.sfit(sfit=sfit, r=r)
    
    #return  vramean, vrasd, vdecmean, vdecsd, cor 
    getwd()
    return(velprior)
    }
    
#