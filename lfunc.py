#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import healpy as hp

import matplotlib.pyplot as plt
from scipy.special import gammaln
import numpy.lib.recfunctions as recfunctions
import emcee


def chooseBins(catalog = None, tag=None, binsize = None, upperLimit = None, lowerLimit = None):
    if binsize is None:
        binsize = 2*( np.percentile(catalog[tag], 75) - np.percentile( catalog[tag], 25 ) ) / (catalog.size ) **(1./3.)
    if upperLimit is None:
        upperLimit = np.max(catalog[tag])
    if lowerLimit is None:
        lowerLimit = np.min(catalog[tag])
    nbins = int( np.ceil( (upperLimit - lowerLimit) / binsize) )
    nEdge = nbins+1
    bins = lowerLimit + binsize * np.arange(nEdge)
    bins[0] = bins[0] - 0.001*binsize
    bins[-1] = bins[-1] + 0.001*binsize
    return bins

        
def makeLikelihoodMatrix( sim=None, truth=None, truthMatched = None, Lcut = 0., ncut = 0.,
                          obs_bins = None, truth_bins = None, simTag = None, truthTag = None):
    if ( len(simTag) == 1 ) and (len(truthTag) == 1 ):
        obs_bin_index = np.digitize(sim[simTag], obs_bins) - 1
        truth_bin_index = np.digitize(truthMatched[truthTag], truth_bins) - 1
        # Limit loop to objects in the given bin ranges.
        nbins_truth = truth_bins.size -1
        nbins_obs = obs_bins.size - 1
        good = ((truth_bin_index > 0) & (truth_bin_index < nbins_truth) &
                (obs_bin_index   > 0) & (obs_bin_index   < nbins_obs) )
        obs_bin_index = obs_bin_index[good]
        truth_bin_index = truth_bin_index[good]
        N_truth, _ = np.histogram( truth[truthTag], bins=truth_bins )

        L = np.zeros( (nbins_obs, nbins_truth) )
    
        for i in xrange(obs_bin_index.size):
            if N_truth[truth_bin_index[i]] > ncut:
                L[obs_bin_index[i], truth_bin_index[i]] = ( L[obs_bin_index[i], truth_bin_index[i]] +
                                                            1./N_truth[truth_bin_index[i]] )
        L[L < Lcut] = 0.
    else:
        # In this case, the user has asked us to make a likelihood
        # matrix that maps an n-dimensional space onto another
        # n-dimensional space.
        # --------------------------------------------------
        #Assume that truth_bins and obs_bins are indexable.
        truth_bin_index = np.digitize(truthMatched[truthTag[0]], truth_bins[0]) - 1
        obs_bin_index = np.digitize(sim[simTag[0]], obs_bins[0]) - 1
        
        good = ((truth_bin_index > 0) & (truth_bin_index < (len(truth_bins[i+1]) - 1)) &
                (obs_bin_index   > 0) & (obs_bin_index   < (len(obs_bins[i+1]) -1)) )
        # --------------------------------------------------
        # Fancy multi-dimensional indexing.
        for i in xrange(len(truthTag) -1 ):
            this_truth_bin_index = np.digitize( truthMatched[truthTag[i+1]], truth_bins[i+1])
            this_obs_bin_index = np.digitize( truthMatched[truthTag[i+1]], obs_bins[i+1])
            good = good & ( (this_truth_bin_index > 0) & (truth_bin_index < (len(truth_bins[i+1]) - 1)) &
                            (this_obs_bin_index   > 0) & (obs_bin_index   < (len(obs_bins[i+1]) -1) ) )
            
            truth_bin_index = truth_bin_index + (len(truth_bins[i])-1) * this_truth_bin_index
            obs_bin_index = obs_bin_index + (len(obs_bins[i])-1) * this_obs_bin_index
        # --------------------------------------------------
        N_truth = np.bincount(truth_bin_index)
        for i in xrange(obs_bin_index.size):
            if N_truth[truth_bin_index[i]] > ncut:
                L[obs_bin_index[i], truth_bin_index[i]] = ( L[obs_bin_index[i], truth_bin_index[i]] +
                                                            1./N_truth[truth_bin_index[i]] )
        L[L < Lcut] = 0.
    return L
    


def getAllLikelihoods( truth=None, sim=None, truthMatched = None, healConfig=None , doplot = False, getBins = False, ncut = 0.,
            ratag= 'ra', dectag = 'dec', obs_bins = None, truth_bins = None, obsTag = 'mag_auto', truthTag = 'mag'):
    
    if healConfig is None:
        healConfig = getHealConfig()
        truth = HealPixifyCatalogs(catalog=truth, healConfig=healConfig, ratag=ratag, dectag = dectag)
        sim = HealPixifyCatalogs(catalog=sim, healConfig=healConfig, ratag=ratag, dectag = dectag)
        truthMatched = HealPixifyCatalogs(catalog=truthMatched, healConfig=healConfig, ratag=ratag, dectag = dectag)

    useInds = np.unique(sim['HEALIndex'])

    if obs_bins is None:
        obs_bins = chooseBins(catalog=sim, tag = obsTag, binsize=0.1,upperLimit=24.5,lowerLimit=15.)
    if truth_bins is None:
        truth_bins = chooseBins(catalog = truthMatched, tag = truthTag, binsize = 0.1,upperLimit=26.,lowerLimit=15)

    truth_bin_centers = (truth_bins[0:-1] + truth_bins[1:])/2.
    obs_bin_centers = (obs_bins[0:-1] + obs_bins[1:])/2.
        
    Lensemble = np.empty( (obs_bins.size-1 , truth_bins.size-1, useInds.size) )

    if doplot is True:
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.colors import LogNorm
        pp = PdfPages('likelihoods.pdf')
        fig,ax = plt.subplots(figsize=(6.,6.))
        # Make a plot of the likelihood of the whole region.
        masterLikelihood =  makeLikelihoodMatrix( sim=sim, truth=truth, truthMatched = truthMatched, Lcut = 0.,
                                                    obs_bins = obs_bins, truth_bins = truth_bins,
                                                    simTag = obsTag, truthTag = truthTag, ncut = ncut)
            
        im = ax.imshow(np.arcsinh(masterLikelihood/1e-3), origin='lower',cmap=plt.cm.Greys,
                       extent = [truth_bin_centers[0],truth_bin_centers[-1],obs_bin_centers[0],obs_bin_centers[-1]])
        ax.set_xlabel('truth ')
        ax.set_ylabel('measured ')
        ax.set_title('full area likelihood')
        fig.colorbar(im,ax=ax)
        pp.savefig(fig)
    else:
        masterLikelihood =  makeLikelihoodMatrix( sim=sim, truth=truth, truthMatched = truthMatched, Lcut = 0.,
                                                  obs_bins = obs_bins, truth_bins = truth_bins,
                                                  simTag = obsTag, truthTag = truthTag, ncut = ncut)
        
        
        
    for hpIndex,i in zip(useInds,xrange(useInds.size)):
        print "Processing likelihood "+str(i)+" of "+str(useInds.size-1)
        thisSim = sim[sim['HEALIndex'] == hpIndex]
        thisTruth = truth[truth['HEALIndex'] == hpIndex]
        thisTruthMatched = truthMatched[sim['HEALIndex'] == hpIndex]
        if thisTruth.size > 100:
            thisLikelihood = makeLikelihoodMatrix( sim=thisSim, truth=thisTruth, truthMatched = thisTruthMatched,Lcut = 0.,
                            obs_bins = obs_bins, truth_bins = truth_bins, simTag = obsTag, truthTag = truthTag)
            Lensemble[:,:,i] = thisLikelihood
            if doplot is True:
                fig,ax = plt.subplots(figsize = (6.,6.))
                im = ax.imshow(np.arcsinh(thisLikelihood/1e-3), origin='lower',cmap=plt.cm.Greys,
                               extent = [truth_bin_centers[0],truth_bin_centers[-1],obs_bin_centers[0],obs_bin_centers[-1]])
                ax.set_xlabel('truth mag.')
                ax.set_ylabel('measured mag.')
                ax.set_title('nside= '+str(healConfig['map_nside'])+', HEALPixel= '+str(hpIndex) )
                fig.colorbar(im,ax=ax)
                pp.savefig(fig)
            
    if doplot is True:
        pp.close()
    if getBins is False:
        return Lensemble, useInds, masterLikelihood, truth_bin_centers, obs_bin_centers
    if getBins is True:
        return Lensemble, useInds, masterLikelihood, truth_bins, obs_bins



def likelihoodPCA(likelihood= None,  doplot = False, band = None,
                  extent = None):
  # This does a simple PCA on the array of likelihood matrices to find
  # a compact basis with which to represent the likelihood.
    print "computing likelihood pca..."
    origShape = np.shape(likelihood)
    likelihood_1d = np.reshape(likelihood, (origShape[0]*origShape[1], origShape[2]))
    
    
    L1d = likelihood_1d.T.copy()
    U,s,Vt = np.linalg.svd(L1d,full_matrices=False)
    V = Vt.T
    ind = np.argsort(s)[::-1]
    ind = np.argsort(s)[::-1]
    U = U[:, ind]
    s = s[ind]
    V = V[:, ind]

    
    likelihood_pcomp = V.reshape(origShape)
    
    if doplot is True:
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.colors import LogNorm, Normalize
        if band is None:
            print "Must supply band (g,r,i,z,Y) in order to save PCA plots."
            stop
            
        pp = PdfPages('likelihood_pca_components-'+band+'.pdf')

        for i,thing in zip(xrange(s.size),s):
            print "plotting pca component "+str(i)+" of "+str(s.size-1)
            fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (6.,6.))
            im = ax.imshow( np.arcsinh(likelihood_pcomp[:,:,i]/1e-3),origin='lower',cmap=plt.cm.Greys, extent = extent)
            ax.set_xlabel(band+' mag (true)')
            ax.set_ylabel(band+' mag (meas)')
            fig.colorbar(im,ax=ax)
            pp.savefig(fig)
        fig,ax = plt.subplots(1,1,figsize = (6.,6.) )
        ax.plot(np.abs(s))
        ax.set_yscale('log')
        ax.set_xlabel('rank')
        ax.set_ylabel('eigenvalue')
        pp.savefig(fig)
        pp.close()
      
    return likelihood_pcomp, s

def doLikelihoodPCAfit(pcaComp = None,  likelihood =None, n_component = 5, Lcut = 0., Ntot = 1e5):

    # Perform least-squares: Find the best combination of pcaComps[:,:,0:n_component] that fits likelihood
    origShape = likelihood.shape

    L1d = likelihood.reshape(likelihood.size)
    pca1d = pcaComp.reshape( ( likelihood.size, pcaComp.shape[-1]) )
    pcafit = pca1d[:,0:(n_component)]

    # Full least-squares, taking the covariances of the likelihood into account.
    # covar(L1d) = Ntot * np.outer( L1d, L1d)
    # in the solution, the scaling Ntot falls out. However, we do need it for getting the errors later.
    L1dCovar = Ntot * np.outer(L1d, L1d)
    #aa= np.linalg.pinv( np.dot( pcafit.T, np.dot(L1dCovar, pcafit)) )
    #bb =  np.dot(pcafit.T, L1dCovar)
    #coeff = np.dot( np.dot(aa,bb), L1d)
    #coeffCovar = aa
    
    coeff, resid, _, _  = np.linalg.lstsq(pcafit, L1d)
    bestFit = np.dot(pcafit,coeff)
    bestFit2d = bestFit.reshape(likelihood.shape)
    bestFit2d[bestFit2d < Lcut] = 0.
    
    return bestFit2d, coeff#, coeffCovar

def mcmcLogL(N_truth, N_data, likelihood, lninf=-1000):
    if np.sum(N_truth < 0.) > 0:
        return -np.inf

    pObs  = np.dot(likelihood, N_truth) / np.sum(N_truth)
    pMiss = 1. - np.sum(pObs)

    Nmiss = np.sum(N_truth) - np.sum( np.dot( likelihood, N_truth) ) 
    Nobs = np.sum( N_data )

    
    if pMiss == 0.:
        logPmiss = -np.inf
    else:
        logPmiss = np.log(pMiss)

    
    lpObs = np.zeros(pObs.size)
    valid = ( pObs > 0. )
    lpObs[valid] = np.log(pObs[valid])
    lpObs[~valid] = lninf

    t4 = np.dot(np.transpose(N_data), lpObs)
    t5 = Nmiss * logPmiss
    t1 = gammaln(1 + Nmiss + Nobs)
    t2 = gammaln(1 + Nmiss)
    t3 = np.sum(gammaln(1 + likelihood))
    logL = t1 - t2 - t3 + t4 + t5

    return logL


def initializeMCMC(N_data, likelihood, multiplier = 1.):
    print "Initializing MCMC..."
    A = likelihood.copy()
    Ainv = np.linalg.pinv(A,rcond = 0.001)
    N_initial = np.abs(np.dot(Ainv, N_data))
    covar_truth = np.diag(N_initial)
    Areg = np.dot(Ainv, A)
    covar_recon = np.dot( np.dot(Areg, covar_truth), Areg.T)
    leakage = np.abs(np.dot( Areg, N_initial) - N_initial)
    errors = np.sqrt( np.diag(covar_recon) ) + leakage
    nParams = likelihood.shape[1]
    nWalkers = np.min( [100*nParams, 2000.] )

    N_initial = N_initial*0. + np.mean(N_data)
    start= np.sqrt( ( N_initial + (multiplier*errors*N_initial) * np.random.randn( nWalkers, nParams ) )**2 )

    return start, nWalkers



def doInference(catalog = None, likelihood = None, obs_bins=None, truth_bins = None, tag = 'mag_auto',
                invType = 'basic', lambda_reg = 1e-6, prior = None, priorNumber = None):


    N_real_obs, _  = np.histogram(catalog[tag], bins = obs_bins)
    N_real_obs = N_real_obs*1.0
    A = likelihood.copy()
    
    if invType is 'basic':
        if prior is None:
            prior = np.zeros(truth_bins.size-1)

        Ainv = np.linalg.pinv(A,rcond = lambda_reg)
        N_real_truth = np.dot(Ainv, N_real_obs - np.dot(A, prior)) + prior
        covar_truth = np.diag(N_real_truth)
        Areg = np.dot(Ainv, A)
        covar_recon = np.dot( np.dot(Areg, covar_truth), Areg.T)
        leakage = np.abs(np.dot( Areg, N_real_truth) - N_real_truth)    
        errors = np.sqrt( np.diag(covar_recon) ) + leakage 

    if invType is 'tikhonov':
        if prior is None:
            prior = np.zeros(truth_bins.size-1)
        Ainv = np.dot( np.linalg.pinv(np.dot(A.T, A) + lambda_reg * np.identity(truth_bins.size - 1 ) ), A.T)
        N_real_truth = np.dot(Ainv, N_real_obs - np.dot(A, prior)) + prior
        covar_truth = np.diag(N_real_truth)
        Areg = np.dot(Ainv, A)
        covar_recon = np.dot( np.dot(Areg, covar_truth), Areg.T)
        leakage = np.abs(np.dot( Areg, N_real_truth) - N_real_truth)
        aa = np.dot(A.T, A)
        aainv = np.linalg.pinv(aa)
        g = np.trace( lambda_reg * aainv)
        reg_err = lambda_reg / (1 + g) * np.dot( np.dot( np.dot( aainv, aainv), A.T) , N_real_obs)        
        errors = np.sqrt( np.diag(covar_recon) ) + leakage + np.abs(reg_err)

    if invType is 'bayesian':
        if prior is None:
            print "must specify a prior in order to use the Bayesian reconstruction option."

        # Build a model for the data covariance.
        covarPrior = np.diag(np.abs(prior))
        covarData = np.dot( np.dot(A, covarPrior), A.T)* catalog.size * 1./priorNumber
        
        P = np.linalg.pinv(covarData)
        Q = np.linalg.pinv(covarPrior)
        Ainv = np.dot( np.linalg.pinv( np.dot(np.dot(A.T, P), A ) + Q ), np.dot(A.T, P) )

        N_real_truth = prior + np.dot( Ainv, N_real_obs - np.dot(A,prior))

        # Calculate errors.
        Areg = np.dot( Ainv, A)
        covar_recon = np.dot( np.dot( Areg, covarPrior ), Areg.T )
        aa = np.dot(A.T, A)
        aainv = np.linalg.pinv(aa)
        g = np.trace( lambda_reg * aainv)
        reg_err = lambda_reg / (1 + g) * np.dot( np.dot( np.dot( aainv, aainv), A.T) , N_real_obs)        
        leakage = np.abs(np.dot(Areg, prior) - prior)
        errors = np.sqrt( np.diag(covar_recon) ) + leakage + np.abs(reg_err)
        
    if invType is 'mcmc':
        start, nWalkers = initializeMCMC(N_real_obs, A)
        nParams = likelihood.shape[1]

        nSteps = 1000
        sampler = emcee.EnsembleSampler(nWalkers, nParams, mcmcLogL, args = [N_real_obs, A], threads = 8)
        

        print "burninating mcmc"
        pos, prob, state = sampler.run_mcmc(start, nSteps)
        mean_accept = np.mean(sampler.acceptance_fraction)
        sampler.reset()
        delta_mean_accept = 1.
        print "Acceptance fraction: ",mean_accept
        print "running mcmc"

        while np.abs(delta_mean_accept) > 0.001:
            pos, prob, state = sampler.run_mcmc( pos, nSteps, rstate0 = state )
            delta_mean_accept = np.mean(sampler.acceptance_fraction) - mean_accept
            mean_accept = np.mean(sampler.acceptance_fraction)
            print "Acceptance fraction: ",mean_accept
            #print "autocorr_time", sampler.acor
            N_real_truth = np.mean( sampler.flatchain, axis=0 )
            errors = np.std( sampler.flatchain, axis=0 )
            sampler.reset()
            

    truth_bins_centers = (truth_bins[0:-1] + truth_bins[1:])/2.

    return N_real_truth, errors, truth_bins_centers


