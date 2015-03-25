#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')

import desdb
import numpy as np
import esutil
import pyfits
import sys
import argparse
import healpy as hp
import os
import functions2
import slr_zeropoint_shiftmap as slr
import numpy.lib.recfunctions as rf

import matplotlib.pyplot as plt




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


def makeLikelihoodMatrix( sim=None, truth=None, truthMatched = None, Lcut = 0.,
                          obs_bins = None, truth_bins = None, simTag = None, truthTag = None):
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
        if N_truth[truth_bin_index[i]] > 0:
            L[obs_bin_index[i], truth_bin_index[i]] = ( L[obs_bin_index[i], truth_bin_index[i]] +
                                                        1./N_truth[truth_bin_index[i]] )
    L[L < Lcut] = 0.
    return L
    

def getAllLikelihoods( truth=None, sim=None, truthMatched = None, healConfig=None , doplot = False, getBins = False,
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
                                                  obs_bins = obs_bins, truth_bins = truth_bins, simTag = obsTag, truthTag = truthTag)
        im = ax.imshow(masterLikelihood, origin='lower',cmap=plt.cm.Greys, norm = LogNorm(vmin=1e-8,vmax=1),
                       extent = [truth_bin_centers[0],truth_bin_centers[-1],obs_bin_centers[0],obs_bin_centers[-1]])
        ax.set_xlabel('truth mag.')
        ax.set_ylabel('measured mag.')
        ax.set_title('full area likelihood')
        fig.colorbar(im,ax=ax)
        pp.savefig(fig)
        
    for hpIndex,i in zip(useInds,xrange(useInds.size)):
        thisSim = sim[sim['HEALIndex'] == hpIndex]
        thisTruth = truth[truth['HEALIndex'] == hpIndex]
        thisTruthMatched = truthMatched[sim['HEALIndex'] == hpIndex]
        if thisTruth.size > 100:
            thisLikelihood = makeLikelihoodMatrix( sim=thisSim, truth=thisTruth, truthMatched = thisTruthMatched,Lcut = 0.,
                            obs_bins = obs_bins, truth_bins = truth_bins, simTag = obsTag, truthTag = truthTag)
            Lensemble[:,:,i] = thisLikelihood
            if doplot is True:
                fig,ax = plt.subplots(figsize = (6.,6.))
                im = ax.imshow(thisLikelihood, origin='lower',cmap=plt.cm.Greys, norm = LogNorm(vmin=1e-6,vmax=1),
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



def likelihoodPCA(likelihood= None,  likelihood_master = None, doplot = False, band = None,
                  extent = None):
  # This does a simple PCA on the array of likelihood matrices to find a compact basis with which to represent the likelihood.
    origShape = np.shape(likelihood)
    likelihood_1d = np.reshape(likelihood, (origShape[0]*origShape[1], origShape[2]))
    
    L1d_master = np.reshape(likelihood_master, origShape[0]*origShape[1])
    
    # Subtract L1d_master from each row of L1d:
    #for i in xrange(origShape[2]):
    #    likelihood_1d[:,i] = likelihood_1d[:,i] - L1d_master
    L1d = likelihood_1d.T
    U,s,Vt = np.linalg.svd(L1d,full_matrices=False)
    V = Vt.T
    ind = np.argsort(s)[::-1]
    ind = np.argsort(s)[::-1]
    U = U[:, ind]
    s = s[ind]
    V = V[:, ind]

    #for i in xrange(origShape[2]):
    #    likelihood_1d[:,i] = likelihood_1d[:,i] + L1d_master
  
    likelihood_pcomp = V.reshape(origShape)
    
    if doplot is True:
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.colors import LogNorm, Normalize
        if band is None:
            print "Must supply band (g,r,i,z,Y) in order to save PCA plots."
            stop
            
        pp = PdfPages('likelihood_pca_components-'+band+'.pdf')

        for i,thing in zip(xrange(s.size),s):
            fig,ax = plt.subplots(nrows=1,ncols=1,figsize = (6.,6.))
            im = ax.imshow( -likelihood_pcomp[:,:,i],origin='lower',cmap=plt.cm.Greys, extent = extent)#,vmin=-1,vmax=1)
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

def doLikelihoodPCAfit(pcaComp = None, master = None, eigenval = None, likelihood =None, n_component = 5, Lcut = 0.):

    # Perform least-squares: Find the best combination of master + pcaComps[:,:,0:n_component] that fits likelihood
    origShape = likelihood.shape
    #L1d = likelihood - master
    L1d = likelihood.reshape(likelihood.size)
    pca1d = pcaComp.reshape( ( likelihood.size, pcaComp.shape[-1]) )
    pcafit = pca1d[:,1:(n_component)]
    m1d = np.reshape(master,master.size)
    #allfit = np.hstack((m1d[:,None], pcafit) )
    allfit = pcafit
    coeff, resid, _, _  = np.linalg.lstsq(allfit, L1d)
    bestFit = np.dot(allfit,coeff)
    bestFit2d = bestFit.reshape(likelihood.shape)
    bestFit2d = bestFit2d + master
    bestFit2d[bestFit2d < Lcut] = 0.

    m_coeff, m_resid, _, _ = np.linalg.lstsq(allfit, m1d)
    m1d_fit = np.dot(allfit, m_coeff)
    m2d = np.reshape(m1d_fit,master.shape)
    
    return bestFit2d, m2d


def doInference(catalog = None, likelihood = None, obs_bins=None, truth_bins = None, tag = 'mag_auto',
                invType = 'basic', lambda_reg = 1e-3):


    N_real_obs, _  = np.histogram(catalog[tag], bins = obs_bins)
    A = likelihood.copy()
    if invType is 'basic':
        Ainv = np.linalg.pinv(A)
    if invType is 'tikhonov':
        Ainv = np.dot( np.linalg.pinv(np.dot(A.T, A) + lambda_reg * np.identity(N_real_obs.size) ), A.T)


        
    truth_bins_centers = (truth_bins[0:-1] + truth_bins[1:])/2.
    N_real_truth = np.dot(Ainv, N_real_obs)
    errors = N_real_truth*0.

    return N_real_truth, errors, truth_bins_centers
