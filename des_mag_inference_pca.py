#!/usr/bin/env python
import matplotlib as mpl
#mpl.use('Agg')

import argparse
import matplotlib.pyplot as plt
import cfunc
import lfunc
import mapfunc
import sys
import numpy as np
import healpy as hp


def makeMapPolygons(theMap = None, N_recon = None, N_recon_err=None, N_raw = None,
                       bin_centers = None, HEALConfig = None, vmin = -2., vmax = 2.):

    mapIndices = np.arange(hp.nside2npix(HEALConfig['out_nside']))
    useInds = mapIndices[theMap[mapIndices] != hp.UNSEEN]
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    nside = hp.npix2nside(theMap.size)

    vertices = np.zeros( (useInds.size, 4, 2) )
    print "Building polygons from HEALPixel map."

    for i, hpInd in zip(xrange(useInds.size), useInds):
        if (i % 10) == 0:
            print "polygon: "+str(i)+" of "+str(useInds.size-1)
        corners = hp.vec2ang( np.transpose(hp.boundaries(nside,hpInd,nest=True) ) )
        vertices[i,:,0] = corners[1] * 180/np.pi
        vertices[i,:,1] = 90 - corners[0] * 180/np.pi
    coll = PolyCollection(vertices, array = theMap, cmap = plt.cm.gray, edgecolors='none',alpha=0.75)
    coll.set_clim(vmin=vmin, vmax=vmax)

    return coll, vertices

def mapHistogramPlots(theMap =None,poly = None, N_recon = None, N_recon_err=None, N_raw = None, band=None,
                       bin_centers = None, HEALConfig = None, vertices=None, N_truth = None, catalog = None):
    mapIndices = np.arange(hp.nside2npix(HEALConfig['out_nside']))
    useInds = mapIndices[theMap[mapIndices] != hp.UNSEEN]
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    from matplotlib.colors import colorConverter
    from copy import deepcopy
    nside = hp.npix2nside(theMap.size)


    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('maps_and_histograms-'+band+'.pdf')
    for i, hpInd in zip(xrange(useInds.size), useInds):
        print "Writing map polygons for HEALPixel "+str(i)
        title = 'HPInd: '+str(hpInd)+' map '
        fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=(12.,6.))
        ax1.plot(catalog['ra'],catalog['dec'],',',color='b')
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        thisPoly = deepcopy(poly)
        ax1.add_collection(thisPoly)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        
        ax1.set_title(title)
        vptsx = np.hstack( (vertices[i,:,0], vertices[i,0,0]))
        vptsy = np.hstack( (vertices[i,:,1], vertices[i,0,1]))
        ax1.plot(vptsx, vptsy,color='red')


        
        ax1.set_xlabel('ra')
        ax1.set_ylabel('dec')
        ax2.plot(bin_centers, N_raw[:,i],label='raw counts')
        ax2.errorbar(bin_centers, N_recon[:,i], N_recon_err[:,i],fmt='.', label='reconstruction')
        ax2.plot(bin_centers, N_truth, '-', label='COSMOS, scaled')
        ax2.set_xlabel('magnitude')
        ax2.set_ylabel('Number')
        ax2.set_yscale('log')
        ax2.legend(loc='best')
        ax2.set_ylim([1e-4,1])
        pp.savefig(fig)
        
    pp.close()


def populateMap(catalog = None, sim=None, truth=None, truthMatched=None, band=None,
                HEALConfig = None, pcaBasis = None, obs_bins = None, truth_bins = None,
                magrange =[22.5,24.0], doplot = False, n_component = 4, prior = None):
    useInds = np.unique(catalog['HEALIndex'])
    mapIndices = np.arange(hp.nside2npix(HEALConfig['out_nside']))
    theMap = np.zeros(mapIndices.size) + hp.UNSEEN
    theMapErr = np.zeros(mapIndices.size) + hp.UNSEEN
    theMapRaw = np.zeros(mapIndices.size) + hp.UNSEEN
    theMapObs =  np.zeros(mapIndices.size) + hp.UNSEEN

    pcaRecon = np.zeros( (truth_bins.size-1, useInds.size) )
    pcaReconErr = np.zeros( (truth_bins.size-1, useInds.size) )
    rawRecon = np.zeros( (truth_bins.size-1, useInds.size) )
    rawReconErr = np.zeros( (truth_bins.size-1, useInds.size) )
    coeff = np.zeros( (n_component, useInds.size) )
    
    # Which bins do we sum over in the end?
    truth_bin_indices = np.arange(truth_bins.size-1)[np.where((truth_bins[0:-1] <= np.max(magrange)) & 
                                                              (truth_bins[1:] >= np.min(magrange) ) )]
    obs_bin_indices = np.arange(obs_bins.size-1)[np.where((obs_bins[0:-1] <= np.max(magrange)) & 
                                                              (obs_bins[1:] >= np.min(magrange) ) )]


    N_truth,_  = np.histogram( truth['mag'], bins = truth_bins )
    N_truth = N_truth * 1./( truth.size )
    
    print "Starting mapmaking..."
    for i, hpInd in zip(xrange(useInds.size), useInds):
        thisSim = sim[sim['HEALIndex'] == hpInd]
        thisMatched = truthMatched[sim['HEALIndex'] == hpInd]
        thisTruth = truth[truth['HEALIndex'] == hpInd]
        thisDES = catalog[catalog['HEALIndex'] == hpInd]

        print "for pixel "+str(i)+' of '+str(useInds.size-1)+":"
        print "   making likelihood matrix..."
        Lraw = lfunc.makeLikelihoodMatrix( sim=thisSim, 
                                           truth=thisTruth, 
                                           truthMatched =thisMatched,
                                           obs_bins = obs_bins, truth_bins = truth_bins, 
                                           simTag = 'mag_auto', truthTag = 'mag')
        print "   fitting likelihood to largest PCA components..."
        Lpca, thisCoeff = lfunc.doLikelihoodPCAfit(pcaComp = pcaBasis,
                                                likelihood = Lraw, 
                                                n_component = n_component, 
                                                Lcut = 1e-3)
        print "   performing regularized inversion..."
        this_recon, this_err , _ = lfunc.doInference(catalog = thisDES, likelihood = Lpca, 
                                                     obs_bins=obs_bins, truth_bins = truth_bins,
                                                     lambda_reg = 1e-2, invType = 'tikhonov',
                                                     prior = prior)

    
        pcaRecon[:, i] = this_recon.copy() * 1./(thisTruth.size)
        pcaReconErr[:,i] = this_err.copy() * 1./(thisTruth.size)
        coeff[:,i] = thisCoeff
        
        theMap[hpInd] = np.sum(this_recon[truth_bin_indices]) * 1. / (thisTruth.size)
        theMapErr[hpInd] = np.sqrt(np.sum(this_err[truth_bin_indices]**2)) * 1./thisTruth.size

        N_obs,_ = np.histogram(thisDES['mag_auto'], bins=truth_bins)
        N_obs = N_obs * 1./thisTruth.size
        rawRecon[:,i] = N_obs
        
        theMapObs[hpInd] = np.sum(N_obs[obs_bin_indices])
        truth_bin_centers = (truth_bins[0:-1] + truth_bins[1:])/2.
        print " HEALPixel, observed number,  reconstructed number ) = ", hpInd, theMapObs[hpInd], theMap[hpInd]

    print "sub-sampling catalog for visualization purposes..."
    sample = catalog[np.random.choice(catalog.size, size=10000,replace=False)]
    print "Making map polygons"
    poly, vertices = makeMapPolygons(theMap = theMap, N_recon = pcaRecon, N_recon_err=pcaReconErr, N_raw = rawRecon,
                           bin_centers = truth_bin_centers, HEALConfig = HEALConfig, vmin = -2., vmax = 2.)
    print "Building combined map+histogram plots"
    mapHistogramPlots(poly = poly, theMap = theMap, N_recon = pcaRecon, N_recon_err=pcaReconErr, N_raw = rawRecon, band=band,
                       bin_centers = truth_bin_centers, HEALConfig = HEALConfig, vertices=vertices, N_truth = N_truth,catalog=sample)
    
    return theMap, theMapObs, coeff


def globalReconstructionPlots(des = None, likelihood = None, obs_bins = None, truth_bins= None,
                         obsTag= 'mag_auto', truthTag = 'mag', truth = None, band = None):
    N_mcmc, errMCMC , _ = lfunc.doInference(catalog = des, likelihood = likelihood,
                                         obs_bins=obs_bins, truth_bins = truth_bins, invType = 'mcmc')
    N_direct,errDirect, _  = lfunc.doInference(catalog = des, likelihood = likelihood,
                                         obs_bins=obs_bins, truth_bins = truth_bins, invType = 'tikhonov')
    N_obs, _ = np.histogram( des[obsTag], bins = obs_bins )
    N_obs = N_obs * 1./truth.size
    N_mcmc = N_mcmc * 1./truth.size
    N_direct = N_direct * 1./truth.size
    errMCMC = errMCMC * 1./truth.size
    errDirect = errDirect * 1./truth.size
    N_truth,_ = np.histogram( truth[truthTag], bins = truth_bins  )
    N_truth = N_truth * 1./truth.size
    obs_bin_centers   = ( obs_bins[0:-1]   + obs_bins[1:]   ) / 2.
    truth_bin_centers = ( truth_bins[0:-1] + truth_bins[1:] ) / 2.
    fig, ax = plt.subplots()
    ax.errorbar(truth_bin_centers, N_mcmc, errMCMC , label = 'reconstr.')
    ax.errorbar(truth_bin_centers, N_direct, errDirect , label = 'tikh')    
    ax.plot(truth_bin_centers, N_truth, label = 'truth')
    ax.plot(obs_bin_centers, N_obs, label = 'obs')
    ax.set_ylim([1e-4,1])
    ax.legend(loc='best')
    ax.set_yscale('log')
    fig.savefig("global_reconstruction-"+band+".png")
    return N_mcmc, N_direct
    

def main(argv):
    parser = argparse.ArgumentParser(description = 'Perform magnitude distribution inference on DES data.')
    parser.add_argument('filter',help='filter name',choices=['g','r','i','z','Y'])
    parser.add_argument("-r","--reload",help='reload catalogs from DESDB', action="store_true")
    parser.add_argument("-np","--no_pca",help='Do not use pca-smoothed likelihoods in reconstruction.', action="store_true")
    parser.add_argument("-ns","--nside",help='HEALPix nside for mapmaking',choices=['64','128','256','512','1024'])
    args = parser.parse_args(argv[1:])
    band = args.filter
    no_pca = args.no_pca
    if args.nside is not None:
        nside  = int(args.nside)
    else:
        nside = 64


    n_component = 8
    # Get the catalogs.
    print "performing inference in band: "+args.filter
    des, sim, truthMatched, truth, HEALConfig= cfunc.getCleanCatalogs(reload = args.reload, band = args.filter, nside=nside)
    truth = truth[truth['mag'] > 15.]
    des =des[des['mag_auto'] > 15.]
    keep = (sim['mag_auto'] > 15.) & (truthMatched['mag'] > 0.)
    sim = sim[keep]
    truthMatched = truthMatched[keep]

    # Find the pca components for the chosen HEALPixelization.
    obsBins = lfunc.chooseBins(des,tag='mag_auto', binsize = 0.5)
    truthBins = lfunc.chooseBins(truth,tag='mag', binsize=0.5)
    print "Building master matrix"
    LmasterRaw = lfunc.makeLikelihoodMatrix( sim=sim, truth=truth, truthMatched =truthMatched, Lcut = 0.,
                          obs_bins = obsBins, truth_bins = truthBins, simTag = 'mag_auto', truthTag = 'mag')
    print "Building HEALPixel likelihood matrices for pca basis."
    out_nside = HEALConfig['out_nside']
    HEALConfig['out_nside'] = 64
    des = cfunc.HealPixifyCatalogs(catalog=des, healConfig=HEALConfig, ratag='ra', dectag = 'dec')
    sim = cfunc.HealPixifyCatalogs(catalog=sim, healConfig=HEALConfig, ratag='ra', dectag = 'dec')
    truth = cfunc.HealPixifyCatalogs(catalog=truth, healConfig=HEALConfig, ratag='ra', dectag = 'dec')
    truthMatched = cfunc.HealPixifyCatalogs(catalog=truthMatched, healConfig=HEALConfig, ratag='ra', dectag = 'dec')

    
    Likelihoods, HEALPixels, masterLikelihood, truth_bins, obs_bins = lfunc.getAllLikelihoods(truth=truth, 
                                                                                              sim=sim,
                                                                                              truthMatched = truthMatched,
                                                                                              healConfig=HEALConfig,
                                                                                              obs_bins = obsBins, truth_bins = truthBins,
                                                                                              ncut = 0.,
                                                                                              doplot = False)

    truth_bin_centers = (truth_bins[0:-1] + truth_bins[1:])/2.
    obs_bin_centers = (obs_bins[0:-1] + obs_bins[1:])/2.
    extent= [truth_bin_centers[0],truth_bin_centers[-1],obs_bin_centers[0],obs_bin_centers[-1]]
    
    print "Finding pca components from HEALPixel stack."
    Lpca, pcaEigen = lfunc.likelihoodPCA(likelihood = Likelihoods, doplot=False, 
                                         band=band, extent = extent)
    print "Re-fitting primary principal components to master likelihood"
    LmasterPCA, coeffMaster = lfunc.doLikelihoodPCAfit(pcaComp = Lpca,
                                            likelihood =masterLikelihood, n_component = n_component, Lcut = 1e-4)
    N_prior_mcmc, N_prior_tik = globalReconstructionPlots(des = des, likelihood = LmasterPCA, obs_bins = obsBins, truth_bins=truthBins,
                                                          obsTag= 'mag_auto', truthTag = 'mag', truth = truth, band = band)
    fig, ( ax1, ax2 ) = plt.subplots( nrows=1, ncols=2, figsize=(14 , 7) )
    from matplotlib.colors import LogNorm
    im1 = ax1.imshow(np.arcsinh(masterLikelihood/1e-4), origin='lower',cmap=plt.cm.Greys, extent=extent)
    ax1.set_xlabel('truth mag.')
    ax1.set_ylabel('measured mag.')
    ax1.set_title('raw global \n likelihood')
    fig.colorbar(im1,ax=ax1)
    im2 = ax2.imshow(np.arcsinh(LmasterPCA/1e-4), origin='lower',cmap=plt.cm.Greys, extent=extent)
    ax2.set_xlabel('truth mag.')
    ax2.set_ylabel('measured mag.')
    ax2.set_title('pca-smoothed \n global likelihood')
    fig.colorbar(im2,ax=ax2)
    fig.savefig("pca-global_likelihood-"+band+".png")

    print "Re-HEALPixifying catalogs for map."
    HEALConfig['out_nside'] = out_nside
    des = cfunc.HealPixifyCatalogs(catalog=des, healConfig=HEALConfig, ratag='ra', dectag = 'dec')
    sim = cfunc.HealPixifyCatalogs(catalog=sim, healConfig=HEALConfig, ratag='ra', dectag = 'dec')
    truth = cfunc.HealPixifyCatalogs(catalog=truth, healConfig=HEALConfig, ratag='ra', dectag = 'dec')
    truthMatched = cfunc.HealPixifyCatalogs(catalog=truthMatched, healConfig=HEALConfig, ratag='ra', dectag = 'dec')
    
    print "Populating maps with HEALPixel reconstructions"
    theMap, theMapObs, coeff = populateMap(catalog = des, sim=sim, truth=truth, truthMatched=truthMatched,
                                           HEALConfig = HEALConfig, pcaBasis = Lpca, obs_bins = obsBins,
                                           prior = N_prior_tik,
                                           truth_bins = truthBins, n_component = n_component, band=band)
    
    fig,ax = plt.subplots()
    pix = theMap[theMap != hp.UNSEEN]
    dpix = pix - np.median(pix)
    for i in np.arange(n_component):
        ax.plot(coeff[i,:],dpix,'.',label=str(i),alpha=0.5)
    ax.legend(loc='best')
    ax.set_ylim([-1,1])
    fig.savefig("pcaCoeffInfluence-"+band+".png")
    
    print "Building (and writing) HEALPixel reconstruction visualizations..."    
    mapfunc.visualizeHealPixMap(theMap, nest=True, title= "pca-mapRecon-"+band, vmin=-1, vmax = 1)
    deltaMap = theMap.copy()
    seen = (theMap != hp.UNSEEN)
    deltaMap[seen] = theMap[seen] / np.median(theMap[seen]) - 1.
    mapfunc.visualizeHealPixMap(deltaMap, nest=True, title= "pca-mapReconDelta-"+band, vmin = -1, vmax=1)

    mapfunc.visualizeHealPixMap(theMapObs, nest=True, title= "mapRecon-"+band, vmin = -1., vmax = 1.)
    mapfunc.visualizeHealPixMap(theMapObs, nest=True, title= "mapRecon_zoomed-"+band, vmin = -.1, vmax = .1)

    deltaMapObs = theMapObs.copy()
    deltaMapObs[seen] = theMapObs[seen]/np.median(theMapObs[seen]) - 1.
    mapfunc.visualizeHealPixMap(deltaMapObs, nest=True, title="mapObsDelta-"+band, vmin = -1., vmax = 1.)
    print "Done."


if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
