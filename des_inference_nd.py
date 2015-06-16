#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')

import argparse
import matplotlib.pyplot as plt
import cfunc
import lfunc
import mapfunc
import sys
import numpy as np
import healpy as hp
import esutil

def star_galaxy_inference(sim=None,des=None,truth=None,truthMatched=None, band=None):
    # Choose the quantities to reconstruct.
    truthTags = ('mag','objtype')
    truthMagBins = np.linspace(15,24.5,25)#lfunc.chooseBins(catalog = truthMatched, tag = 'mag')
    truthTypeBins = np.array( [ 0,2, 4])
    truthBins = [truthMagBins, truthTypeBins]
    obsTags = ('mag_auto','modtype')
    obsMagBins = np.linspace(15,24.5,25) #lfunc.chooseBins(catalog = des, tag = 'mag_auto')
    obsTypeBins = np.array( [ 0,2, 4,6] )
    obsBins = [obsMagBins, obsTypeBins]
    
    
    # Measure the global likelihood function.
    print "Making global likelihood function."
    L = lfunc.makeLikelihoodMatrix( sim= sim, truth=truth, truthMatched = truthMatched, Lcut = 0., ncut = 0.,
                                    obs_bins = obsBins, truth_bins = truthBins, simTag = obsTags, truthTag = truthTags)
    fig, ax = plt.subplots()
    im = ax.imshow(np.arcsinh(L/0.001), origin='lower', cmap=plt.cm.Greys)
    ax.set_xlabel('truth mag/stellarity')
    ax.set_ylabel('obs mag/stellarity')
    fig.savefig("des_mag-stellarity_likelihood-"+band+".png")
    fig.colorbar(im,ax = ax)
    fig.show()
    
    # Do the inversion inference.
    N_sim_obs, _ = np.histogramdd([des['mag_auto'],des['modtype']], bins = obsBins)
    N_obs_plot,_ = np.histogramdd([des['mag_auto'],des['modtype']], bins = truthBins)
    N_sim_truth, _ = np.histogramdd([truth['mag'], truth['objtype']], bins= truthBins)

    N_est, errs, _ = lfunc.doInference( catalog = des, likelihood = L, obs_bins = obsBins, truth_bins = truthBins,
                                          tag = obsTags, lambda_reg = 0.001)

    fig,ax = plt.subplots()
    # For plotting purposes, clip the errors to exclude values <= 0
    errs = np.clip( errs, 0, N_est-1e-6)
    ax.errorbar( (truthMagBins[0:-1] + truthMagBins[1:])/2. , N_est[:,0], errs[:,0],linestyle= '--', label='galaxies (est.)')
    ax.errorbar( (truthMagBins[0:-1] + truthMagBins[1:])/2. , N_est[:,1], errs[:,0],linestyle= '--', label = 'stars (est)')
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:] )/2. , N_obs_plot[:,0], '.', label='galaxies (obs.)')
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:] )/2. , N_obs_plot[:,1], '.', label='stars (obs.)')
    N_gal_hist, _ = np.histogram(truth['mag'][truth['objtype'] == 1],bins=truthMagBins)
    N_star_hist, _ = np.histogram(truth['mag'][truth['objtype'] == 3], bins=truthMagBins)
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:])/2. , N_gal_hist  , label='galaxies (true)')
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:])/2., N_star_hist, label='stars (true)')
    ax.legend(loc='best')
    ax.set_yscale('log')
    ax.set_ylim([1,2.*np.max(N_gal_hist)])
    fig.savefig("nd-reconstruction_mag_stellarity-"+band+".png")
    fig.show()


def size_stellarity_inference(sim=None,des=None,truth=None,truthMatched=None, band=None):
    # Choose the quantities to reconstruct.
    
    sim[sim['modtype'] == 5]['modtype'] = 1
    truthTags = ('radius','objtype')
    truthSizeBins = np.insert(-0.2, 1, np.logspace(-1, 3., 30)) 
    truthTypeBins = np.array( [ 0,2, 4])
    truthBins = [truthSizeBins, truthTypeBins]
    obsTags = ('flux_radius','modtype')
    obsSizeBins = np.insert(-0.2, 1, np.logspace(-.5, 3., 30)) 
    obsTypeBins = np.array( [ 0,2, 4,6] )
    obsBins = [obsSizeBins, obsTypeBins]
    
    
    # Measure the global likelihood function.
    print "Making global likelihood function."
    L = lfunc.makeLikelihoodMatrix( sim= sim, truth=truth, truthMatched = truthMatched, Lcut = 0., ncut = 0.,
                                    obs_bins = obsBins, truth_bins = truthBins, simTag = obsTags, truthTag = truthTags)
    fig, ax = plt.subplots()
    im = ax.imshow(np.arcsinh(L/0.001), origin='lower', cmap=plt.cm.Greys, interpolation='none')
    ax.set_xlabel('truth size/stellarity')
    ax.set_ylabel('obs size/stellarity')
    fig.savefig("des_size-stellarity_likelihood-"+band+".png")
    fig.colorbar(im,ax = ax)
    fig.show()
    
    # Do the inversion inference.
    N_sim_obs, _ = np.histogramdd([des['flux_radius'],des['modtype']], bins = obsBins)
    N_obs_plot,_ = np.histogramdd([des['flux_radius'],des['modtype']], bins = truthBins)
    N_sim_truth, _ = np.histogramdd([truth['radius'], truth['objtype']], bins= truthBins)

    N_est, errs, _ = lfunc.doInference( catalog = des, likelihood = L, obs_bins = obsBins, truth_bins = truthBins,
                                          tag = obsTags, lambda_reg = 0.001)

    fig,ax = plt.subplots()
    # For plotting purposes, clip the errors to exclude values <= 0
    errs = np.clip( errs, 0, N_est-1e-6)
    ax.errorbar( truthSizeBins[1:] , N_est[:,0], errs[:,0],linestyle= '--', label='galaxies (est.)')
    ax.plot( truthSizeBins[1:] , N_obs_plot[:,0], '.-', label='galaxies (obs.)')
    ax.plot(  truthSizeBins[1:] , N_obs_plot[:,1], '.-', label='stars (obs.)')
    N_gal_hist, _ = np.histogram(truth['radius'][truth['objtype'] == 1],bins=truthSizeBins)
    N_star_hist, _ = np.histogram(truth['radius'][truth['objtype'] == 3], bins=truthSizeBins)
    ax.plot(  truthSizeBins[1:] , N_gal_hist  , label='galaxies (true)')
    ax.legend(loc='best')
    ax.set_yscale('symlog')
    ax.set_xscale('log')
    #ax.set_xlim([1e-2,100])
    ax.set_ylim([1e3,2.*np.max(N_gal_hist)])
    fig.savefig("nd-reconstruction_size_stellarity-"+band+".png")
    fig.show()


    
def mag_size_inference(sim=None,des=None,truth=None,truthMatched=None, band=None):
    
    truthTags = ('mag','radius')
    obsTags = ('mag_auto','flux_radius')

    obsMagBins = np.linspace(15,24.,30)
    truthMagBins = np.linspace(15,25.,30)
    obsSizeBins = np.insert(-0.2, 1, np.logspace(-.3, 3., 40)) 
    truthSizeBins = np.insert(-0.2, 1, np.logspace(-2, 3., 40)) 

    obsMagBins_cen = ( obsMagBins[0:-1] + obsMagBins[1:] )/2.
    truthMagBins_cen = ( truthMagBins[0:-1] + truthMagBins[1:] ) /2.
    obsSizeBins_cen = ( obsSizeBins[0:-1] + obsSizeBins[1:] ) / 2.
    truthSizeBins_cen = ( truthSizeBins[0:-1] + truthSizeBins[1:] ) /2.

    
    truthBins = [truthMagBins, truthSizeBins]
    obsBins = [obsMagBins, obsSizeBins]


    L = lfunc.makeLikelihoodMatrix(sim= sim, truth=truth, truthMatched = truthMatched,
                                     obs_bins = obsBins, truth_bins = truthBins, simTag = obsTags,
                                     truthTag = truthTags)
    fig, ax2 = plt.subplots(nrows = 1, ncols = 1)
    ax2.imshow(np.arcsinh(L/0.001), origin='lower', cmap=plt.cm.Greys)
    ax2.set_xlabel('truth mag/size')
    ax2.set_ylabel('obs mag/size')
    fig.savefig("nd-likelihood_test-mag_size"+band+".png")
    fig.show()

    N_sim_truth, _ = np.histogramdd([truth['mag'], truth['radius']], bins = truthBins )
    N_obs_plot,  _ = np.histogramdd([des['mag_auto'],des['flux_radius']], bins = truthBins)
    N_est, errs, _ = lfunc.doInference( catalog = des, likelihood = L, obs_bins = obsBins, truth_bins = truthBins,
                                          tag = obsTags, lambda_reg = 0.00001, prior = N_sim_truth)

    
    A = L.copy()
    
    fig, ( (ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(19,13))
    ax1.set_xlabel('size (arcsec)')
    ax1.set_ylabel('mag_auto')
    im1 = ax1.imshow(np.arcsinh(N_sim_truth/0.01),origin='lower',cmap=plt.cm.Greys, extent = [truthSizeBins_cen[0],truthSizeBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto')
    ax1.set_title('truth')
    im2 = ax2.imshow(np.arcsinh(N_est/0.01), origin='lower',cmap=plt.cm.Greys,vmin=0., extent = [truthSizeBins_cen[0],truthSizeBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto')
    ax2.set_title('reconstruction')

    im3 = ax3.imshow(np.arcsinh(N_obs_plot/0.01),origin='lower',cmap=plt.cm.Greys, extent = [truthSizeBins_cen[0],truthSizeBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto')
    ax3.set_title('uncorrected observations')
    im4 = ax4.imshow(np.arcsinh(( N_est / N_sim_truth-1 )),origin='lower',cmap=plt.cm.seismic, extent = [truthSizeBins_cen[0],truthSizeBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto', vmax=1, vmin = -1)
    ax4.set_title('reconstr. / truth -1 \n (frac. residuals)')
    im5 = ax5.imshow( ( ( N_est - N_sim_truth ) / errs ), origin='lower',cmap=plt.cm.seismic, extent = [truthSizeBins_cen[0],truthSizeBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto', vmax=1, vmin = -1)
    ax5.set_title( '( reconstr. - truth) / err. \n (significance of residuals)')
    im6 = ax6.imshow(np.abs(( N_est / errs )),origin='lower',cmap=plt.cm.Greys, extent = [truthSizeBins_cen[0],truthSizeBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto',vmin=0,vmax=3)
    ax6.set_title('reconstr. / errors \n ( significance )')
    fig.colorbar(im1,ax=ax1)
    fig.colorbar(im2,ax=ax2)
    fig.colorbar(im3,ax=ax3)
    fig.colorbar(im4,ax=ax4)
    fig.colorbar(im5,ax=ax5)
    fig.colorbar(im6,ax=ax6)
    fig.savefig('nd-des-recon-mag_size-'+band+'.png')
    fig.show()


def populateMapStellarity(catalog = None, sim=None, truth=None, truthMatched=None, band=None,
                HEALConfig = None, pcaBasis = None, obs_bins = None, truth_bins = None,
                magrange =[19.5,23.], stellarity = 0, doplot = False, n_component = 8, prior = None):
    
    useInds = np.unique(catalog['HEALIndex'])
    mapIndices = np.arange(hp.nside2npix(HEALConfig['out_nside']))
    theMap = np.zeros(mapIndices.size) + hp.UNSEEN
    theMapErr = np.zeros(mapIndices.size) + hp.UNSEEN
    theMapRaw = np.zeros(mapIndices.size) + hp.UNSEEN
    theMapObs =  np.zeros(mapIndices.size) + hp.UNSEEN

    pcaRecon = np.zeros( (truth_bins[0].size-1, useInds.size) )
    pcaReconErr = np.zeros( (truth_bins[0].size-1, useInds.size) )
    rawRecon = np.zeros( (truth_bins[0].size-1, useInds.size) )
    rawReconErr = np.zeros( (truth_bins[0].size-1, useInds.size) )
    coeff = np.zeros( (n_component, useInds.size) )
    
    N_truth,_  = np.histogramdd( [ truth['mag'], truth['objtype'] ], bins = truth_bins )
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
                                           simTag = ['mag_auto','modtype'], truthTag = ['mag','objtype'])
        print "   fitting likelihood to largest PCA components..."
        Lpca, thisCoeff = lfunc.doLikelihoodPCAfit(pcaComp = pcaBasis,
                                                   likelihood = Lraw,
                                                   n_component = n_component,
                                                   Lcut = 1e-3, Ntot = thisSim.size)
        print "   performing regularized inversion..."
        scaledPrior = prior * thisTruth.size

        truth_mag_bin_indices = ( truth_bins[0] >= np.min(magrange) ) & ( truth_bins[0] <= np.max(magrange))
        obs_mag_bin_indices   = ( obs_bins[0] >= np.min(magrange) ) & ( obs_bins[0] <= np.max(magrange))
        
        this_recon, this_err , _ = lfunc.doInference(catalog = thisDES, likelihood = Lpca, 
                                                     obs_bins=obs_bins, truth_bins = truth_bins,
                                                     lambda_reg = 1e-2, invType = 'tikhonov',
                                                     tag = ['mag_auto','modtype'],
                                                     prior = scaledPrior, priorNumber = sim.size)
    
        
        pcaRecon[:, i] = this_recon[:,stellarity].copy() * 1./(thisTruth.size)
        pcaReconErr[:,i] = this_err[:,stellarity].copy() * 1./(thisTruth.size)
        coeff[:,i] = thisCoeff
        wt = 1./this_err[truth_mag_bin_indices, stellarity]**2
        theMap[hpInd] = np.sum(this_recon[truth_mag_bin_indices, stellarity] * wt) / np.sum(wt) * 1. / (thisTruth.size)
        theMapErr[hpInd] = np.sqrt(np.sum(this_err[truth_mag_bin_indices, stellarity]**2)) * 1./thisTruth.size

        N_obs,_ = np.histogramdd([thisDES['mag_auto'], thisDES['modtype']], bins=truth_bins)
        N_obs = N_obs * 1./thisTruth.size
        rawRecon[:,i] = N_obs[:,stellarity]
        
        theMapObs[hpInd] = np.sum(N_obs[obs_mag_bin_indices, stellarity])
        truth_bin_centers = (truth_bins[0][0:-1] + truth_bins[0][1:])/2.
            
        print " HEALPixel, observed number,  reconstructed number ) = ", hpInd, theMapObs[hpInd], theMap[hpInd]

    
    return theMap, theMapObs, coeff
    

def star_galaxy_maps(sim=None,des=None,truth=None,truthMatched=None, band=None, healConfig = None, nside_pca = 64):
    
    truthTags = ('mag','objtype')
    truthMagBins = np.linspace(15,24.5,25)#lfunc.chooseBins(catalog = truthMatched, tag = 'mag')
    truthTypeBins = np.array( [ 0,2, 4])
    truthBins = [truthMagBins, truthTypeBins]
    obsTags = ('mag_auto','modtype')
    obsMagBins = np.linspace(15,24,25) #lfunc.chooseBins(catalog = des, tag = 'mag_auto')
    obsTypeBins = np.array( [ 0,2, 4,6] )
    obsBins = [truthMagBins, truthTypeBins]
    n_component = 8


    print "Building HEALPixel likelihood matrices for pca basis."
    out_nside = healConfig['out_nside']
    healConfig['out_nside'] = nside_pca
    des = cfunc.HealPixifyCatalogs(catalog=des, healConfig=healConfig, ratag='ra', dectag = 'dec')
    sim = cfunc.HealPixifyCatalogs(catalog=sim, healConfig=healConfig, ratag='ra', dectag = 'dec')
    truth = cfunc.HealPixifyCatalogs(catalog=truth, healConfig=healConfig, ratag='ra', dectag = 'dec')
    truthMatched = cfunc.HealPixifyCatalogs(catalog=truthMatched, healConfig=healConfig, ratag='ra', dectag = 'dec')

    N_sim_truth, _ = np.histogramdd([truth['mag'], truth['objtype']], bins = truthBins)
    N_sim_truth = N_sim_truth * 1./truth.size
    
    Likelihoods, HEALPixels, masterLikelihood, truth_bins, obs_bins = lfunc.getAllLikelihoods(truth=truth, 
                                                                                              sim=sim,
                                                                                              truthMatched = truthMatched,
                                                                                              healConfig=healConfig,
                                                                                              obs_bins = obsBins,
                                                                                              truth_bins = truthBins,
                                                                                              ncut = 0.,
                                                                                              obsTag = obsTags,
                                                                                              truthTag = truthTags,
                                                                                              getBins = True,
                                                                                              doplot = True)
    Lpca, pcaEigen = lfunc.likelihoodPCA(likelihood = Likelihoods, doplot=True, band=band, extent = None)
    print "Re-fitting primary principal components to master likelihood"
    LmasterPCA, coeffMaster = lfunc.doLikelihoodPCAfit(pcaComp = Lpca,
                                                       likelihood =masterLikelihood,
                                                       n_component = n_component,
                                                       Lcut = 1e-4)
    print "Re-HEALPixifying catalogs for map."
    healConfig['out_nside'] = out_nside
    des = cfunc.HealPixifyCatalogs(catalog=des, healConfig=healConfig, ratag='ra', dectag = 'dec')
    sim = cfunc.HealPixifyCatalogs(catalog=sim, healConfig=healConfig, ratag='ra', dectag = 'dec')
    truth = cfunc.HealPixifyCatalogs(catalog=truth, healConfig=healConfig, ratag='ra', dectag = 'dec')
    truthMatched = cfunc.HealPixifyCatalogs(catalog=truthMatched, healConfig=healConfig, ratag='ra', dectag = 'dec')

    theMap, theMapObs, coeff = populateMapStellarity(catalog = des, sim= sim, truth=truth, truthMatched=truthMatched, band=band,
                                                     HEALConfig = healConfig, pcaBasis = Lpca, obs_bins = obsBins, truth_bins = truthBins,
                                                     magrange =[22.5,24.], stellarity = 0,  n_component = 8,
                                                     prior = N_sim_truth)
    seen = theMap != hp.UNSEEN
    deltaMap = theMap * 0. + hp.UNSEEN
    deltaObs = theMapObs * 0. + hp.UNSEEN
    deltaMap[seen] = theMap[seen] / np.median(theMap[seen]) - 1
    deltaObs[seen] = theMapObs[seen] / np.median(theMapObs[seen]) - 1
    mapfunc.visualizeHealPixMap(deltaMap, nest=True, title="delta-galaxies-recon-"+band, vmin = -1, vmax = 1)
    mapfunc.visualizeHealPixMap(deltaObs, nest=True, title="delta-galaxies-raw-"+band, vmin = -1, vmax = 1)
    
    theMap, theMapObs, coeff = populateMapStellarity(catalog = des, sim= sim, truth=truth, truthMatched=truthMatched, band=band,
                                                     HEALConfig = healConfig, pcaBasis = Lpca, obs_bins = obsBins, truth_bins = truthBins,
                                                     magrange =[22.5,24.], stellarity = 1,  n_component = 8,
                                                     prior = N_sim_truth)
    seen = theMap != hp.UNSEEN
    deltaMap = theMap * 0. + hp.UNSEEN
    deltaObs = theMapObs * 0. + hp.UNSEEN
    deltaMap[seen] = theMap[seen] / np.median(theMap[seen]) - 1
    deltaObs[seen] = theMapObs[seen] / np.median(theMapObs[seen]) - 1
    mapfunc.visualizeHealPixMap(deltaMap, nest=True, title="delta-stars-recon-"+band, vmin = -2, vmax = 2)
    mapfunc.visualizeHealPixMap(deltaObs, nest=True, title="delta-stars-raw-"+band, vmin = -2, vmax = 2)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    
    for i in xrange(8):
        coeffMap = np.zeros(theMap.size) + hp.UNSEEN
        coeffMap[seen] = coeff[i,:]
        mapfunc.visualizeHealPixMap(coeff0Map, nest=True, title='pca'+str(i)+'-'+band)
        esutil.io.write('pca'+str(i)+'-mapRecon-'+band+'.fits',coeffMap)
    
    

def color_match(sim1=None, des1=None, truth1=None, truthMatched1=None, band = None):
    des, sim, truthMatched, truth = cfunc.getStellarityCatalogs(reload = args.reload, band = args.filter)
    truth = truth[truth['mag'] > 15.]
    des =des[( des['mag_auto'] > 15.) & (des['flux_radius'] > 0) & (des['flux_radius'] < 10.)] 
    keep = (sim['mag_auto'] > 15.) & (truthMatched['mag'] > 0.) & (sim['flux_radius'] > 0) & (sim['flux_radius'] < 10.)
    sim = sim[keep]
    truthMatched = truthMatched[keep]

    
    des = esutil.numpy_util.add_fields(des,('color',des['mag_auto'].dtype))
    
    
    ind1, ind2 = esutil.numpy_util.match(sim1['balrog_index'], sim['balrog_index'])
    

    

def color_mag_inference(sim=None,des=None,truth=None,truthMatched=None, band1=None, band2 = None):
    
    truthTags = ('mag','color')
    obsTags = ('mag_auto','color_auto')

    obsMagBins = np.linspace(15,24.,20)
    truthMagBins = np.linspace(15,25.,20)
    obsColorBins = np.linspace(-3, 3., 20)
    truthColorBins = np.linspace(-3,3., 20)

    obsMagBins_cen = ( obsMagBins[0:-1] + obsMagBins[1:] )/2.
    truthMagBins_cen = ( truthMagBins[0:-1] + truthMagBins[1:] ) /2.
    obsColorBins_cen = ( obsColorBins[0:-1] + obsColorBins[1:] ) / 2.
    truthColorBins_cen = ( truthColorBins[0:-1] + truthColorBins[1:] ) /2.

    
    truthBins = [truthMagBins, truthColorBins]
    obsBins = [obsMagBins, obsColorBins]


    L = lfunc.makeLikelihoodMatrix(sim= sim, truth=truth, truthMatched = truthMatched,
                                     obs_bins = obsBins, truth_bins = truthBins, simTag = obsTags,
                                     truthTag = truthTags)
    fig, ax2 = plt.subplots(nrows = 1, ncols = 1)
    ax2.imshow(np.arcsinh(L/0.001), origin='lower', cmap=plt.cm.Greys)
    ax2.set_xlabel('truth mag/color')
    ax2.set_ylabel('obs mag/color')
    fig.savefig("nd-likelihood_test-mag_color.png")
    fig.show()

    N_sim_truth, _ = np.histogramdd([truth['mag'], truth['color']], bins = truthBins )
    N_obs_plot,  _ = np.histogramdd([des['mag_auto'],des['color_auto']], bins = truthBins)
    N_est, errs, _ = lfunc.doInference( catalog = des, likelihood = L, obs_bins = obsBins, truth_bins = truthBins,
                                          tag = obsTags, lambda_reg = 0.01, prior = N_sim_truth)

    
    A = L.copy()
    lambda_reg = 0.01
    
    fig, ( (ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(19,13))
    ax1.set_xlabel('color ('+band1+' - '+band2+')')
    ax1.set_ylabel('mag_auto')
    im1 = ax1.imshow(np.arcsinh(N_sim_truth/0.01),origin='lower',cmap=plt.cm.Greys, extent = [truthColorBins_cen[0],truthColorBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto')
    ax1.set_title('truth')
    im2 = ax2.imshow(np.arcsinh(N_est/0.01), origin='lower',cmap=plt.cm.Greys,vmin=0., extent = [truthColorBins_cen[0],truthColorBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto')
    ax2.set_title('reconstruction')

    im3 = ax3.imshow(np.arcsinh(N_obs_plot/0.01),origin='lower',cmap=plt.cm.Greys, extent = [truthColorBins_cen[0],truthColorBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto')
    ax3.set_title('uncorrected observations')
    im4 = ax4.imshow(np.arcsinh(( N_est / N_sim_truth-1 )),origin='lower',cmap=plt.cm.seismic, extent = [truthColorBins_cen[0],truthColorBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto', vmax=1, vmin = -1)
    ax4.set_title('reconstr. / truth -1 \n (frac. residuals)')
    im5 = ax5.imshow( ( ( N_est - N_sim_truth ) / errs ), origin='lower',cmap=plt.cm.seismic, extent = [truthColorBins_cen[0],truthColorBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto', vmax=1, vmin = -1)
    ax5.set_title( '( reconstr. - truth) / err. \n (significance of residuals)')
    im6 = ax6.imshow(np.abs(( N_est / errs )),origin='lower',cmap=plt.cm.Greys, extent = [truthColorBins_cen[0],truthColorBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto',vmin=0,vmax=3)
    ax6.set_title('reconstr. / errors \n ( significance )')
    fig.colorbar(im1,ax=ax1)
    fig.colorbar(im2,ax=ax2)
    fig.colorbar(im3,ax=ax3)
    fig.colorbar(im4,ax=ax4)
    fig.colorbar(im5,ax=ax5)
    fig.colorbar(im6,ax=ax6)
    fig.savefig('nd-reconstruction_des-mag_color-'+band+'.png')
    fig.show()


    
def main(argv):
    parser = argparse.ArgumentParser(description = 'Perform magnitude distribution inference on DES data.')
    parser.add_argument('filter',help='filter name',choices=['g','r','i','z','Y'])
    parser.add_argument("-r","--reload",help='reload catalogs from DESDB', action="store_true")
    parser.add_argument("-np","--no_pca",help='Do not use pca-smoothed likelihoods in reconstruction.', action="store_true")
    parser.add_argument("-ns","--nside",help='HEALPix nside for mapmaking',choices=['64','128','256','512','1024'])
    args = parser.parse_args(argv[1:])
    no_pca = args.no_pca
    if args.nside is not None:
        nside  = int(args.nside)
    else:
        nside = 256

    healConfig = cfunc.getHealConfig(out_nside = nside)
        
    des, sim, truthMatched, truth = cfunc.getStellarityCatalogs(reload = args.reload, band = args.filter)
    truth = truth[truth['mag'] > 15.]
    des =des[( des['mag_auto'] > 15.) & (des['flux_radius'] > 0) & (des['flux_radius'] < 100.)] 
    keep = (sim['mag_auto'] > 15.) & (truthMatched['mag'] > 0.) & (sim['flux_radius'] > 0) & (sim['flux_radius'] < 100.)
    sim = sim[keep]
    truthMatched = truthMatched[keep]
    #star_galaxy_inference(sim=sim,truth=truth,des=des,truthMatched=truthMatched, band=args.filter)
    #mag_size_inference(sim=sim,truth=truth,des=des,truthMatched=truthMatched, band=args.filter)
    size_stellarity_inference(sim=sim,truth=truth,des=des,truthMatched=truthMatched, band=args.filter)
    #star_galaxy_maps(sim=sim,truth=truth,des=des,truthMatched=truthMatched, band=args.filter, healConfig = healConfig)
    stop


if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
