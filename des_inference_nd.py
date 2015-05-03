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
    obsMagBins = np.linspace(15,24,25) #lfunc.chooseBins(catalog = des, tag = 'mag_auto')
    obsTypeBins = np.array( [ 0,2, 4,6] )
    obsBins = [truthMagBins, truthTypeBins]
    
    
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

    obsShape = N_sim_obs.shape
    truthShape = N_sim_truth.shape
    N_sim_obs_flat = np.ravel(N_sim_obs, order='F')
    N_sim_truth_flat = np.ravel(N_sim_truth, order='F')
    A = L.copy()
    lambda_reg = 0.0001
    Ainv = np.dot( np.linalg.pinv(np.dot(A.T, A) + lambda_reg * np.identity(N_sim_truth_flat.size ) ), A.T)
    #Ainv = np.linalg.pinv(A)
    N_est_flat = np.dot(Ainv, N_sim_obs_flat)
    N_est = np.reshape(N_est_flat, truthShape, order='F')
    fig,ax = plt.subplots()
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:])/2. , N_est[:,0],'--', label='galaxies (est.)')
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:])/2. , N_est[:,1],'--', label = 'stars (est)')
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:] )/2. , N_obs_plot[:,0], '.', label='galaxies (obs.)')
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:] )/2. , N_obs_plot[:,1], '.', label='stars (obs.)')
    N_gal_hist, _ = np.histogram(truth['mag'][truth['objtype'] == 1],bins=truthMagBins)
    N_star_hist, _ = np.histogram(truth['mag'][truth['objtype'] == 3], bins=truthMagBins)
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:])/2. , N_gal_hist  , label='galaxies (true)')
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:])/2., N_star_hist, label='stars (true)')
    ax.legend(loc='best')
    ax.set_yscale('log')
    ax.set_ylim([1,2.*np.max(N_sim_truth)])
    fig.savefig("nd-reconstruction_mag_stellarity-"+band+".png")
    fig.show()


def mag_size_inference(sim=None,des=None,truth=None,truthMatched=None, band=None):
    
    truthTags = ('mag','radius')
    obsTags = ('mag_auto','flux_radius')

    
    obsMagBins = np.linspace(15,24.,30)
    truthMagBins = np.linspace(15,25.,30)
    obsSizeBins = np.linspace(0., 5., 30)
    truthSizeBins = np.linspace(0.,5., 30)

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
    fig.savefig("nd-likelihood_test-mag_size.png")
    fig.show()

    N_sim_truth, _ = np.histogramdd([truth['mag'], truth['radius']], bins= truthBins)
    N_obs,   _ = np.histogramdd([des['mag_auto'],des['flux_radius']], bins = obsBins)
    N_obs_plot,  _ = np.histogramdd([des['mag_auto'],des['flux_radius']], bins = truthBins)
    truthShape = N_sim_truth.shape
    N_obs_flat = np.ravel(N_obs, order='F')
    N_sim_truth_flat = np.ravel(N_sim_truth, order='F')

    A = L.copy()
    lambda_reg = 0.01
    Ainv = np.dot( np.linalg.pinv(np.dot(A.T, A) + lambda_reg * np.identity(N_sim_truth_flat.size ) ), A.T)
    #N_est_flat = np.dot(Ainv, N_obs_flat)
    N_est_flat = N_sim_truth_flat + np.dot(Ainv, N_obs_flat - np.dot(L, N_sim_truth_flat) )
    N_est = np.reshape(N_est_flat, truthShape, order='F')
    
    fig, ( (ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(19,13))
    ax1.set_xlabel('size (arcsec)')
    ax1.set_ylabel('mag_auto')
    im1 = ax1.imshow(np.arcsinh(N_sim_truth/0.01),origin='lower',cmap=plt.cm.Greys, extent = [truthSizeBins_cen[0],truthSizeBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto')
    ax1.set_title('truth')
    im2 = ax2.imshow(np.arcsinh(N_est/0.01), origin='lower',cmap=plt.cm.Greys,vmin=0., extent = [truthSizeBins_cen[0],truthSizeBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto')
    ax2.set_title('reconstruction')

    im3 = ax3.imshow(np.arcsinh(N_obs_plot/0.01),origin='lower',cmap=plt.cm.Greys, extent = [truthSizeBins_cen[0],truthSizeBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto')
    ax3.set_title('uncorrected observations')
    im5 = ax4.imshow(np.arcsinh(( N_est / N_sim_truth-1 )),origin='lower',cmap=plt.cm.seismic, extent = [truthSizeBins_cen[0],truthSizeBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto', vmax=5, vmin = -5)
    ax5.set_title('reconstr. / truth -1 \n (frac. residuals)')

    
    im6 = ax6.imshow(np.arcsinh(( N_obs_plot / N_sim_truth-1 )),origin='lower',cmap=plt.cm.seismic, extent = [truthSizeBins_cen[0],truthSizeBins_cen[-1],truthMagBins_cen[0],truthMagBins_cen[-1]], aspect = 'auto', vmax=5, vmin = -5)
    ax6.set_title('observed / truth -1 \n (frac. residuals)')
    fig.colorbar(im1,ax=ax1)
    fig.colorbar(im2,ax=ax2)
    fig.colorbar(im3,ax=ax3)
    fig.colorbar(im5,ax=ax5)
    fig.colorbar(im6,ax=ax6)
    fig.savefig('nd-reconstruction_des-mag_size-'+band+'.png')
    fig.show()

    
def populateMapStellarity(catalog = None, sim=None, truth=None, truthMatched=None, band=None,
                HEALConfig = None, pcaBasis = None, obs_bins = None, truth_bins = None,
                magrange =[22.5,24.], stellarity = 0, doplot = False, n_component = 4, prior = None):
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
    truth_mag_bins = truth_bins[stellarity]
    obs_mag_Bins = obs_bins[stellarity]
    truth_bin_indices = np.arange(truth_bins.size-1)[np.where((truth_mag_bins[0:-1] <= np.max(magrange)) & 
                                                              (truth_mag_bins[1:] >= np.min(magrange) ) )]
    obs_bin_indices = np.arange(obs_bins.size-1)[np.where((obs_bins[0:-1] <= np.max(magrange)) & 
                                                              (obs_bins[1:] >= np.min(magrange) ) )]


    N_truth,_  = np.histogramdd( [ truth['mag'], truth['stellarity'] ], bins = truth_bins )
    N_truth = N_truth * 1./( truth.size )y


        
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
                                                   Lcut = 1e-3, Ntot = thisSim.size)
        print "   performing regularized inversion..."
        scaledPrior = prior * thisTruth.size
        this_N_truth, _ = np.histogramdd([thisTruth['mag'], thisTruth['radius']], bins= truthBins)
        this_N_obs,   _ = np.histogramdd([thisDES['mag_auto'],thisDES['flux_radius']], bins = obsBins)
        this_N_obs_plot,  _ = np.histogramdd([thisDES['mag_auto'],thisDES['flux_radius']], bins = truthBins)
        truthShape = N_sim_truth.shape
        N_obs_flat = np.ravel(N_obs, order='F')
        N_sim_truth_flat = np.ravel(scaledPrior, order='F')

        A = L.copy()
        lambda_reg = 0.01
        Ainv = np.dot( np.linalg.pinv(np.dot(A.T, A) + lambda_reg * np.identity(N_sim_truth_flat.size ) ), A.T)
        #N_est_flat = np.dot(Ainv, N_obs_flat)
        N_est_flat = N_sim_truth_flat + np.dot(Ainv, N_obs_flat - np.dot(A, N_sim_truth_flat) )
        N_est = np.reshape(N_est_flat, truthShape, order='F')
        this_recon = N_est[:,stellarity]
        
        #this_recon, this_err , _ = lfunc.doInference(catalog = thisDES, likelihood = Lpca, 
        #                                             obs_bins=obs_bins, truth_bins = truth_bins,
        #                                             lambda_reg = 1e-2, invType = 'tikhonov',
        #                                             prior = scaledPrior, priorNumber = sim.size)
    
    
        pcaRecon[:, i] = this_recon.copy() * 1./(thisTruth.size)
        #pcaReconErr[:,i] = this_err.copy() * 1./(thisTruth.size)
        coeff[:,i] = thisCoeff
        wt = 1.#/this_err[truth_bin_indices]**2
        theMap[hpInd] = np.sum(this_recon[truth_bin_indices] * wt) / np.sum(wt) * 1. / (thisTruth.size)
        #theMapErr[hpInd] = np.sqrt(np.sum(this_err[truth_bin_indices]**2)) * 1./thisTruth.size

        N_obs,_ = np.histogram(thisDES['mag_auto'], bins=truth_bins)
        N_obs = N_obs * 1./thisTruth.size
        rawRecon[:,i] = N_obs
        
        theMapObs[hpInd] = np.sum(N_obs[obs_bin_indices])
        truth_bin_centers = (truth_bins[0:-1] + truth_bins[1:])/2.
        print " HEALPixel, observed number,  reconstructed number ) = ", hpInd, theMapObs[hpInd], theMap[hpInd]

    print "sub-sampling catalog for visualization purposes..."
    sample = catalog[np.random.choice(catalog.size, size=10000,replace=False)]
    print "Making map polygons"
    poly, vertices = makeMapPolygons(theMap = theMap, N_recon = pcaRecon, N_recon_err=None, N_raw = rawRecon,
                           bin_centers = truth_bin_centers, HEALConfig = HEALConfig, vmin = -2., vmax = 2.)
    #print "Building combined map+histogram plots"
    #mapHistogramPlots(poly = poly, theMap = theMap, N_recon = pcaRecon, N_recon_err=pcaReconErr, N_raw = rawRecon, band=band,
    #                   bin_centers = truth_bin_centers, HEALConfig = HEALConfig, vertices=vertices, N_truth = N_truth,catalog=sample)
    
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


    print "Building HEALPixel likelihood matrices for pca basis."
    out_nside = healConfig['out_nside']
    healConfig['out_nside'] = nside_pca
    des = cfunc.HealPixifyCatalogs(catalog=des, healConfig=healConfig, ratag='ra', dectag = 'dec')
    sim = cfunc.HealPixifyCatalogs(catalog=sim, healConfig=healConfig, ratag='ra', dectag = 'dec')
    truth = cfunc.HealPixifyCatalogs(catalog=truth, healConfig=healConfig, ratag='ra', dectag = 'dec')
    truthMatched = cfunc.HealPixifyCatalogs(catalog=truthMatched, healConfig=healConfig, ratag='ra', dectag = 'dec')

    
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
                                                                                              doplot = False)
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
                                                                                              doplot = False)
    print "Populating maps with HEALPixel reconstructions"

    theMap, theMapObs, coeff = populateMapStellarity(catalog = des, sim=sim, truth=truth, truthMatched=truthMatched,
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
    fig, ax = plt.subplots()
    ax.hist(deltaMap[seen & np.isfinite(deltaMap)],bins=100)
    ax.set_xlabel('pca map overdensity')
    ax.set_ylabel('Number of HEALPixels')
    fig.savefig("pca-mapRecon-hist-"+band+".png")
    print "Making maps of pca 0 and 1"
    coeff0Map = np.zeros(theMap.size) + hp.UNSEEN
    coeff1Map = np.zeros(theMap.size) + hp.UNSEEN
    coeff2Map = np.zeros(theMap.size) + hp.UNSEEN
    coeff3Map = np.zeros(theMap.size) + hp.UNSEEN
    coeff0Map[seen] = coeff[0,:]
    coeff1Map[seen] = coeff[1,:]
    coeff2Map[seen] = coeff[2,:]
    coeff3Map[seen] = coeff[3,:]
    mapfunc.visualizeHealPixMap(coeff0Map, nest=True, title='pca0-'+band)
    mapfunc.visualizeHealPixMap(coeff1Map, nest=True, title='pca1-'+band)
    mapfunc.visualizeHealPixMap(coeff2Map, nest=True, title='pca2-'+band)
    mapfunc.visualizeHealPixMap(coeff3Map, nest=True, title='pca3-'+band)

    fig, ax = plt.subplots()
    ax.plot(coeff0Map[seen]-np.mean(coeff0Map[seen]), deltaMap[seen],'.',label='PC0')
    ax.plot(coeff1Map[seen]-np.mean(coeff1Map[seen]), deltaMap[seen],'.',label='PC1',alpha=0.6)
    ax.plot(coeff2Map[seen]-np.mean(coeff2Map[seen]), deltaMap[seen],'.',label='PC2', alpha=0.3)
    ax.plot(coeff3Map[seen]-np.mean(coeff3Map[seen]), deltaMap[seen],'.',label='PC3', alpha = 0.25)
    ax.legend(loc='best')
    fig.savefig('pca-density-'+band+'.png')
    print "Done."

    esutil.io.write('pca-mapRecon-'+band+'.fits',theMap)
    esutil.io.write('mapObs-i'+band+'.fits',theMapObs)

    stop


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
        nside = 64

    healConfig = cfunc.getHealConfig(out_nside = nside)
        
    des, sim, truthMatched, truth = cfunc.getStellarityCatalogs(reload = args.reload, band = args.filter)
    truth = truth[truth['mag'] > 15.]
    des =des[( des['mag_auto'] > 15.) & (des['flux_radius'] > 0) & (des['flux_radius'] < 10.)] 
    keep = (sim['mag_auto'] > 15.) & (truthMatched['mag'] > 0.) & (sim['flux_radius'] > 0) & (sim['flux_radius'] < 10.)
    sim = sim[keep]
    truthMatched = truthMatched[keep]
    #star_galaxy_inference(sim=sim,truth=truth,des=des,truthMatched=truthMatched, band=args.filter)
    #mag_size_inference(sim=sim,truth=truth,des=des,truthMatched=truthMatched, band=args.filter)
    star_galaxy_maps(sim=sim,truth=truth,des=des,truthMatched=truthMatched, band=args.filter, healConfig = healConfig)
    stop


if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
