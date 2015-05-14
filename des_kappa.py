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

def truthModKappa(truth = None, kappa = None, sizeTag = 'radius', magTag = 'mag'):
    import copy
    truthMod = copy.deepcopy(truth)
    galaxies = ( truthMod['objtype'] == 1 )
    truthMod[magTag][galaxies] = truthMod[magTag][galaxies] - 2.5*np.log10(1 + 2 * kappa)
    truthMod[sizeTag][galaxies] = truthMod[sizeTag][galaxies] * (1 + kappa )
    return truthMod

def computeDnDkappa( truth= None, tags = None, bins = None, dkappa = 0.1, magTag = 'mag', sizeTag = 'radius' ):
    pts = [truth[thisTag] for thisTag in tags]
    n_tags = len(tags)
    N_truth, _  = np.histogramdd( pts, bins = bins )
    
    truth_mod = truthModKappa( truth = truth, sizeTag = sizeTag, magTag = magTag, kappa = dkappa )
    pts_mod = [truth_mod[thisTag] for thisTag in tags]
    N_mod, _ = np.histogramdd( pts_mod, bins = bins )

    dN_dk = (N_mod - N_truth) / dkappa *1.  / truth.size
    
    return dN_dk

def kappa_est(hist_obs = None, likelihood = None, dN_dkappa = None, hist_fid = None):

    obs_shape = hist_obs.shape
    fid_shape = hist_fid.shape
    hist_fid_flat =  np.ravel( hist_fid, order='F' )
    hist_obs_flat = np.ravel ( hist_obs, order = 'F')
    dN_dkappa_flat = np.ravel( dN_dkappa, order = 'F') * np.sum(hist_fid)

    dNobs_dkappa_flat = np.dot(likelihood, dN_dkappa_flat)
    

    
    kappa = np.dot(  (hist_obs_flat - np.dot(likelihood, hist_fid_flat) ) , dNobs_dkappa_flat) / np.dot(dNobs_dkappa_flat, dNobs_dkappa_flat)

    if ~np.isfinite(kappa):
        stop
    return kappa
    


def kappa_maps(sim=None,des=None,truth=None,truthMatched=None, band=None, healConfig = None, nside_pca = 64, dN_dkappa = None):
    truthTags = ('mag','radius')
    truthMagBins = np.linspace(16,25.2,30)
    truthSizeBins = np.insert(-0.2, 1,np.logspace(-2, 1., 30)) 
    truthBins = [truthMagBins, truthSizeBins]
    
    obsTags = ('mag_auto', 'flux_radius')
    obsMagBins = np.linspace(18,23.5,30)
    obsSizeBins = np.logspace(-.5, 1., 30)
    obsBins = [obsMagBins, obsSizeBins]
    
    n_component = 8

    print "Building HEALPixel likelihood matrices for pca basis."

    out_nside = healConfig['out_nside']
    healConfig['out_nside'] = nside_pca
    des = cfunc.HealPixifyCatalogs(catalog=des, healConfig=healConfig, ratag='ra', dectag = 'dec')
    sim = cfunc.HealPixifyCatalogs(catalog=sim, healConfig=healConfig, ratag='ra', dectag = 'dec')
    truth = cfunc.HealPixifyCatalogs(catalog=truth, healConfig=healConfig, ratag='ra', dectag = 'dec')
    truthMatched = cfunc.HealPixifyCatalogs(catalog=truthMatched, healConfig=healConfig, ratag='ra', dectag = 'dec')

    N_sim_truth, _ = np.histogramdd([truth['mag'], truth['radius']], bins = truthBins)
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
                                                                                              doplot = False)
    pcaBasis, pcaEigen = lfunc.likelihoodPCA(likelihood = Likelihoods, doplot=False, band=band, extent = None)
    print "Re-fitting primary principal components to master likelihood"
    LmasterPCA, coeffMaster = lfunc.doLikelihoodPCAfit(pcaComp = pcaBasis,
                                                       likelihood = masterLikelihood,
                                                       n_component = n_component,
                                                       Lcut = 1e-4)
    print "Re-HEALPixifying catalogs for map."
    healConfig['out_nside'] = out_nside
    des = cfunc.HealPixifyCatalogs(catalog=des, healConfig=healConfig, ratag='ra', dectag = 'dec')
    sim = cfunc.HealPixifyCatalogs(catalog=sim, healConfig=healConfig, ratag='ra', dectag = 'dec')
    truth = cfunc.HealPixifyCatalogs(catalog=truth, healConfig=healConfig, ratag='ra', dectag = 'dec')
    truthMatched = cfunc.HealPixifyCatalogs(catalog=truthMatched, healConfig=healConfig, ratag='ra', dectag = 'dec')
    useInds = np.unique(des['HEALIndex'])
    mapIndices = np.arange(hp.nside2npix(healConfig['out_nside']))
    theMap = np.zeros(mapIndices.size) + hp.UNSEEN
    print "Computing histogram kappa derivative..."
    dN_dk = computeDnDkappa( truth= truth, tags = truthTags, bins = truthBins , magTag = 'mag', sizeTag = 'radius' )
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(7,7))
    im = ax.imshow(dN_dk,origin='lower',cmap=plt.cm.bwr)
    fig.colorbar(im,cax=ax)
    fig.savefig('kappa-filter-'+band)

    
    print "Starting mapmaking..."
    for i, hpInd in zip(xrange(useInds.size), useInds):
        thisSim = sim[sim['HEALIndex'] == hpInd]
        thisMatched = truthMatched[sim['HEALIndex'] == hpInd]
        thisTruth = truth[truth['HEALIndex'] == hpInd]
        thisDES = des[des['HEALIndex'] == hpInd]

        print "for pixel "+str(i)+' of '+str(useInds.size-1)+":"
        print "   making likelihood matrix..."
        Lraw = lfunc.makeLikelihoodMatrix( sim=thisSim, 
                                           truth=thisTruth, 
                                           truthMatched =thisMatched,
                                           obs_bins = obsBins, truth_bins = truthBins, 
                                           simTag = ['mag_auto','flux_radius'], truthTag = ['mag','radius'])

        print "   fitting likelihood to largest PCA components..."
        Lpca, thisCoeff = lfunc.doLikelihoodPCAfit(pcaComp = pcaBasis,
                                                   likelihood = Lraw,
                                                   n_component = n_component,
                                                   Lcut = 1e-3, Ntot = thisSim.size)
        
        pts = [thisDES[thisTag] for thisTag in obsTags]
        N_obs, _ = np.histogramdd(pts, bins = obs_bins)
        
        N_obs_scaled = N_obs * 1./thisTruth.size * (sim.size * 1./des.size)

        kappa = kappa_est( hist_obs = N_obs_scaled, likelihood = Lraw, dN_dkappa = dN_dk, hist_fid = N_sim_truth )
        print hpInd, kappa
        theMap[hpInd] = kappa
        
    return theMap
    


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

    kMap = kappa_maps(sim=sim,truth=truth,des=des,truthMatched=truthMatched, band=args.filter, healConfig = healConfig)
    seen = kMap != hp.UNSEEN
    dkappa = kMap*0. + hp.UNSEEN
    dkappa[seen] = kMap[seen] - np.median(kMap[seen])
    theRange = 3*np.max( [np.abs(np.percentile(dkappa[seen],[25,75]))])
    mapfunc.visualizeHealPixMap(dkappa, nest=True, title="kappa-"+args.filter, vmin=-theRange, vmax= theRange)
    esutil.io.write('kappa-'+args.filter+'.fits',dkappa)
    stop

if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
