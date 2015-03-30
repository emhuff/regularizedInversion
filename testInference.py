#!/usr/bin/env python
import numpy as np
from numpy import recarray
from scipy import linalg as slin

import sys
import esutil
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import lfunc
import cfunc
import numpy.lib.recfunctions as recfunctions



def fluxToMag(mag, zeropoint = 25.):
    return 10.**((zeropoint - mag)/2.5)

def magToFlux(flux, zeropoint = 25.):
    mag = zeropoint - 2.5*np.log10(flux)
    return mag

def addMagnitudes(mag1, mag2, zeropoint = 25.):
    flux1 = fluxToMag(mag1, zeropoint = zeropoint)
    flux2 = fluxToMag(mag2, zeropoint = zeropoint)
    newFlux = flux1 + flux2
    newMag = zeropoint - 2.5*np.log10(newFlux)
    return newMag
    

def generateTruthCatalog(n_obj = 10000, slope = 1.0, downsample = False):
    # Draw catalog entries from some simple distribution.
    # (A catalog should be a Numpy recarray, or similar, so we can get arrays indexed by keyword.)
    # (the catalog should have a 'data' and an 'index' field)
    mag = []
    if downsample is False:
        mag =  15 + 12*np.random.power(1+slope, size=n_obj)
    else:
        while len(mag) < n_obj:
            thismag =  15 + 15*np.random.power(1+slope)
            if np.random.rand() > np.exp((thismag - 26)/3.):
                mag.append(thismag)
        mag = np.array(mag)
    log_size =  (-0.287* mag + 4.98) + (0.0542* mag - 0.83 )* np.random.randn(n_obj)
    flux = fluxToMag(mag)
    size = np.exp(log_size) # Fortunately, this is basically in arcsec
    surfaceBrightness = flux / size / size / np.pi
    # A typical sky brightness is 22.5 mag / arcsec^2
    sky_sb =  np.repeat( 10.**( -(21.5 - 22.5) / 2.5 ), n_obj)

    sky_flux = np.pi * size * size * sky_sb

    # The flux calibration will be total photon counts per unit of flux, integrated over the exposure time.
    # It's really just there to give us a noise model.
    calibration = np.repeat(100.,n_obj)
    error = np.sqrt( (sky_flux + flux) * calibration ) / calibration
    
    index = np.arange(int(n_obj))

    catalog = np.empty((n_obj),dtype=[('data',mag.dtype),('error',error.dtype),('balrog_index',index.dtype),
                                      ('calibration',calibration.dtype),('size',size.dtype), ('SB',surfaceBrightness.dtype),
                                      ('sky_flux',sky_flux.dtype),('sky_SB',sky_sb.dtype),('flux',flux.dtype),
                                      ('blended',type(True))])
    
    catalog['data'] = mag
    catalog['flux'] = mag
    catalog['error'] = error
    catalog['balrog_index'] = index
    catalog['calibration'] = calibration
    catalog['size'] = size
    catalog['SB'] = surfaceBrightness
    catalog['sky_SB'] = sky_sb
    catalog['blended'] = False

    catalog = recfunctions.append_fields(catalog, 'data_truth', catalog['data'])
    catalog = np.array(catalog)
    return catalog
    

def blend(catalog, blend_fraction=0.1):
    # Choose two random subsets of the galaxies.
    # Add the flux of the first subset to that of the second, then remove the first.
    subset1 = np.random.choice(catalog,np.round(blend_fraction*catalog.size),replace=False)
    subset2 = np.random.choice(catalog,np.round(blend_fraction*catalog.size),replace=False)
    for entry1, entry2 in zip(subset1, subset2):
        newMag = addMagnitudes(entry1['data'], entry2['data'])
        ii = (catalog['balrog_index'] == entry1['balrog_index'])
        catalog[ii]['data'] = newMag
        catalog[ii]['flux'] = magToFlux(newMag)
        catalog[ii]['blended'] = True
        catalog[ii]['size'] = np.max( (entry1['size'], entry2['size']) )
        catalog[ii]['error'] = np.min( (entry1['error'], entry2['error']) )
        catalog[ii]['SB'] = catalog['flux'][ii] / ( catalog['size'][ii] **2) / np.pi
    keep = np.in1d(catalog['balrog_index'], subset2['balrog_index'], assume_unique=True, invert=True)
    if np.sum(keep) == 0:
        stop
    catalog = catalog[keep]
    return catalog
        

def applyTransferFunction(catalog, SN_cut = 5., cbias = 0.0, mbias = 0.0, blend_fraction = 0.00):
    # This will add noise to the catalog entries, and apply cuts to remove some unobservable ones.
    # The noise is not trivial.
    obs_catalog = catalog.copy()

    # Blend. This happens before anything else.
    if blend_fraction > 0.0:
        obs_catalog = blend(obs_catalog, blend_fraction = blend_fraction)
    
    # Generate a noise vector based on the errors.
    #noise = obs_catalog['error']*np.random.randn(len(obs_catalog))
    noise = obs_catalog['error']*np.random.standard_cauchy(size=len(obs_catalog))
    newFlux = 10.**((25. - obs_catalog['data'])/2.5) + noise
    newMag = 25. - 2.5*np.log10(newFlux)
    obs_catalog['data'] = newMag
    # Now recalculate the surface brightness.
    SB_new = obs_catalog['SB'] + noise / (obs_catalog['size'])**2 / np.pi
    
    # Apply a selection based on the new, noisy surface brightness. Take things that are >5sigma above sky.
    
    obs_catalog = obs_catalog[(SB_new >  5. * obs_catalog['sky_SB']) & (obs_catalog['size'] > 0.2) & (newFlux > 0)]

    return obs_catalog

def truncateTruthCatalog(truth=None,mag_cut= 30.,size_cut= 0., mag_tag = 'data', size_tag = 'size'):
    keep = (truth[mag_tag] < mag_cut) & (truth[size_tag] > size_cut)
    return truth[keep]
    
    

def main(argv):
    # Generate a simulated simulated truth catalog.
    catalog_sim_truth = generateTruthCatalog(n_obj  = 100000, slope = 2.500, downsample = False)

    
    catalog_ref_truth = truncateTruthCatalog(truth = catalog_sim_truth, mag_cut = 24.0, size_cut = 0.05)
    
    # Apply some complicated transfer function, get back a simulated
    # simulated observed catalog.
    catalog_ref_obs = applyTransferFunction(catalog_ref_truth)
    catalog_sim_obs = applyTransferFunction(catalog_sim_truth)

    truthMatched_sim = cfunc.mergeCatalogsUsingPandas(sim=catalog_sim_obs, truth=catalog_sim_truth, key='balrog_index')
    truthMatched_ref = cfunc.mergeCatalogsUsingPandas(sim=catalog_ref_obs, truth=catalog_ref_truth, key='balrog_index')

    # Generate a simulated `real' truth catalog.
    catalog_real_truth = generateTruthCatalog(n_obj = 100000, slope = 2.50, downsample = False)
    
    # Generate a simulated `real' observed catalog.
    catalog_real_obs = applyTransferFunction(catalog_real_truth)

    truthMatched_real = cfunc.mergeCatalogsUsingPandas(sim=catalog_real_obs, truth=catalog_real_truth, key='balrog_index')

    # Get the likelihood matrix.
    obsBins   = lfunc.chooseBins(catalog=catalog_real_obs , tag='data', binsize=0.25)
    obs_bin_centers = ( obsBins[0:-1] + obsBins[1:])/2.
    truthBins = lfunc.chooseBins(catalog=catalog_sim_truth, tag='data', binsize=0.25)
    
    L = lfunc.makeLikelihoodMatrix(sim=catalog_sim_obs, truth=catalog_sim_truth, truthMatched = truthMatched_sim,
                                   simTag = 'data', truthTag = 'data', obs_bins = obsBins, truth_bins = truthBins)
    Lref = lfunc.makeLikelihoodMatrix(sim=catalog_ref_obs, truth=catalog_ref_truth, truthMatched = truthMatched_ref,
                                   simTag = 'data', truthTag = 'data', obs_bins = obsBins, truth_bins = truthBins)

    N_real_est, est_errs, truth_bin_centers = lfunc.doInference(catalog = catalog_real_obs, likelihood=L, tag='data',
                                                                obs_bins = obsBins, truth_bins = truthBins,
                                                                lambda_reg = 1e-2, invType = 'tikhonov')
    N_real_ref, ref_errs, _ = lfunc.doInference(catalog = catalog_real_obs, likelihood=Lref, tag='data',
                                                                obs_bins = obsBins, truth_bins = truthBins,
                                                                lambda_reg = 1e-2, invType = 'tikhonov')

    N_real_truth, _ = np.histogram(catalog_real_truth['data'], bins = truthBins)
    N_real_obs ,_ = np.histogram(catalog_real_obs['data'], bins= obsBins)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
    ax1.plot(truth_bin_centers, N_real_est,label='est.')
    ax1.plot(truth_bin_centers, N_real_truth,label='truth')
    ax1.plot(truth_bin_centers, N_real_ref, label='ref')
    ax1.plot(obs_bin_centers, N_real_obs, label='obs')
    ax1.legend(loc='best')
    ax1.set_ylim([0,np.max(N_real_truth)])

    extent= [truth_bin_centers[0],truth_bin_centers[-1],obs_bin_centers[0],obs_bin_centers[-1]]
    im2 = ax2.imshow(np.arcsinh(L/1e-3), origin='lower', extent = extent, cmap=plt.cm.Greys)
    fig.colorbar(im2, ax=ax2)
    im3 = ax3.imshow(np.arcsinh(Lref/1e-3), origin='lower', extent=  extent, cmap=plt.cm.Greys)
    fig.colorbar(im3, ax=ax3)
    stop

if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
