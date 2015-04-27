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
import esutil
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

def generateGalaxyTruthCatalog(n_obj = 10000, slope = 2.0):
    # Draw catalog entries from some simple distribution.
    # (A catalog should be a Numpy recarray, or similar, so we can get arrays indexed by keyword.)
    # (the catalog should have a 'mag' and an 'index' field)
    mag = []
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
    #error = np.sqrt( (sky_flux + flux) * calibration ) / calibration
    
    index = np.arange(int(n_obj))

    catalog = np.empty((n_obj),dtype=[('mag',mag.dtype),('balrog_index',index.dtype),
                                      ('size',size.dtype),
                                      ('flux',flux.dtype),
                                      ('stellarity',np.int),
                                      ('calibration',calibration.dtype),
                                      ('blended',type(True))])
    
    catalog['mag'] = mag
    catalog['flux'] = flux
    catalog['balrog_index'] = index
    catalog['size'] = size
    catalog['blended'] = False
    catalog['calibration'] = calibration
    catalog['stellarity'] = 0

    catalog = recfunctions.append_fields(catalog, 'data_truth', catalog['mag'])
    catalog = np.array(catalog)
    return catalog
    

def generateStarTruthCatalog(n_obj = 10000, slope = 1.0):
    stars =  generateGalaxyTruthCatalog(n_obj = n_obj, slope = slope)
    stars['size'] = 0.
    stars['stellarity'] = 1
    return stars


def blend(catalog, blend_fraction=0.1):
    # Choose two random subsets of the galaxies.
    # Add the flux of the first subset to that of the second, then remove the first.
    subset1 = np.random.choice(catalog,np.round(blend_fraction*catalog.size),replace=False)
    subset2 = np.random.choice(catalog,np.round(blend_fraction*catalog.size),replace=False)
    for entry1, entry2 in zip(subset1, subset2):
        newMag = addMagnitudes(entry1['mag'], entry2['mag'])
        ii = (catalog['balrog_index'] == entry1['balrog_index'])
        catalog[ii]['mag'] = newMag
        catalog[ii]['flux'] = magToFlux(newMag)
        catalog[ii]['blended'] = True
        catalog[ii]['size'] = np.max( (entry1['size'], entry2['size']) )
        catalog[ii]['stellarity'] = 0
    keep = np.in1d(catalog['balrog_index'], subset2['balrog_index'], assume_unique=True, invert=True)
    if np.sum(keep) == 0:
        stop
    catalog = catalog[keep]
    return catalog
        

def applyTransferFunction(catalog, SN_cut = 5., cbias = 0.0, mbias = 0.0, blend_fraction = 0.00, psf_size = 0.00):
    # This will add noise to the catalog entries, and apply cuts to remove some unobservable ones.
    # The noise is not trivial.
    obs_catalog = catalog.copy()

    # Apply psf.
    size_obs = np.sqrt( psf_size**2 + obs_catalog['size']**2 )
    obs_catalog['size'] = size_obs

    sky_sb =  np.repeat( 10.**( -(21.5 - 22.5) / 2.5 ), catalog.size)
    sky_flux = np.pi * size_obs * size_obs * sky_sb
    
    
    # Generate a noise vector based on the errors.
    flux_error = np.sqrt( (sky_flux + obs_catalog['flux']) * obs_catalog['calibration'] ) / obs_catalog['calibration']
    size_error = 2 * size_obs * flux_error / obs_catalog['flux']
    flux_noise = flux_error * np.random.standard_cauchy(size=len(obs_catalog))
    size_noise1 = size_error * np.random.randn(len(obs_catalog))
    size_noise2 = size_error * np.random.randn(len(obs_catalog))
    newFlux = obs_catalog['flux'] + flux_noise
    newSize = np.sqrt( ( (size_obs + size_noise1)**2 + (size_obs + size_noise2)**2 )/2)
    newMag = 25. - 2.5*np.log10(newFlux)
    obs_catalog['mag'] = newMag
    obs_catalog['flux'] = newFlux
    obs_catalog['size'] = newSize
    # Now recalculate the surface brightness.
    SB_new = newFlux / newSize**2 / np.pi + sky_sb
    
    # Apply a selection based on the new, noisy surface brightness. Take things that are >5 sigma above sky.
    # Blend. This happens after everything else.
    if blend_fraction > 0.0:
        obs_catalog = blend(obs_catalog, blend_fraction = blend_fraction)

    obs_catalog = obs_catalog[(SB_new >  5. * sky_sb)]
    
    return obs_catalog

def generateTruthCatalog(n_gal  = 100000, n_star = 10000, gal_slope = 2.5, star_slope = 1.0):
    stars = generateStarTruthCatalog(n_obj = n_star, slope = star_slope)
    galaxies = generateGalaxyTruthCatalog(n_obj = n_gal, slope = gal_slope)
    catalog = np.hstack( (stars, galaxies) )
    np.random.shuffle(catalog)
    return catalog

def main(argv):
    # Generate a simulated simulated truth catalog.
    psf_size = 0.5
    catalog_sim_truth = generateTruthCatalog(n_gal  = 100000, n_star = 10000, gal_slope = 2.5, star_slope = 1.0)
    catalog_sim_obs = applyTransferFunction(catalog_sim_truth, psf_size = psf_size)

    ind1, ind2 = esutil.numpy_util.match(catalog_sim_truth['balrog_index'],catalog_sim_obs['balrog_index'])
    '''
    plt.plot(catalog_sim_obs['size'], catalog_sim_obs['mag'],',',color='blue')
    plt.axvline(psf_size,linestyle='--',color='red')
    plt.gca().invert_yaxis()
    plt.show()
    
    bins = np.linspace( 0.3,1.0, 300)
    plt.hist(catalog_sim_obs['size'],bins=bins,color='blue',label='all')
    plt.hist(catalog_sim_obs[catalog_sim_obs['stellarity'] == 0]['size'],bins=bins,color='yellow',label='galaxies',alpha=0.5)
    plt.hist(catalog_sim_obs[catalog_sim_obs['stellarity'] == 1]['size'],bins=bins,color='orange',label='stars',alpha=0.5)
    plt.axvline(psf_size*1.04,linestyle='--',color='red')
    plt.xlim([0.33,1.0])
    plt.legend(loc='best')
    plt.show()
    '''
    
    obsStar = catalog_sim_obs['size'] <= psf_size * 1.04
    obsGal  = catalog_sim_obs['size'] >  psf_size * 1.04
    catalog_sim_obs['stellarity'][obsStar] = 1
    catalog_sim_obs['stellarity'][obsGal] = 0

    truthMatched = catalog_sim_truth[ind1].copy()
    catalog_sim_obs = catalog_sim_obs[ind2]
    obsMagBins = np.linspace(15,23,20)
    truthMagBins = np.linspace(15,25,25)
    starBins = np.array([0, 0.5, 1])
    reconBins = [truthMagBins, starBins]
    obsBins = [obsMagBins, starBins]

    #fig, ax = plt.subplots()
    in_var = truthMatched['mag'] + 1.01*np.max(truthMatched['mag']) * truthMatched['stellarity']
    out_var = catalog_sim_obs['mag'] + 1.01*np.max(catalog_sim_obs['mag']) * catalog_sim_obs['stellarity']
    plt.plot(in_var, out_var,',')
    plt.show()
    L = lfunc.makeLikelihoodMatrix(sim= catalog_sim_obs, truth=catalog_sim_truth, truthMatched = truthMatched,
                                     obs_bins = obsBins, truth_bins = reconBins, simTag = ['mag','stellarity'],
                                     truthTag = ['mag', 'stellarity'])
    #stop
    plt.imshow(np.arcsinh(L/0.001), origin='lower', cmap=plt.cm.Greys)
    plt.show()

    

if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
