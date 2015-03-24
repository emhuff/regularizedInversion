#!/usr/bin/env python

import matplotlib.pyplot as plt
import cfunc
import numpy as np
import sys
import argparse

def position_angle(cat1, cat2, ratag1 = 'ra', dectag1='dec',ratag2='ra',dectag2='dec'):
    dra = cat1[ratag1] - cat2[ratag2]
    ddec = cat1[dectag1] - cat2[dectag2]
    phi = np.arctan2(ddec, dra)
    return phi


def xcorrStars( catalog= None, ratag = 'ra', dectag = 'dec', starfile='../../Data/2mass_stars_south_bgt20.fits', radius = 300./3600):
    import esutil
    stars = esutil.io.read(starfile)
    
    depth=10
    h = esutil.htm.HTM(depth)
    catind,starind, d12 = h.match(catalog[ratag],catalog[dectag],stars['RA'],stars['DEC'],radius,maxmatch=0)
    
    posAngle = position_angle(catalog[catind], stars[starind],ratag2='RA',dectag2='DEC')
    return catalog[catind], stars[starind], posAngle

def angleBinnedErrors(truthMatched = None, thresh=False):

    mGals, mStars, mAngles = xcorrStars( catalog=truthMatched)
    var = ( mGals['mag_auto'] - mGals['mag'] ) **2
    dev = ( mGals['mag_auto'] - mGals['mag'] )

    if thresh is False:
        thetaHistWeights, thetaBins = np.histogram(mAngles, bins=50,weights=dev )
        thetaHistNum,_ = np.histogram(mAngles, bins=thetaBins)
        bin_centers = (thetaBins[0:-1] + thetaBins[1:])/2.
        thetaMean = thetaHistWeights/thetaHistNum
        return thetaMean,bin_centers
    else:
        bad = np.sqrt(var) > thresh
        thetaHist,thetaBins = np.histogram(mAngles[bad],bins=50,density=True)
        bin_centers = (thetaBins[0:-1] + thetaBins[1:])/2.
        return thetaHist,bin_centers

def main(argv):
    parser = argparse.ArgumentParser(description = 'Perform magnitude distribution inference on DES data.')
    parser.add_argument('filter',help='filter name',choices=['g','r','i','z','Y'])
    parser.add_argument("-r","--reload",help='reload catalogs from DESDB', action="store_true")
    parser.add_argument("-np","--no_pca",help='Do not use pca-smoothed likelihoods in reconstruction.', action="store_true")
    args = parser.parse_args(argv[1:])
    band = args.filter
    no_pca = args.no_pca
    
    # Get the catalogs.
    print "performing inference in band: "+args.filter
    print "Reloading from DESDM:", args.reload
    des, sim, truthMatched, truth, tileInfo = cfunc.getCatalogs(reload = args.reload, band = args.filter)
    print "Excluding regions Eli says are bad."
    des, sim, truthMatched, truth = cfunc.excludeBadRegions(des,sim, truthMatched, truth,band=band)
    print sim.size

    # Now, how much of what's left over comes from diffraction spikes?
    errs, angle = angleBinnedErrors( truthMatched = truthMatched,thresh=False)
    errsThresh, angleThresh = angleBinnedErrors(truthMatched = truthMatched,thresh= 2.)
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize = (14,7))
    ax1.plot(angle* 180/np.pi,errs)
    ax1.set_xlabel('position angle relative to nearest bright star (degrees)')
    ax1.set_ylabel('average Balrog magnitude error')
    ax2.plot(angleThresh*180/np.pi, errsThresh)
    ax2.set_xlabel('position angle relative to nearest bright star (degrees)')
    ax2.set_ylabel('fraction of sample with catastrophic mag. err ( |err|>2 ) ')
    fig.savefig("magnitude_error_stars_xcorr.png")

if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
