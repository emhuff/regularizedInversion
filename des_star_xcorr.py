#!/usr/bin/env python

import matplotlib.pyplot as plt
import cfunc
import numpy as np
import sys
import argparse
import os

def position_angle(cat1, cat2, ratag1 = 'ra', dectag1='dec',ratag2='ra',dectag2='dec'):
    dra = (cat1[ratag1] - cat2[ratag2]) * np.cos(cat1[dectag1]*np.pi/180.)
    rad = np.pi/180.

    ddec = (cat1[dectag1] - cat2[dectag2])
    phi = np.arctan2(ddec, dra)
    return phi, dra, ddec


def xMatchStars( catalog= None, ratag = 'ra', dectag = 'dec', starfile='../../Data/2mass_stars_south_bgt20.fits', radius = 100./3600.):
    import esutil
    stars = esutil.io.read(starfile)
    depth=10
    h = esutil.htm.HTM(depth)
    catind,starind, d12 = h.match(catalog[ratag],catalog[dectag],stars['RA'],stars['DEC'],radius,maxmatch=0)
    posAngle,dx,dy = position_angle(catalog[catind], stars[starind],ratag2='RA',dectag2='DEC')
    return catalog[catind], stars[starind], posAngle, dx, dy, d12


def xCorrStars(catalog = None, ratag = 'ra', dectag = 'dec',starfile='../../Data/2mass_stars_south_bgt20.fits', radius = 300./3600., starMagBins = None, thresh = 1.0, nbins=100):

    mCat, mStars,_,_,_, d12 = xMatchStars(catalog=catalog, ratag=ratag, dectag=dectag, starfile=starfile, radius=radius)
    
    magBinsLower = starMagBins[0:-1]
    magBinsUpper = starMagBins[1:]
    rbins = np.linspace(0,radius,nbins+1)
    bin_centers = (rbins[0:-1] + rbins[1:])/2.
    xi_arr = np.zeros((nbins,magBinsLower.size))

    for lower, upper, i in zip(magBinsLower, magBinsUpper, xrange(magBinsLower.size)):
        these_all = ( mStars['J_M'] > lower) & (mStars['J_M'] <= upper)
        these_bad = these_all & ( np.abs(mCat['mag_auto'] - mCat['mag']) > thresh)
        h_bad,_ = np.histogram(d12[these_bad], bins= rbins, density=True)
        h_all,_ = np.histogram(d12[these_all], bins=rbins, density=True)
        xi_arr[:,i]  = h_bad*1./h_all-1.
    
    xi_arr[~np.isfinite(xi_arr)] = 0.0
    return xi_arr, bin_centers

def angleBinnedErrors(truthMatched = None, thresh=False):

    mGals, mStars, mAngles, _, _, _ = xMatchStars( catalog=truthMatched )
    var = ( mGals['mag_auto'] - mGals['mag'] ) **2
    dev = ( mGals['mag_auto'] - mGals['mag'] )
    radius = 100./3600. # In degrees
    nbins = 250.
    if thresh is False:
        bins = np.linspace(-radius, radius, nbins)
        thetaHistWeights, thetaBins = np.histogram(mAngles*180./np.pi , bins=bins, weights=dev )
        thetaHistNum,_ = np.histogram(mAngles, bins=thetaBins)
        bin_centers = (thetaBins[0:-1] + thetaBins[1:])/2.
        thetaMean = thetaHistWeights/thetaHistNum
        return thetaMean,bin_centers
    else:
        bad = np.sqrt(var) > thresh
        # What fraction of the badness is accounted for by stars?
        
        bins = np.linspace(-radius,radius,nbins)
        thetaHist,thetaBins = np.histogram(mAngles[bad]*180./np.pi,bins=bins,density=True)
        bin_centers = (thetaBins[0:-1] + thetaBins[1:])/2.
        return thetaHist,bin_centers

def binned2dErrors(truthMatched=None, thresh=False):
    
    mGals, mStars, _, dx, dy, _ = xMatchStars( catalog=truthMatched )
    var = ( mGals['mag_auto'] - mGals['mag'] ) **2
    dev = ( mGals['mag_auto'] - mGals['mag'] )

    radius = 100./3600. #in degrees
    nbins = 100.
    bins = np.linspace(-radius,radius,nbins)
                          
    if thresh is False:
        errHistWeights, xBins,yBins = np.histogram2d(dx,dy, bins=(bins,bins),weights=dev )
        errHistNumber, xBins,yBins = np.histogram2d(dx,dy, bins=(bins,bins) )
        xbin_centers = (xBins[0:-1] + xBins[1:])/2.
        ybin_centers =  (yBins[0:-1] + yBins[1:])/2.
        errMean = errHistWeights/ errHistNumber
        return errMean,xbin_centers, ybin_centers
    else:
        bad = np.sqrt(var) > thresh
        errHist,xBins, yBins = np.histogram2d(dx[bad],dy[bad] , bins=(bins,bins), normed=True)
        xbin_centers = (xBins[0:-1] + xBins[1:])/2.
        ybin_centers =  (yBins[0:-1] + yBins[1:])/2.
        return errHist,xbin_centers, ybin_centers





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
    errsThresh, angleThresh = angleBinnedErrors(truthMatched = truthMatched, thresh= 2.)
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize = (21,7))
    ax1.plot(angle* 180/np.pi,errs)
    ax1.set_xlabel('position angle relative to nearest bright star (degrees)')
    ax1.set_ylabel('average Balrog magnitude error')
    ax2.plot(angleThresh*180/np.pi, errsThresh)
    ax2.set_xlabel('position angle relative to nearest bright star (degrees)')
    ax2.set_ylabel('fraction of sample with catastrophic mag. err ( |err|>2 ) ')
    fig.savefig("magnitude_error_stars_xcorr.png")

    # Now, how much of what's left over comes from diffraction spikes?
    avgErrs2d, xcen,ycen = binned2dErrors( truthMatched = truthMatched,thresh=False)
    threshErrs2d, xcenThresh,ycenThresh = binned2dErrors( truthMatched = truthMatched,thresh=1.0)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=3, figsize=(21,7))
    xcen = 3600*xcen 
    ycen = 3600*ycen
    xcenThresh = 3600*xcenThresh
    ycenThresh = 3600*ycenThresh
    ax1.imshow(avgErrs2d, extent= [np.min(xcen), np.max(xcen), np.min(ycen),np.max(ycen)],origin="lower",aspect="equal")
    ax1.set_xlabel('dra from star (arcsec)')
    ax1.set_ylabel('ddec from star (arcsec) ')
    ax1.set_title("avg \n  magnitude error")
    im2 = ax2.imshow(threshErrs2d, extent= [np.min(xcenThresh), np.max(xcenThresh), np.min(ycenThresh),np.max(ycenThresh)],origin="lower",aspect="equal")
    ax2.set_xlabel('dra from star (arcsec)')
    ax2.set_ylabel('ddec from star (arcsec) ')
    ax2.set_title("positions of catastrophic \n mag. errors")
    fig.colorbar(im2,ax=ax2)
    
    nMagBins = 8
    magbins = np.linspace(5. ,18. ,nMagBins+1)
    xi, r = xCorrStars(catalog = truthMatched, starMagBins =magbins, thresh = 1.0, nbins=50)
    for i in xrange(nMagBins):
        label = str(np.round(magbins[i],decimals=1)) + ' < J_M < ' + str(np.round(magbins[i+1],decimals=1))
        ax3.plot(r*3600., np.arcsinh(xi[:,i]), label = label,color=plt.cm.cool(i*(1./nMagBins)))

    ax3.set_ylabel('arcsinh(xi)')
    ax3.set_xlabel('sep. (arcsec)')
    ax3.set_ylim([-1,2])
    ax3.axhline(0.,linestyle='--',color='red')
    ax3.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
    #ax3.legend(loc='best')
    np.savez("star_proximity_effects",xi=xi,r=r,magbins=magbins)
    
    fig.savefig("magnitude_error_stars_2d"+band+".png")




if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
