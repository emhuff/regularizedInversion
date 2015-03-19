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




def NoSimFields(band='i'):
    q = """
    SELECT
        balrog_index,
        mag_auto,
        flags
    FROM
        SUCHYTA1.balrog_sva1v2_nosim_%s
    """ %(band)
    return q



def SimFields(band='i',table='sva1v2'):
    q = """
    SELECT
        t.tilename as tilename,
        m.xwin_image as xwin_image,
        m.ywin_image as ywin_image,
        m.xmin_image as xmin_image,
        m.ymin_image as ymin_image,
        m.xmax_image as xmax_image,
        m.ymax_image as ymax_image,
        m.balrog_index as balrog_index,
        m.alphawin_j2000 as ra,
        m.deltawin_j2000 as dec,
        m.mag_auto as mag_auto,
        m.spread_model as spread_model,
        m.spreaderr_model as spreaderr_model,
        m.class_star as class_star,
        m.mag_psf as mag_psf,
        t.mag as truth_mag_auto,
        m.flags as flags
    FROM
        SUCHYTA1.balrog_%s_sim_%s m
        JOIN SUCHYTA1.balrog_%s_truth_%s t ON t.balrog_index = m.balrog_index
    """ %(table, band, table, band)
    return q





def DESFields(tilestuff, band='i'):
    q = """
        SELECT
           tilename,
           coadd_objects_id,
           mag_auto_%s as mag_auto,
           alphawin_j2000_%s as ra,
           deltawin_j2000_%s as dec,
           spread_model_%s as spread_model,
           spreaderr_model_%s as spreaderr_model,
           class_star_%s as class_star,
           mag_psf_%s as mag_psf,
           flags_%s as flags
        FROM
           sva1_coadd_objects
        WHERE
           tilename in %s
        """ % (band,band,band,band,band,band,band,band,str(tuple(np.unique(tilestuff['tilename']))))
    return q


def TruthFields(band='i', table = 'sva1v2'):
    q = """
    SELECT
        balrog_index,
        tilename,
        ra,
        dec,
        objtype,
        mag
    FROM
        SUCHYTA1.balrog_%s_truth_%s        
    """%(table,band)
    return q
    

def GetDESCat( depthmap, nside, tilestuff, tileinfo, band='i',depth = 0.0):
    cur = desdb.connect()
    q = DESFields(tileinfo, band=band)
    detcat = cur.quick(q, array=True)
    detcat = functions2.ValidDepth(depthmap, nside, detcat, rakey='ra', deckey='dec',depth = depth)
    detcat = functions2.RemoveTileOverlap(tilestuff, detcat, col='tilename', rakey='ra', deckey='dec')
    return detcat



def getTileInfo(catalog, HealConfig=None):
    if HealConfig is None:
        HealConfig = getHealConfig()
        
    tiles = np.unique(catalog['tilename'])
    cur = desdb.connect()
    q = "SELECT tilename, udecll, udecur, urall, uraur FROM coaddtile"
    tileinfo = cur.quick(q, array=True)
    tilestuff = {}
    for i in range(len(tileinfo)):
        tilestuff[ tileinfo[i]['tilename'] ] = tileinfo[i]
    max = np.power(map_nside/float(HealConfig['out_nside']), 2.0)
    depthmap, nside = functions2.GetDepthMap(HealConfig['depthfile'])
    return depthmap, nside


def cleanCatalog(catalog, tag='mag_auto'):
    # We should get rid of obviously wrong things.
    keep = np.where( (catalog[tag] > 15. ) & (catalog[tag] < 30.) & (catalog['flags'] < 2) )
    return catalog[keep]

def removeBadTilesFromTruthCatalog(truth, tag='mag_auto', goodfrac = 0.8):
    tileList = np.unique(truth['tilename'])
    number = np.zeros(tileList.size)
    for tile, i in zip(tileList,xrange(number.size)):
        number[i] = np.sum(truth['tilename'] == tile)
    tileList = tileList[number > goodfrac*np.max(number)]
    keep = np.in1d( truth['tilename'], tileList )
    return truth[keep]



def mergeCatalogsUsingPandas(sim=None, truth=None, key='balrog_index', suffixes = ['_sim','']):
    import pandas as pd
    simData = pd.DataFrame(sim)
    truthData = pd.DataFrame(truth)
    matched = pd.merge(simData, truthData, on=key, suffixes = suffixes)
    matched_arr = matched.to_records(index=False)
    # This last step is necessary because Pandas converts strings to Objects when eating structured arrays.
    # And np.recfunctions flips out when it has one.
    oldDtype = matched_arr.dtype.descr
    newDtype = oldDtype
    for thisOldType,i in zip(oldDtype, xrange(len(oldDtype) )):
        if 'O' in thisOldType[1]:
            newDtype[i] = (thisOldType[0], 'S12')
    matched_arr = np.array(matched_arr,dtype=newDtype)
    return matched_arr



def GetFromDB( band='i', depth = 0.0,tables =['sva1v2','sva1v3_2']): # tables =['sva1v2','sva1v3','sva1v3_2']
    depthfile = '../sva1_gold_1.0.2-4_nside4096_nest_i_auto_weights.fits'

    cur = desdb.connect()
    q = "SELECT tilename, udecll, udecur, urall, uraur FROM coaddtile"
    tileinfo = cur.quick(q, array=True)
    tilestuff = {}
    for i in range(len(tileinfo)):
        tilestuff[ tileinfo[i]['tilename'] ] = tileinfo[i]
    depthmap, nside = functions2.GetDepthMap(depthfile)
    truths = []
    sims = []
    truthMatcheds = []
    
    for tableName in tables:
        q = TruthFields(band=band,table=tableName)
        truth = cur.quick(q, array=True)

        truth = removeBadTilesFromTruthCatalog(truth)
        truth = functions2.ValidDepth(depthmap, nside, truth, depth = depth)
        truth = functions2.RemoveTileOverlap(tilestuff, truth)
        unique_binds, unique_inds = np.unique(truth['balrog_index'],return_index=True)
        truth = truth[unique_inds]

        q = SimFields(band=band, table=tableName)
        sim = cur.quick(q, array=True)
        sim = cleanCatalog(sim,tag='mag_auto')
        unique_binds, unique_inds = np.unique(sim['balrog_index'],return_index=True)
        sim = sim[unique_inds]
        
        
        truthMatched = mergeCatalogsUsingPandas(sim=sim,truth=truth)
        
        sim = sim[np.in1d(sim['balrog_index'],truthMatched['balrog_index'])]
        sim.sort(order='balrog_index')
        truthMatched.sort(order='balrog_index')
        
        truthMatcheds.append(truthMatched)
        truths.append(truth)
        sims.append(sim)

    sim = np.hstack(sims)
    truth = np.hstack(truths)
    truthMatched = np.hstack(truthMatcheds)
    
    des = GetDESCat(depthmap, nside, tilestuff, sim, band=band,depth = depth)
    des = cleanCatalog(des, tag='mag_auto')
    
    return des, sim, truthMatched, truth, tileinfo


def getCatalogs(reload=False,band='i'):

    # Check to see whether the catalog files exist.  If they do, then
    # use the files. If at least one does not, then get what we need
    # from the database

    fileNames = ['desCatalogFile-'+band+'.fits','BalrogObsFile-'+band+'.fits',
                 'BalrogTruthFile-'+band+'.fits', 'BalrogTruthMatchedFile-'+band+'.fits',
                 'BalrogTileInfo.fits']
    exists = True
    for thisFile in fileNames:
        print "Checking for existence of: "+thisFile
        if not os.path.isfile(thisFile): exists = False
    if exists and not reload:
        desCat = esutil.io.read(fileNames[0])
        BalrogObs = esutil.io.read(fileNames[1])
        BalrogTruth = esutil.io.read(fileNames[2])
        BalrogTruthMatched = esutil.io.read(fileNames[3])
        BalrogTileInfo = esutil.io.read(fileNames[4])
    else:
        print "Cannot find files, or have been asked to reload. Getting data from DESDB."
        desCat, BalrogObs, BalrogTruthMatched, BalrogTruth, BalrogTileInfo = GetFromDB(band=band)
        esutil.io.write( fileNames[0], desCat , clobber=True)
        esutil.io.write( fileNames[1], BalrogObs , clobber=True)
        esutil.io.write( fileNames[2], BalrogTruth , clobber=True)
        esutil.io.write( fileNames[3], BalrogTruthMatched , clobber=True)
        esutil.io.write( fileNames[4], BalrogTileInfo, clobber=True)
        
    return desCat, BalrogObs, BalrogTruthMatched, BalrogTruth, BalrogTileInfo



def hpHEALPixelToRaDec(pixel, nside=4096, nest=True):
    theta, phi = hp.pix2ang(nside, pixel, nest=nest)
    ra, dec = convertThetaPhiToRaDec(theta, phi)
    return ra, dec

def hpRaDecToHEALPixel(ra, dec, nside=  4096, nest= True):
    phi = ra * np.pi / 180.0
    theta = (90.0 - dec) * np.pi / 180.0
    hpInd = hp.ang2pix(nside, theta, phi, nest= nest)
    return hpInd

def convertThetaPhiToRaDec(theta, phi):
    ra = phi*180.0/np.pi
    dec = 90.0 - theta*180.0/np.pi
    return ra,dec

def convertRaDecToThetaPhi(ra, dec):
    theta = (90.0 - dec) * np.pi / 180.0
    phi =  ra * np.pi / 180.0
    return theta, phi

def HealPixifyCatalogs(catalog=None, healConfig=None, ratag='ra', dectag = 'dec'):
    HealInds = hpRaDecToHEALPixel( catalog[ratag],catalog[dectag], nside= healConfig['map_nside'], nest= healConfig['nest'])    
    healCat = rf.append_fields(catalog,'HEALIndex',HealInds,dtypes=HealInds.dtype)
    return healCat


def getHealConfig(map_nside = 4096, out_nside = 128, depthfile = '../sva1_gold_1.0.2-4_nside4096_nest_i_auto_weights.fits'):
    HealConfig = {}
    HealConfig['map_nside'] = map_nside
    HealConfig['out_nside'] = out_nside
    HealConfig['finer_nside'] = map_nside
    HealConfig['depthfile'] = depthfile
    HealConfig['nest'] = True
    return HealConfig


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
        if N_truth[truth_bin_index[i]] > 1:
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
        obs_bins = chooseBins(catalog=sim, tag = obsTag, binsize=0.5,upperLimit=26.,lowerLimit=18.)
    if truth_bins is None:
        truth_bins = chooseBins(catalog = truthMatched, tag = truthTag, binsize = 0.5,upperLimit=27.,lowerLimit=18)

    truth_bin_centers = (truth_bins[0:-1] + truth_bins[1:])/2.
    obs_bin_centers = (obs_bins[0:-1] + obs_bins[1:])/2.
        
    Lensemble = np.empty( (obs_bins.size-1 , truth_bins.size-1, useInds.size) )

    if doplot is True:
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.colors import LogNorm
        pp = PdfPages('likelihoods.pdf')
        fig,ax = plt.subplots(figsize=(6.,6.))
        # Make a plot of the likelihood of the whole region.
        masterLikelihood =  makeLikelihoodMatrix( sim=sim, truth=truth, truthMatched = truthMatched, Lcut = 1e-4,
                                                  obs_bins = obs_bins, truth_bins = truth_bins, simTag = obsTag, truthTag = truthTag)
        im = ax.imshow(masterLikelihood, origin='lower',cmap=plt.cm.Greys, norm = LogNorm(vmin=1e-6,vmax=1),
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
            thisLikelihood = makeLikelihoodMatrix( sim=thisSim, truth=thisTruth, truthMatched = thisTruthMatched,Lcut = 1e-3,
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
    

def getGoodRegionIndices(catalog=None, badHPInds=None, nside=4096):
    hpInd = hpRaDecToHEALPixel(catalog['ra'], catalog['dec'], nside=nside, nest= True)
    keep = ~np.in1d(hpInd, badHPInds)
    return keep


def excludeBadRegions(des,balrogObs, balrogTruthMatched, balrogTruth, band='i'):
    eliMap = hp.read_map("sva1_gold_1.0.4_goodregions_04_equ_nest_4096.fits", nest=True)
    nside = hp.npix2nside(eliMap.size)
    maskIndices = np.arange(eliMap.size)
    badIndices = maskIndices[eliMap == 1]

    obsKeepIndices = getGoodRegionIndices(catalog=balrogObs, badHPInds=badIndices, nside=nside)
    truthKeepIndices = getGoodRegionIndices(catalog=balrogTruth, badHPInds=badIndices, nside=nside)
    desKeepIndices = getGoodRegionIndices(catalog=des, badHPInds=badIndices, nside=nside)

    balrogObs = balrogObs[obsKeepIndices]
    balrogTruthMatched = balrogTruthMatched[obsKeepIndices]
    balrogTruth = balrogTruth[truthKeepIndices]
    des = des[desKeepIndices]

    return des,balrogObs, balrogTruthMatched, balrogTruth


def likelihoodPCA(likelihood= None,  likelihood_master = None):
  # This does a simple PCA on the array of likelihood matrices to find a compact basis with which to represent the likelihood.
  origShape = np.shape(likelihood)
  likelihood_1d = np.reshape(likelihood, (origShape[0]*origShape[1], origShape[2]))
  
  L1d_master = np.reshape(likelihood_master, origShape[0]*origShape[1])

  # Subtract L1d_master from each row of L1d:
  for i in xrange(origShape[2]):
    likelihood_1d[:,i] = likelihood_1d[:,i] - L1d_master
  L1d = likelihood_1d.T
  U,s,Vt = np.linalg.svd(L1d,full_matrices=False)
  V = Vt.T
  ind = np.argsort(s)[::-1]
  ind = np.argsort(s)[::-1]
  U = U[:, ind]
  s = s[ind]
  V = V[:, ind]

  for i in xrange(origShape[2]):
    likelihood_1d[:,i] = likelihood_1d[:,i] + L1d_master
  
  likelihood_pcomp = V.reshape(origShape)
    
  return likelihood_pcomp, s

def doLikelihoodPCAfit(pcaComp = None, master = None, eigenval = None, likelihood =None, n_component = 5, Lcut = 0.):

    # Perform least-squares: Find the best combination of master + pcaComps[:,:,0:n_component] that fits likelihood
    origShape = likelihood.shape
    L1d = likelihood - master
    L1d = likelihood.reshape(likelihood.size)
    pca1d = pcaComp.reshape( ( likelihood.size, pcaComp.shape[-1]) )
    pcafit = pca1d[:,0:(n_component)]
    m1d = np.reshape(master,master.size)
    #allfit = np.hstack((m1d[:,None], pcafit) )
    allfit = pcafit
    coeff, resid, _, _  = np.linalg.lstsq(allfit, L1d)
    bestFit = np.dot(allfit,coeff)
    bestFit2d = bestFit.reshape(likelihood.shape)
    bestFit2d = bestFit2d + master
    bestFit2d[bestFit2d < Lcut] = 0.
    return bestFit2d


def main(argv):
    parser = argparse.ArgumentParser(description = 'Perform magnitude distribution inference on DES data.')
    parser.add_argument('filter',help='filter name',choices=['g','r','i','z','Y'])
    parser.add_argument("-r","--reload",help='reload catalogs from DESDB', action="store_true")
    args = parser.parse_args(argv[1:])
    band = args.filter
    
    print "performing inference in band: "+args.filter
    print "Reloading from DESDM:", args.reload
    des, sim, truthMatched, truth, tileInfo = getCatalogs(reload = args.reload, band = args.filter)
    print "Excluding regions Eli says are bad."
    des, sim, truthMatched, truth = excludeBadRegions(des,sim, truthMatched, truth,band=band)
    print sim.size

    truth = truth[truth['mag'] > 0]
    keep = (truthMatched['mag'] > 0)
    sim = sim[keep]
    truthMatched = truthMatched[keep]

    HEALConfig = getHealConfig(map_nside = 64)
    print "Getting likelihood matrices for each HEALPixel"
    Likelihoods, HEALPixels, masterLikelihood, truth_bin_centers, obs_bin_centers = getAllLikelihoods(truth=truth, sim=sim,
                                                                                                      truthMatched = truthMatched,
                                                                                                      healConfig=HEALConfig ,doplot = True)
    

    # Solve for the pca components.
    print "Performing PCA fit over all the likelihoods."
    Lpca, pcaEigen = likelihoodPCA(likelihood = Likelihoods, likelihood_master = masterLikelihood)
    # Loop over the likelihoods again. Find the best low-n PCA fit to each.
    L_fit = Likelihoods * 0.
    # And make a plot showing the likelihood, the best fit, and the residual map.
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm, Normalize
    pp = PdfPages('likelihood_pca_fit-'+band+'.pdf')
    print "Making plot of all the likelihoods and their best fits"
    for i in xrange(HEALPixels.size):
        thisLike = Likelihoods[:,:,i]
        L_fit[:,:,i] = doLikelihoodPCAfit( pcaComp = Lpca, master = masterLikelihood, Lcut = 1e-3,
                                           eigenval =  pcaEigen, likelihood = thisLike, n_component = 6)
        fig,ax = plt.subplots(nrows=1,ncols=4,sharey=True,figsize = (20.,5.))
        im0 = ax[0].imshow(masterLikelihood, origin='lower',cmap=plt.cm.Greys, norm = LogNorm(vmin=1e-6,vmax=1),
                          extent = [truth_bin_centers[0],truth_bin_centers[-1],obs_bin_centers[0],obs_bin_centers[-1]])
        ax[0].set_xlabel('truth '+band+' mag.')
        ax[0].set_ylabel('measured '+band+' mag.')
        ax[0].set_title('Full SVA1 Balrog area')
        im1 = ax[1].imshow(Likelihoods[:,:,i], origin='lower',cmap=plt.cm.Greys, norm = LogNorm(vmin=1e-6,vmax=1),
                          extent = [truth_bin_centers[0],truth_bin_centers[-1],obs_bin_centers[0],obs_bin_centers[-1]])
        ax[1].set_xlabel('truth '+band+' mag.')
        ax[1].set_title('Balrog likelihood, \n HEALPixel='+str(HEALPixels[i])+', nside='+str(HEALConfig['out_nside']))
        im2 = ax[2].imshow(L_fit[:,:,i], origin='lower',cmap=plt.cm.Greys, norm = LogNorm(vmin=1e-6,vmax=1),
                          extent = [truth_bin_centers[0],truth_bin_centers[-1],obs_bin_centers[0],obs_bin_centers[-1]])
        ax[2].set_xlabel('truth '+band+' mag.')
        ax[2].set_title('PCA-smoothed Balrog likelihood')
        im3 = ax[3].imshow(Likelihoods[:,:,i] - L_fit[:,:,i], origin='lower', cmap=plt.cm.Greys, norm = Normalize(vmin=-1,vmax = 1),
                           extent = [truth_bin_centers[0],truth_bin_centers[-1],obs_bin_centers[0],obs_bin_centers[-1]])
        ax[3].set_xlabel('truth '+band+' mag.')
        ax[3].set_title('residuals')
        fig.colorbar(im2,ax=ax[2])
        fig.colorbar(im3, ax = ax[3])
        pp.savefig(fig)
    pp.close()

    import esutil
    print "Writing likelihoods to file masterLikelihoodFile.fits"
    esutil.io.write('masterLikelihoodFile.fits',Likelihoods, ext=1)
    esutil.io.write('masterLikelihoodFile.fits',L_fit,ext=2)
    esutil.io.write('masterLikelihoodFile.fits',HEALPixels, ext=3)
    
        
    print "Done."

    
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
