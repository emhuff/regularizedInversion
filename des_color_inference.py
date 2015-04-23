#!/usr/bin/env python

import matplotlib as mpl

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
    

def getSingleFilterCatalogs(reload=False,band='i'):

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





def modestify(data, band='i'):
    modest = np.zeros(len(data), dtype=np.int32)

    galcut = (data['flags_%s'%(band)] <=3) & -( ((data['class_star_%s'%(band)] > 0.3) & (data['mag_auto_%s'%(band)] < 18.0)) | ((data['spread_model_%s'%(band)] + 3*data['spreaderr_model_%s'%(band)]) < 0.003) | ((data['mag_psf_%s'%(band)] > 30.0) & (data['mag_auto_%s'%(band)] < 21.0)))
    modest[galcut] = 1

    starcut = (data['flags_%s'%(band)] <=3) & ((data['class_star_%s'%(band)] > 0.3) & (data['mag_auto_%s'%(band)] < 18.0) & (data['mag_psf_%s'%(band)] < 30.0) | (((data['spread_model_%s'%(band)] + 3*data['spreaderr_model_%s'%(band)]) < 0.003) & ((data['spread_model_%s'%(band)] +3*data['spreaderr_model_%s'%(band)]) > -0.003)))
    modest[starcut] = 3

    neither = -(galcut | starcut)
    modest[neither] = 5

    data = rf.append_fields(data, 'modtype_%s'%(band), modest)
    print len(data), np.sum(galcut), np.sum(starcut), np.sum(neither)
    return data



def getMultiBandCatalogs(reload=False, band1 = 'g', band2 = 'i'):

    
    des1, balrogObs1, balrogTruthMatched1, balrogTruth1, balrogTileInfo = getSingleFilterCatalogs(reload=reload, band=band1)
    des2, balrogObs2, balrogTruthMatched2, balrogTruth2, _              = getSingleFilterCatalogs(reload=reload, band=band2)


    # Now merge these across filters.
    des = mergeCatalogsUsingPandas(des1, des2, key='coadd_objects_id', suffixes = ['_'+band1,'_'+band2])    
    balrogObs = mergeCatalogsUsingPandas(balrogObs1, balrogObs2, key='balrog_index', suffixes = ['_'+band1,'_'+band2])    
    balrogTruthMatched = mergeCatalogsUsingPandas(balrogTruthMatched1, balrogTruthMatched2, key='balrog_index', suffixes = ['_'+band1,'_'+band2])
    balrogTruth = mergeCatalogsUsingPandas(balrogTruth1, balrogTruth2, key='balrog_index', suffixes = ['_'+band1,'_'+band2])

    des = modestify(des, band=band1)
    des = modestify(des, band=band2)
    balrogObs = modestify(balrogObs,band=band1)
    balrogObs = modestify(balrogObs,band=band2)
    
    balrogTruthMatched = modestify(balrogTruthMatched,band=band1)
    balrogTruthMatched = modestify(balrogTruthMatched,band=band2)

    
    # Finally, add colors.
    des = rf.append_fields(des, 'color_%s_%s'%(band1,band2), ( des['mag_auto_'+band1] - des['mag_auto_'+band2] ) )
    balrogObs = rf.append_fields(balrogObs, 'color_%s_%s'%(band1,band2), ( balrogObs['mag_auto_'+band1] - balrogObs['mag_auto_'+band2] ) )
    balrogTruthMatched = rf.append_fields(balrogTruthMatched, 'color_%s_%s'%(band1,band2), ( balrogTruthMatched['mag_auto_'+band1] - balrogTruthMatched['mag_auto_'+band2] ) )
    balrogTruth = rf.append_fields(balrogTruth, 'color_%s_%s'%(band1,band2), ( balrogTruth['mag_'+band1] - balrogTruth['mag_'+band2] ) )
    
    return des, balrogObs, balrogTruthMatched, balrogTruth, balrogTileInfo

def hpHEALPixelToRaDec(pixel, nside=4096, nest=True):
    theta, phi = hp.pix2ang(nside, pixel, nest=nest)
    ra, dec = convertThetaPhiToRaDec(theta, phi)
    return ra, dec

def hpRaDecToHEALPixel(ra, dec, nside=  4096, nest= True):
    phi = ra * np.pi / 180.0
    theta = (90.0 - dec) * np.pi / 180.0
    hpInd = hp.ang2pix(nside, theta, phi, nest= nest)
    return hpInd

def getGoodRegionIndices(catalog=None, badHPInds=None, nside=4096,band='i'):
    hpInd = hpRaDecToHEALPixel(catalog['ra_'+band], catalog['dec_'+band], nside=nside, nest= True)
    keep = ~np.in1d(hpInd, badHPInds)
    return keep


def excludeBadRegions(des,balrogObs, balrogTruthMatched, balrogTruth, band='i'):
    eliMap = hp.read_map("sva1_gold_1.0.4_goodregions_04_equ_nest_4096.fits", nest=True)
    nside = hp.npix2nside(eliMap.size)
    maskIndices = np.arange(eliMap.size)
    badIndices = maskIndices[eliMap == 1]

    obsKeepIndices = getGoodRegionIndices(catalog=balrogObs, badHPInds=badIndices, nside=nside, band=band)
    truthKeepIndices = getGoodRegionIndices(catalog=balrogTruth, badHPInds=badIndices, nside=nside,band=band)
    desKeepIndices = getGoodRegionIndices(catalog=des, badHPInds=badIndices, nside=nside,band=band)

    balrogObs = balrogObs[obsKeepIndices]
    balrogTruthMatched = balrogTruthMatched[obsKeepIndices]
    balrogTruth = balrogTruth[truthKeepIndices]
    des = des[desKeepIndices]

    return des,balrogObs, balrogTruthMatched, balrogTruth

def main(argv):
    band1 = 'g'
    band2 = 'r'

    des, balrogObs, balrogTruthMatched, balrogTruth, balrogTileInfo = getCatalogs(reload=False, band1=band1,band2=band2)
    des, balrogObs, balrogTruthMatched, balrogTruth = excludeBadRegions(des,balrogObs, balrogTruthMatched, balrogTruth,band=band2)

    

    
    import MCMC
    
    #truthcolumns = ['objtype_%s'%(band1), 'mag_%s'%(band1), 'mag_%s'%(band2)]
    #truthbins = [np.arange(0.5,5,2.0), np.arange(17.5,27,0.5),np.arange(17.5,27,0.5)]

    #measuredcolumns = ['modtype_%s'%(band1),'mag_auto_%s'%(band1), 'mag_auto_%s'%(band2)]
    #measuredbins=[np.arange(0.5, 7, 2.0), np.arange(17.5,27,0.5), np.arange(17.5,27,0.5)]

    truthcolumns = ['objtype_%s'%(band1), 'color_%s_%s'%(band1,band2), 'mag_%s'%(band2)]
    truthbins = [np.arange(0.5,5,2.0), np.arange(-4,4,0.5),np.arange(17.5,27,0.5)]

    measuredcolumns = ['modtype_%s'%(band1), 'color_%s_%s'%(band1,band2), 'mag_auto_%s'%(band2)]
    measuredbins=[np.arange(0.5, 7, 2.0), np.arange(-4,4,0.25), np.arange(17.5,27,0.25)]

    
    BalrogObject = MCMC.BalrogLikelihood(balrogTruth, balrogTruthMatched,
                                         truthcolumns = truthcolumns,
                                         truthbins = truthbins,
                                         measuredcolumns= measuredcolumns,
                                         measuredbins = measuredbins)
    nWalkers = 2000
    burnin = 1000
    steps = 1000
    
    ReconObject = MCMC.MCMCReconstruction(BalrogObject, des, MCMC.ObjectLogL,
                                          truth=balrogTruth, nWalkers=nWalkers, reg=1.0e-10)
    ReconObject.BurnIn(burnin)
    ReconObject.Sample(steps)

    print np.average(ReconObject.Sampler.acceptance_fraction)
    
    fig = plt.figure(1,figsize=(14,7))
    ax = fig.add_subplot(1,2, 1)
    where = [0, None]
    BalrogObject.PlotTruthHistogram1D(where=where, ax=ax, plotkwargs={'label':'BT-G', 'color':'Blue'})
    BalrogObject.PlotMeasuredHistogram1D(where=where, ax=ax, plotkwargs={'label':'BO-G', 'color':'Red'})
    ReconObject.PlotMeasuredHistogram1D(where=where, ax=ax, plotkwargs={'label':'DO-G', 'color':'Gray'})
    ReconObject.PlotReconHistogram1D(where=where, ax=ax, plotkwargs={'label':'DR-G', 'color':'black', 'fmt':'o', 'markersize':3})
    ax.legend(loc='best', ncol=2)
    ax.set_yscale('log')
    ax = fig.add_subplot(1,2, 2)
    where = [1, None, 1]
    BalrogObject.PlotTruthHistogram1D(where=where, ax=ax, plotkwargs={'label':'BT-G', 'color':'Blue'})
    BalrogObject.PlotMeasuredHistogram1D(where=where, ax=ax, plotkwargs={'label':'BO-G', 'color':'Red'})
    ReconObject.PlotMeasuredHistogram1D(where=where, ax=ax, plotkwargs={'label':'DO-G', 'color':'Gray'})
    ReconObject.PlotReconHistogram1D(where=where, ax=ax, plotkwargs={'label':'DR-G', 'color':'black', 'fmt':'o', 'markersize':3})
    ax.legend(loc='best', ncol=2)
    ax.set_yscale('log')
    fig.savefig("star-galaxy-magnitude-reconstruction")
    plt.show(block=True)

    fullRecon, fullReconErrs = ReconObject.GetReconstruction()
    nBins = np.array([thing.size for thing in truthbins])-1
    recon2d = np.reshape(fullRecon, nBins)
    err2d = np.reshape(fullReconErrs, nBins)
    stop

    
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
