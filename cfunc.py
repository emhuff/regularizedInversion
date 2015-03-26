#!/usr/bin/env python

# A library of stuff used for getting Balrog data from the databse,
# HEALPixifying it, and general munging.


import desdb
import numpy as np
import esutil
import sys
import healpy as hp
import os
import numpy.lib.recfunctions as rf


def GetDepthMap(depth_file):
    map = hp.read_map(depth_file, nest=True)
    nside = hp.npix2nside(map.size)
    return map, nside


def GetPhi(ra):
    return ra * np.pi / 180.0

def GetRa(phi):
    return phi*180.0/np.pi

def GetTheta(dec):
    return (90.0 - dec) * np.pi / 180.0

def GetDec(theta):
    return 90.0 - theta*180.0/np.pi

def GetRaDec(theta, phi):
    return [GetRa(phi), GetDec(theta)]

def GetPix(nside, ra, dec, nest=True):
    phi = GetPhi(ra)
    theta = GetTheta(dec)
    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    return pix

def GetDepthCut(map, nside, ra, dec, depth = 0.0):
    pix = GetPix(nside, ra, dec)
    depths = map[pix]
    ok_depths =  (depths > depth)
    return ok_depths

def ValidDepth(map, nside, arr, rakey='ra', deckey='dec', depth = 0.0):
    ok_depths = GetDepthCut(map, nside, arr[rakey], arr[deckey], depth = depth)
    arr = arr[ok_depths]
    return arr

def InTile(data, ura, udec, rakey='ra', deckey='dec'):
    inside = (data[rakey] > ura[0]) & (data[rakey] < ura[1]) & (data[deckey] > udec[0]) & (data[deckey] < udec[1])
    return inside

def RemoveTileOverlap(tilestuff, data, col='tilename', rakey='ra', deckey='dec'):
    datatile = data[col]
    tiles = np.unique(datatile)
    keep = np.zeros( len(data), dtype=np.bool_)
    for tile in tiles:
        cut = (datatile==tile)
        entry = tilestuff[tile]
        ura = (entry['urall'], entry['uraur'])
        udec = (entry['udecll'], entry['udecur'])
        u = InTile(data[cut], ura, udec, rakey=rakey, deckey=deckey)
        keep[cut] =  u
    return data[keep]

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
    detcat = ValidDepth(depthmap, nside, detcat, rakey='ra', deckey='dec',depth = depth)
    detcat = RemoveTileOverlap(tilestuff, detcat, col='tilename', rakey='ra', deckey='dec')
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
    depthmap, nside = GetDepthMap(HealConfig['depthfile'])
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
    depthmap, nside = GetDepthMap(depthfile)
    truths = []
    sims = []
    truthMatcheds = []
    
    for tableName in tables:
        q = TruthFields(band=band,table=tableName)
        truth = cur.quick(q, array=True)

        truth = removeBadTilesFromTruthCatalog(truth)
        truth = ValidDepth(depthmap, nside, truth, depth = depth)
        truth = RemoveTileOverlap(tilestuff, truth)
        truth = cleanCatalog(truth,tag='mag')
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
    HealInds = hpRaDecToHEALPixel( catalog[ratag],catalog[dectag], nside= healConfig['out_nside'], nest= healConfig['nest'])    
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


def removeNeighbors(thing1, thing2, radius= 2./3600):
    # Returns the elements of thing 1 that are outside of the matching radius from thing 2
    
    depth=10
    h = esutil.htm.HTM(depth)
    m1, m2, d12 = h.match(thing1['ra'],thing1['dec'],thing2['ra'],thing2['dec'],radius,maxmatch=0)

    keep = ~np.in1d(thing1['balrog_index'],thing1['balrog_index'][m1])
    return keep


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


def getCleanCatalogs( reload = False, band=None, isolated=False, nside=None):
    
    if band is None:
        print "Please specify a filter using the 'band' keyword argument."
        print "Choose one of [grizY]."
        
        
    des, balrogObs, balrogTruthMatched, balrogTruth, balrogTileInfo = getCatalogs(reload=reload, band=band)

    # Remove things in regions that are officially masked.
    des, balrogObs, balrogTruthMatched, balrogTruth = excludeBadRegions(des,balrogObs, balrogTruthMatched, balrogTruth, band=band)
    
    # if the isolated keyword is set, exclude things within some
    # separation from a pre-existing DES detection.
    if isolated is not False:
        print "isolated keyword should be set to a number, in arcseconds."
        keep = removeNeighbors(balrogTruthMatched, des, radius= float(isolated)/3600)
        balrogTruthMatched = balrogTruthMatched[keep]
        balrogObs = balrogObs[keep]
    
    # nside is set, then create a HEALPixel configuration and
    # assign pixel indices to each object.
    if nside is not None:
        HEALConfig = getHealConfig(map_nside = 4096, out_nside = nside )
        des = HealPixifyCatalogs(catalog=des, healConfig=HEALConfig)
        balrogObs = HealPixifyCatalogs(catalog=balrogObs, healConfig=HEALConfig)
        balrogTruth = HealPixifyCatalogs(catalog=balrogTruth, healConfig=HEALConfig)
        balrogTruthMatched = HealPixifyCatalogs(catalog=balrogTruthMatched, healConfig=HEALConfig)
        return des, balrogObs, balrogTruthMatched, balrogTruth,HEALConfig
    else:
        return des, balrogObs, balrogTruthMatched, balrogTruth
