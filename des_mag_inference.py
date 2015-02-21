#!/usr/bin/env python

import desdb
import numpy as np
import esutil
import pyfits
import sys
import healpy as hp
import os
import functions2
import slr_zeropoint_shiftmap as slr
import numpy.lib.recfunctions as rf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def doInference(sim_truth_catalog, sim_truth_matched_catalog, sim_obs_catalog, real_obs_catalog, lambda_reg = .01, tag='data', doplot = False, obs_bins = None, truth_bins = None):
    # --------------------------------------------------
    # Settle on a binning scheme for the data.
    # Use the Freedman-Diaconis rule, which suggests:
    #    dx = 2 * IQR(x)/n^(1/3)
    #    IQR is the interQuartile range, and n is the number of data points.
    if obs_bins is None:
        obs_binsize = 0.5#(np.percentile(sim_obs_catalog[tag],75) - np.percentile(sim_obs_catalog[tag],25))/(len(sim_obs_catalog[tag]))**(1/3.)
        obs_nbins = int(np.ceil( (np.max( sim_obs_catalog[tag]) - np.min( sim_obs_catalog[tag] ) ) / obs_binsize) + 1) 
        obs_bins = np.concatenate( (np.array([np.min( sim_obs_catalog[tag] )])-0.001*obs_binsize,np.array([np.min( sim_obs_catalog[tag] )]) + obs_binsize*np.arange(obs_nbins)) )
    else:
        obs_nbins = obs_bins.size-1
        obs_binsize = obs_bins[1]-obs_bins[0]
    # --------------------------------------------------
    # Next, create the data histogram arrays.
    N_sim_obs, _   = np.histogram(sim_obs_catalog[tag], bins = obs_bins)
    N_real_obs, _  = np.histogram(real_obs_catalog[tag], bins = obs_bins)
    #--------------------------------------------------
    # Important: we can only reconstruct the input histogram in bins from which objects are actually detected.
    if truth_bins is None:
        truth_binsize = .50# (np.percentile(sim_truth_matched_catalog[tag],75) - np.percentile(sim_truth_matched_catalog[tag],25))/(len(sim_truth_matched_catalog[tag]))**(1/3.)
        truth_nbins = int(np.ceil( (np.max(sim_truth_matched_catalog[tag]) - np.min(sim_truth_matched_catalog[tag] ) ) / truth_binsize) + 1) 
        truth_bins = np.concatenate( (np.array([np.min( sim_truth_matched_catalog[tag] )])-0.001*truth_binsize,
                                    np.array([np.min( sim_truth_matched_catalog[tag] )]) + truth_binsize*np.arange(truth_nbins)) )
    else:
        truth_nbins = truth_bins.size-1
        truth_binsize = truth_bins[1]-truth_bins[0]
    N_sim_truth, _ = np.histogram(sim_truth_catalog[tag], bins = truth_bins)
    N_sim_truth_sorted, _  = np.histogram(sim_truth_matched_catalog[tag], bins=truth_bins)
    # Now the entries in sorted_sim_truth should correspond
    # element-by-element to the entries in sim_obs_catalog, so we can
    # construct the arrays necessary to build the sum over the index
    # function.
    obs_bin_index = np.digitize(sim_obs_catalog[tag],obs_bins)-1
    truth_bin_index = np.digitize(sim_truth_matched_catalog[tag],truth_bins)-1
    indicator = np.zeros( (truth_nbins, obs_nbins) )
    A = np.zeros( (obs_nbins, truth_nbins) )

    # Finally, compute the response matrix.
    for i in xrange(obs_bin_index.size):
        if N_sim_truth_sorted[truth_bin_index[i]] > 0:
            A[obs_bin_index[i],truth_bin_index[i]] = A[obs_bin_index[i],truth_bin_index[i]]+1./N_sim_truth_sorted[truth_bin_index[i]]

    lambda_reg = .001
    lambda_reg_cov = 1e-12
    
    Cinv_data = np.diag(1./(N_real_obs+ lambda_reg_cov))
    #lambda_best =  setRegularizationParameter(A, Cinv_data, N_real_obs)
    Ainv_reg = np.dot( np.linalg.inv(np.dot( np.dot( np.transpose(A), Cinv_data ), A) + lambda_reg * np.identity( N_sim_truth.size ) ), np.dot( np.transpose( A ), Cinv_data) )
    

    # Everything.
    window = np.dot( Ainv_reg, N_sim_obs) / ( N_sim_truth + lambda_reg_cov)
    detFrac = N_sim_truth_sorted* 1.0 / (N_sim_truth + lambda_reg_cov)
    N_real_truth_nocorr = np.dot( Ainv_reg, N_real_obs)
    N_real_truth = N_real_truth_nocorr / window
    N_real_truth[~np.isfinite(N_real_truth)] = 0.
    
    Covar_orig = np.diag(N_real_truth + lambda_reg_cov)
    Amod = np.dot( Ainv_reg, A )
    Covar= np.dot(  np.dot(Amod, Covar_orig), np.transpose(Amod) )

    leakage = np.dot( (Amod - np.diag(Amod)) , N_real_truth)
    Covar_orig = Covar_orig
    errors = np.sqrt(np.diag(Covar) + leakage**2)
    truth_bins_centers = (truth_bins[0:-1] + truth_bins[1:])/2.
    obs_bins_centers = (obs_bins[0:-1] + obs_bins[1:])/2.
    if doplot:
        from matplotlib.colors import LogNorm
        plt.hist2d(sim_truth_matched_catalog['mag'],sim_obs_catalog['mag'],bins = (truth_bins,obs_bins),norm=LogNorm(),normed=True)
        plt.colorbar()
        plt.xlabel("truth magnitude")
        plt.ylabel("obs magnitude")
        plt.show(block=True)
    return N_real_truth, truth_bins_centers, truth_bins, obs_bins, errors



def NoSimFields(band='i'):
    q = """
    SELECT
        balrog_index,
        mag_auto as mag,
        flags
    FROM
        SUCHYTA1.balrog_sva1v2_nosim_%s
    """ %(band)
    return q



def SimFields(band='i',table='sva1v2'):
    q = """
    SELECT
        t.tilename as tilename,
        m.balrog_index as balrog_index,
        m.alphawin_j2000 as ra,
        m.deltawin_j2000 as dec,
        m.mag_auto as mag,
        t.mag as truth_mag,
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
           mag_auto_%s as mag,
           alphawin_j2000_%s as ra,
           deltawin_j2000_%s as dec,
           flags_%s as flags
        FROM
           sva1_coadd_objects
        WHERE
           tilename in %s
        """ % (band,band,band, band, str(tuple(np.unique(tilestuff['tilename']))))
    return q


def TruthFields(band='i', table = 'sva1v2'):
    q = """
    SELECT
        balrog_index,
        tilename,
        ra,
        dec,
        mag
    FROM
        SUCHYTA1.balrog_%s_truth_%s        
    """%(table,band)
    return q
    

def GetDESCat( depthmap, nside, tilestuff, tileinfo, band='i',depth = 50.0):
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
    


def cleanCatalog(catalog, tag='data'):
    # We should get rid of obviously wrong things.
    keep = np.where( (catalog[tag] > 15 ) & (catalog[tag] < 30) & (catalog['flags'] < 2) )
    return catalog[keep]


def removeBadTilesFromTruthCatalog(truth, tag='data', goodfrac = 0.8):
    tileList = np.unique(truth['tilename'])
    number = np.zeros(tileList.size)
    for tile, i in zip(tileList,xrange(number.size)):
        number[i] = np.sum(truth['tilename'] == tile)
    tileList = tileList[number > goodfrac*np.max(number)]
    keep = np.in1d( truth['tilename'], tileList )
    return truth[keep]

def GetFromDB( band='i', depth = 50.0):
    depthfile = '../sva1_gold_1.0.2-4_nside4096_nest_i_auto_weights.fits'
    slrfile = '../slr_zeropoint_shiftmap_v6_splice_cosmos_griz_EQUATORIAL_NSIDE_256_RING.fits'

    cur = desdb.connect()
    q = "SELECT tilename, udecll, udecur, urall, uraur FROM coaddtile"
    tileinfo = cur.quick(q, array=True)
    tilestuff = {}
    for i in range(len(tileinfo)):
        tilestuff[ tileinfo[i]['tilename'] ] = tileinfo[i]
    depthmap, nside = functions2.GetDepthMap(depthfile)
    slr_map = slr.SLRZeropointShiftmap(slrfile, band)
   
    q = TruthFields(band=band,table='sva1v2')
    truth = cur.quick(q, array=True)
    truth = removeBadTilesFromTruthCatalog(truth)
    truth = functions2.ValidDepth(depthmap, nside, truth, depth = depth)
    truth = functions2.RemoveTileOverlap(tilestuff, truth)
    slr_mag, slr_quality = slr_map.addZeropoint(band, truth['ra'], truth['dec'], truth['mag'], interpolate=True)
    truth['mag'] = slr_mag

    q = SimFields(band=band, table='sva1v2')
    sim = cur.quick(q, array=True)

    
    cut = np.in1d(sim['balrog_index'],truth['balrog_index'])
    sim = sim[cut]
    slr_mag, slr_quality = slr_map.addZeropoint(band, sim['ra'], sim['dec'], sim['mag'], interpolate=True)
    sim['mag'] = slr_mag
    sim = cleanCatalog(sim,tag='mag')

    des = GetDESCat(depthmap, nside, tilestuff, sim, band=band,depth = depth)
    slr_mag, slr_quality = slr_map.addZeropoint(band, des['ra'], des['dec'], des['mag'], interpolate=True)
    des['mag'] = slr_mag
    des = cleanCatalog(des, tag='mag')

    uind, simUniqueIndices = np.unique(sim['balrog_index'] , return_index = True)
    sim = sim[simUniqueIndices]
    
    sim = np.sort(sim,order='balrog_index')
    truth = np.sort(truth,order='balrog_index')
    truthMatched = truth[np.in1d(truth['balrog_index'], sim['balrog_index'], assume_unique=True)]
    sim = sim[np.in1d(sim['balrog_index'], truthMatched['balrog_index'], assume_unique = True)]

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


def makeHistogramPlots(hist_est, bin_centers, errors, catalog_real_obs, catalog_sim_obs, catalog_sim_truth,
                       catalog_sim_truth_matched,
                       bin_edges = None, tag='data'):

    hist_sim, _ = np.histogram(catalog_sim_truth[tag], bins = bin_edges)
    hist_sim_obs, _  = np.histogram(catalog_sim_obs[tag], bins = bin_edges)
    hist_obs, obs_bin_edges  = np.histogram(catalog_real_obs[tag],bins = bin_edges)
    obs_bin_centers = (obs_bin_edges[0:-1] + obs_bin_edges[1:])/2.

    peak_location = np.where(hist_obs == np.max(hist_obs) )[0]
    if len(peak_location) > 1:
        peak_location = peak_location[0]
    norm_factor = np.sum(hist_obs[(peak_location-2):(peak_location+2)])*1. / np.sum(hist_sim_obs[(peak_location-2):(peak_location+2)])
    hist_sim_renorm = hist_sim*norm_factor
    hist_sim_obs_renorm = hist_sim_obs*norm_factor
    print "Number of objects detected: ",catalog_real_obs.size
    print "Number of objects recovered: ", np.sum(hist_est)
    fig = plt.figure(1, figsize=(14,7))
    ax = fig.add_subplot(1,2,1)
    ax.semilogy(bin_centers, hist_est,'.', c='blue', label='inferred')
    ax.errorbar(bin_centers, hist_est,np.clip(errors,1.,1e9), c='blue',linestyle=".")
    ax.plot(obs_bin_centers, hist_obs, c='black',label='observed')
    ax.plot(bin_centers, hist_sim_renorm,c='green', label='simulated')
    ax.plot(bin_centers, hist_sim_obs_renorm, c = 'orange', label = 'sim. observed')
    ax.legend(loc='best')
    ax.set_ylim([100,1e6])
    ax.set_xlim([15,30])
    ax.set_ylabel('Number')
    ax.set_xlabel('magnitude')
    
    ax = fig.add_subplot(1,2,2)
    ax.axhspan(-.1,.1,facecolor='red',alpha=0.2)
    ax.axhline(y=0.0,color='grey',linestyle='--')
    ax.plot(bin_centers, (hist_est/(hist_sim_renorm+1e-12)-1),'.',color='blue')
    ax.errorbar(bin_centers, (hist_est/(hist_sim_renorm+1e-12)-1), errors/(hist_sim_renorm+1e-6), linestyle=".",c='blue')
    ax.set_xlabel('magnitude')
    ax.set_ylabel('normalized reconstruction residuals')
    ax.set_ylim([-1,1])
    ax.set_xlim([15,30])

    plt.show(block=True)
    


def HealPixifyCatalogs(catalog, HealConfig):
    
    HealInds = functions2.GetPix(HealConfig['out_nside'], catalog['ra'], catalog['dec'], nest=True)
    healCat = rf.append_fields(catalog,'HealInd',HealInds,dtypes=HealInds.dtype)

    return healCat



def getHealConfig(map_nside = 4096, out_nside = 128, depthfile = '../sva1_gold_1.0.2-4_nside4096_nest_i_auto_weights.fits'):
    HealConfig = {}
    HealConfig['map_nside'] = map_nside
    HealConfig['out_nside'] = out_nside
    HealConfig['finer_nside'] = map_nside
    HealConfig['depthfile'] = depthfile
    return HealConfig

          
def getEffectiveArea( catalog, areaMap, depthMap, HealConfig ,depth=0., rakey = 'ra',deckey = 'dec'):
    # create a list of tile indices from the catalog.
    catalogIndex = functions2.GetPix(HealConfig['out_nside'],catalog[rakey], catalog[deckey],nest=True)
    tileIndices = np.unique(catalogIndex)
    # Make a much finer grid.
    finer_index = np.arange(hp.nside2npix(HealConfig['finer_nside']))
    finer_theta, finer_phi = hp.pix2ang(HealConfig['finer_nside'], finer_index,nest=True)
    # Reject those tileIndices that aren't allowed by the depthmap(?)
    cut =  (areaMap > 0.0) & (depthMap > 0.0)
    areaMap = areaMap[cut]
    finer_pix = hp.ang2pix(HealConfig['out_nside'], finer_theta[cut], finer_phi[cut],nest=True)
    # Count the number of finer gridpoints inside each tileIndex.
    area = np.zeros(tileIndices.size)
    maxGridsInside =  np.power(HealConfig['map_nside']*1./HealConfig['out_nside'], 2.0)
    for tileIndex,i in zip(tileIndices,np.arange(tileIndices.size) ):
        area[i] = np.mean(areaMap[finer_pix == tileIndex] ) * hp.nside2pixarea(HealConfig['out_nside'],
                                                                               degrees=True) / maxGridsInside
    return area, tileIndices


def removeNeighbors(thing1, thing2, radius= 2./3600):
    # Returns the elements of thing 1 that are outside of the matching radius from thing 2
    
    depth=10
    h = esutil.htm.HTM(depth)
    m1, m2, d12 = h.match(thing1['ra'],thing1['dec'],thing2['ra'],thing2['dec'],radius,maxmatch=0)

    keep = ~np.in1d(thing1['balrog_index'],thing1['balrog_index'][m1])
    return keep


def makeTheMap(des=None, truth=None, truthMatched=None, sim=None, tileinfo = None,maglimits = [22.5, 24.5],band='i'):
    # Get the unique tile list.
    from matplotlib.collections import PolyCollection
    tiles = np.unique(truth['tilename'])
    reconBins = np.array([15.0, maglimits[0], maglimits[1],30])
    theMap = np.zeros(tiles.size) - 999
    mapErr = np.zeros(tiles.size)
    vertices = np.zeros((len(tiles), 4, 2))
    for i,tile in zip(xrange(len(tiles)),tiles):
        # find all galaxies in this tile.
        thisDES = des[des['tilename'] == tile]
        theseBalrog = truthMatched['tilename'] == tile
        thisTruthMatched = truthMatched[theseBalrog]
        thisSim = sim[theseBalrog]
        thisTruth = truth[truth['tilename'] == tile]
        # reconstruct the total number in the desired interval
        this_N_est, _, _, _, errors = doInference(thisTruth, thisTruthMatched, thisSim, thisDES,
                                                  truth_bins = reconBins,tag='mag',doplot=False)
        norm = np.sum( ( truth['mag'] > np.min(maglimits) ) & (truth['mag'] < np.max(maglimits) ))
        theMap[i] = this_N_est[1] * 1. / norm
        mapErr[i] = errors[1] * 1. / norm
        thisInfo = tileinfo[np.core.defchararray.equal(tileinfo['tilename'], tile)]
        ra_ll, ra_lr, ra_ur, ra_ul = thisInfo['urall'][0], thisInfo['uraur'][0], thisInfo['uraur'][0], thisInfo['urall'][0]
        dec_ll, dec_lr, dec_ur, dec_ul = thisInfo['udecll'][0], thisInfo['udecll'][0], thisInfo['udecur'][0], thisInfo['udecur'][0]
        vertices[i,:,0] = np.array((ra_ll,  ra_lr,  ra_ur,  ra_ul))
        vertices[i,:,1] = np.array((dec_ll, dec_lr, dec_ur, dec_ul))
        
    # Normalize the map to relative fluctuations.
    normedMap = theMap / np.median(theMap) - 1
    good = theMap > 0.01
    bad = ~good
    
    fig, ax = plt.subplots()
    coll = PolyCollection(vertices[good,:,:], array=normedMap[good], cmap = plt.cm.gray, edgecolors='none')
    badcoll = PolyCollection(vertices[bad,:,:],facecolors='red',edgecolors='none')
    ax.add_collection(coll)
    ax.add_collection(badcoll)
    ax.autoscale_view()
    ax.set_xlabel('ra')
    ax.set_ylabel('dec')
    ax.set_title('number density fluctuations, in range: ['+str(maglimits[0])+'< '+band+' <'+str(maglimits[1])+']')
    fig.colorbar(coll,ax=ax)
    fig.savefig("normalized_number_map")

    
    errMap = (theMap - np.median(theMap) ) / mapErr 
    fig, ax = plt.subplots()
    coll = PolyCollection(vertices[good,:,:], array=errMap[good], cmap = plt.cm.gray, edgecolors='none')
    badcoll = PolyCollection(vertices[bad,:,:],facecolors='red',edgecolors='none')
    ax.add_collection(coll)
    ax.add_collection(badcoll)
    ax.autoscale_view()
    ax.set_xlabel('ra')
    ax.set_ylabel('dec')
    ax.set_title('chi fluctuations, in range: ['+str(maglimits[0])+'< '+band+' <'+str(maglimits[1])+']')
    fig.colorbar(coll,ax=ax)
    fig.savefig("error_map")

    stop


def main(argv):
    # Get catalogs.
    band = 'r'
    des, sim, truthMatched, truth, tileInfo = getCatalogs(reload = False, band=band)
    
    # Find the inference for things with zero 
    #pure_inds =  removeNeighbors(sim, des) & ( truthMatched['mag']>0 )
    #truthMatched = truthMatched[pure_inds]
    #sim = sim[pure_inds]
    stop
    y = makeTheMap(des=des, truth=truth, truthMatched = truthMatched, sim=sim, tileinfo = tileInfo,band=band )
    # Infer underlying magnitude distribution for whole catalog.
    N_real_est, truth_bins_centers, truth_bins, obs_bins, errors = doInference(truth, truthMatched, sim, des,
                                                                               lambda_reg = .01, tag='mag', doplot = False)
    N_obs_all,_ = np.histogram(des['mag'],bins=truth_bins)
    N_obs_all = N_obs_all*1./truth.size
    N_real_est_all = N_real_est_all*1./truth.size
    makeHistogramPlots(N_real_est, truth_bins_centers, errors,
                       des, sim, truth, truthMatched, 
                       bin_edges = truth_bins, tag='mag')

    # --------------------------------------------------
    


    
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
