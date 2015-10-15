#!/usr/bin/env python
import matplotlib as mpl
#mpl.use('Agg')

import argparse
import matplotlib.pyplot as plt
import cfunc
import mapfunc
import sys
import numpy as np
import healpy as hp
import esutil
import atpy
import sklearn
from sklearn.neighbors import NearestNeighbors as NN

def get_cosmos_mask(path = '../../Data/COSMOS/', file = "zCOSMOS-mask.dat"):
    
    catalog = np.loadtxt(path+file,dtype=[('dec',np.float),('ra',np.float),('f1',np.float),
                                                   ('f2',np.float),('f3',np.float),('mag1',np.float),
                                                   ('f4',np.float),('mag2',np.float),('f5',np.int)])
    return catalog

def get_cosmos_morph_catalog(path = '../../Data/COSMOS/', file = "/cosmos_morphology_2005.tbl"):
    catalog = atpy.Table(path+file)
    return catalog

def get_cosmos_phot_catalog(path = '../../Data/COSMOS/', file = "cosmos_phot_20060103.tbl"):
    catalog = atpy.Table(path+file)
    # add a couple of color fields.
    ri = catalog['r_mag'] - catalog['i_mag']
    catalog.add_column('ri_color', ri)
    keep = (catalog['blend_mask'] == 0) &  (catalog['star'] == 0) & (catalog['auto_flag'] > -1)
    catalog = catalog[keep]

    mask = get_cosmos_mask(path=path)
    h = esutil.htm.HTM()
    rejDist = 10./3600.
    m1,m2,d12 = h.match(catalog['RA'],catalog['DEC'], mask['ra'],mask['dec'],rejDist,maxmatch=0)
    keep = ~np.in1d(np.arange(len(catalog)),m1)
    catalog = catalog[keep]

    return catalog

def get_cosmos_catalogs_all():
    pass


def get_des_truth_catalog(path = '../../Data/',  sample = 'bright'):
    
    if sample is 'bright':
        simfile = 'sim-58-21-22.fits'
        catalog = atpy.Table(path+simfile)
        ri = catalog['mag_r'] - catalog['mag_i']
        catalog.add_column('ri_color', ri)
    if sample is 'faint':
        simfile = 'sim-no-v3_2-23-24.fits'
        catalog = atpy.Table(path+simfile)    
        ri = catalog['mag_r'] - catalog['mag_i']
        catalog.add_column('ri_color', ri)

    
    return catalog

def reweightCosmos(rwt_tags = None, cosmos= None, des = None, sample = 'faint'):

    
    if des is None:
        des = get_des_truth_catalog(sample = sample)
    if cosmos is None:
        cosmos = get_cosmos_phot_catalog()
        if sample is 'bright':
            cut = ( cosmos['i_mag_auto'] < 22. ) & (cosmos['i_mag_auto'] > 21.)
        if sample is 'faint':
            cut = ( cosmos['i_mag_auto'] > 23. ) & (cosmos['i_mag_auto'] < 24.)
    # rwt_tags is a dictionary; the key is the field name in des, and
    # the value is the corresponding field name to match to in the
    # cosmos catalog.

    if rwt_tags is None:
        rwt_tags = {'ri_color':'ri_color','i_mag_auto':'mag_i'}
    des_arr = np.zeros(( len(des), len(rwt_tags.keys()) ))
    cosmos_arr = np.zeros((len(cosmos), len(rwt_tags.keys())))
    
    for thing,i in zip(rwt_tags.keys(),xrange(len(rwt_tags.keys()))):
        des_arr[:,i] =  des[rwt_tags[thing]]
        cosmos_arr[:,i] = cosmos[thing]

    NP = calcNN(50, des_arr,cosmos_arr)
    NT = calcNN(50, cosmos_arr, cosmos_arr)

    use = NP > 1
    wts = NP[use]*1./NT[use]

    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    bins = np.linspace(15,25,100)
    ax1.hist(des['mag_i'],bins=bins,label='des truth', normed=True)
    ax1.hist(cosmos[use]['i_mag_auto'],weights=wts,bins=bins,alpha=0.33,label='cosmos reweighted', normed=True)
    ax1.hist(cosmos[cut]['i_mag_auto'],bins=bins,alpha=0.33,label='cosmos unweighted', normed=True)
    ax1.set_xlabel('i mag')
    ax1.legend(loc='best')
    ax2.hist(des['mag_z'],bins=bins, label='des (truth)', normed=True)
    ax2.hist(cosmos[use]['z_mag'] + cosmos[use]['auto_offset'],bins=bins,weights=wts,alpha=0.33,label='cosmos reweighted', normed=True)
    ax2.hist(cosmos[cut]['z_mag'] + cosmos[cut]['auto_offset'],bins=bins,alpha=0.33,label='cosmos unweighted', normed=True)
    ax2.set_xlabel('z mag')
    ax2.legend(loc='best')
    fig.savefig("reweighted_magnitudes-faint")
    fig.show()
    stop
    wts = NP*1./NT
    cosmos = esutil.numpy_util.add_fields(cosmos,[('weight',wts.dtype)])
    cosmos['weight'] = wts
    cosmos = cosmos[wts > 0]
    esutil.io.write("weighted_cosmos_phot-faint.fits",cosmos)

def calcNN(Nnei, magP, magT):
    from sklearn.neighbors import NearestNeighbors as NN
    # Find Nnei neighbors around each point
    # in the training sample.
    # Nnei+1 because [0] is the point itself.
    nbrT        = NN(n_neighbors = Nnei+1).fit(magT)
    distT, indT = nbrT.kneighbors(magT, n_neighbors = Nnei+1)
   
    # Find how many neighbors are there around 
    # each point in the photometric sample 
    # within the radius that contains Nnei in 
    # the training one. 
    nbrP        = NN(radius = distT[:, Nnei]).fit(magP)
    distP, indP = nbrP.radius_neighbors(magT, radius = distT[:, Nnei])
   
    # Get the number of photometric neighbors
    NP = []
    for i in range(len(distP)): NP.append(len(distP[i])-1)
    #NP = aa(NP)
    
    NP = np.asarray(NP)
    del nbrT, nbrP, distT, distP, indT, indP
    return NP
    

def main(argv):
    reweightCosmos()

if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

