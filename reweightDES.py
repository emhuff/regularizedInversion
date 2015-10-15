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
import numpy.lib.recfunctions as rf
from sklearn.neighbors import NearestNeighbors as NN


def modestify(data):
    modest = np.zeros(len(data), dtype=np.int32)
    

    galcut = (data['flags_i'] <=3) & -( ((data['class_star_i'] > 0.3) & (data['mag_auto_i'] < 18.0)) | ((data['spread_model_i'] + 3*data['spreaderr_model_i']) < 0.003) | ((data['mag_psf_i'] > 30.0) & (data['mag_auto_i'] < 21.0)))
    modest[galcut] = 1

    starcut = (data['flags_i'] <=3) & ((data['class_star_i'] > 0.3) & (data['mag_auto_i'] < 18.0) & (data['mag_psf_i'] < 30.0) | (((data['spread_model_i'] + 3*data['spreaderr_model_i']) < 0.003) & ((data['spread_model_i'] +3*data['spreaderr_model_i']) > -0.003)))
    modest[starcut] = 3

    neither = -(galcut | starcut)
    modest[neither] = 5

    data = rf.append_fields(data, 'modtype', modest)
    print len(data), np.sum(galcut), np.sum(starcut), np.sum(neither)
    return data

def reweightMatch(rwt_tags = None, truthSample= None, matchSample = None, N_nearest = 100):

    if rwt_tags is None:
        rwt_tags = ['','']
    truthSample_arr = np.zeros(( len(truthSample), len(rwt_tags) ))
    matchSample_arr = np.zeros((len(matchSample), len(rwt_tags)))
    
    for thing,i in zip(rwt_tags,xrange(len(rwt_tags))):
        truthSample_arr[:,i] =  truthSample[thing]
        matchSample_arr[:,i] =  matchSample[thing]

    NP = calcNN(N_nearest, truthSample_arr,matchSample_arr)
    NT = calcNN(N_nearest, matchSample_arr, matchSample_arr)

    bad = (NP == 1)
    wts = NP * 1./NT
    wts[bad] = 0.

    return wts

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

    
    NP = np.asarray(NP)
    del nbrT, nbrP, distT, distP, indT, indP
    return NP
    

def perturb(cat, orig_size, tags = None, sigma = 0.1):
    newcat= np.random.choice(cat,size=orig_size)
    for thistag in tags:
        newcat[thistag] = newcat[thistag] + sigma*np.random.randn(orig_size)
    return newcat

def kdEst(val1,val2,x,y):
    import scipy.stats as st
    xx,yy = np.meshgrid(x,y)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    vals = np.vstack([val1,val2])
    kernel = st.gaussian_kde(vals)
    f = np.reshape(kernel(positions).T, xx.shape)
    return f, xx, yy


def getDES():
    import pyfits
    path = '../../Data/GOODS/'
    desAll = esutil.io.read(path+"des_i-griz.fits")
    #desAll = pyfits.getdata(path+"des_i-griz.fits")
    desAll = modestify(desAll)
    desStars = desAll[desAll['modtype'] == 3]

def getTruthStars():
    path = '../../Data/GOODS/training-stars.fits'
    data = esutil.io.read(path,ext=1)
    data = data[data["TRUE_CLASS"] == 1]
    #data = data[data['MAG_MODEL_I'] < 21]
    usable_fields = [10,11,12,15,19]
    cat = []
    for item in data:
        if item['FIELD'] in usable_fields:
            cat.append(item)
    cat = np.array(cat)
    return cat

def getConfidenceLevels(hist):
    h2, bins = np.histogram(hist,bins=500)
    centers = (bins[0:-1] + bins[1:])/2.
    c_arr = (np.cumsum(h2)*1./np.sum(h2))[::-1]
    clevels = np.interp([0.68, 0.95], c_arr,centers)
    stop
    return clevels

def main(argv):
    path = '../../Data/GOODS/'
    #desStars = np.random.choice(desStars,size=10000)

    balrogStars = esutil.io.read(path+"matched_i-griz.fits")
    desStars = getDES()
    #balrogStars = np.random.choice(balrogStars,size = 10000)
    desStars = getTruthStars()
    desKeep =( (desStars['MAG_AUTO_G'] < 50) & 
               (desStars['MAG_AUTO_R'] < 50) & 
               (desStars['MAG_AUTO_I'] < 50) & 
               (desStars['MAG_AUTO_Z'] < 50)   )

    des  = np.empty(desStars.size, dtype = [('g-r',np.float),('r-i',np.float),('i-z',np.float),('i',np.float),('r',np.float),('g',np.float)])
    balrog = np.empty(balrogStars.size, dtype = [('g-r',np.float),('r-i',np.float),('i-z',np.float),('i',np.float)])

    des['g-r'] = desStars['MAG_AUTO_G'] - desStars['MAG_AUTO_R']
    des['r-i'] = desStars['MAG_AUTO_R'] - desStars['MAG_AUTO_I']
    des['i-z'] = desStars['MAG_AUTO_I'] - desStars['MAG_AUTO_Z']
    des['i'] = desStars['MAG_AUTO_I']
    des['g'] = desStars['MAG_AUTO_G']
    des['r'] = desStars['MAG_AUTO_R']
    des = des[desKeep]

    balrog['g-r'] = balrogStars['mag_auto_g'] - balrogStars['mag_auto_r']
    balrog['r-i'] = balrogStars['mag_auto_r'] - balrogStars['mag_auto_i']
    balrog['i-z'] = balrogStars['mag_auto_i'] - balrogStars['mag_auto_z']
    balrog['i'] = balrogStars['mag_auto_i']
    

    wts = reweightMatch(truthSample = des, matchSample = balrog, rwt_tags = ['g-r','r-i','i-z'])
    keep = np.random.random(balrog.size) * np.max(wts) <= wts
    balrog_rwt = balrog[keep]

    balrog_out = balrogStars[keep]

    esutil.io.write('balrog-des-reweighted.fits',balrog_out,clobber=True)


    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,6))
    from matplotlib.colors import LogNorm, Normalize
    x_b = np.linspace(-1,3,100)
    y_b = np.linspace(-1,3,100)
    bContours, xx_b, yy_b = kdEst(balrog_out['mag_r']-balrog_out['mag_i'],balrog_out['mag_g'] - balrog_out['mag_r'],x_b,y_b)
    bLevels = getConfidenceLevels(bContours)
    cfset = ax1.contour(xx_b, yy_b, bContours, bLevels, cmap='Blues')
    
    # For labeling:
    import matplotlib.patches as mpatches
    red_patch  = mpatches.Patch(color='red', label='DES confirmed stars')
    blue_patch = mpatches.Patch(color='blue', label='deconvolved locus')
    bright = (des['i'] < 21.) & (des['r'] < 21.) & ( des['g'] < 21. )
    ax1.plot(des[bright]['r-i'],des[bright]['g-r'],',',lw=0,markersize=0.2,color='red',alpha=0.5)
    ax1.set_xlim(-1,2)
    ax1.set_ylim(-1,3)
    ax1.set_xlabel("r-i")
    ax1.set_ylabel("g-r")
    ax1.legend(loc='best',handles=[red_patch,blue_patch])
    
    x_r = np.linspace(-1,3,100)
    y_r = np.linspace(-2,3,100) 

    rContours, xx_r, yy_r = kdEst(balrog_out['mag_i'] - balrog_out['mag_z'],balrog_out['mag_r'] - balrog_out['mag_i'],x_r,y_r)
    rLevels = getConfidenceLevels(rContours)
    cfset = ax2.contour(xx_r, yy_r, rContours, rLevels, cmap='Blues')
    ax2.plot(des[bright]['i-z'],des[bright]['r-i'],',',lw=0,markersize=5,color='red',alpha=0.55)
    ax2.set_xlim(-1,2)
    ax2.set_ylim(-1,3)
    ax2.set_ylabel("r-i")
    ax2.set_xlabel("i-z")


    fig.savefig("des_deconvolved_locus.png")
    print "(iter 1) fraction of original sample kept: ",balrog_out.size * 1./balrog.size

    stop
    #plt.show()

if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

