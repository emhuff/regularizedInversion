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


def getCOSMOS():
    data = esutil.io.read('matched_sva1_cosmos_roi.fits',ext=1)

    use =( (data['flags_r'] == 0) & (data['flags_i'] == 0) & 
           (data['flags_z'] == 0) & (data['flags_g'] == 0) & 
           (data['flags_acs'] == 0) )
    data = data[use]
    stars_cut = ( ((data['mag_auto_acs'] - data['mu_max_acs']) > 4.2 ) & (data['mu_max_acs'] < 21.5) & (data['mu_max_acs'] > 14.5) ) | ((data['mu_max_acs'] < 14.5) & (data['mag_auto_acs'] < 20) )
    stars = data[stars_cut]
    return stars

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

    #use = NP  > 1
    #wts = NP[use]*1./NT[use]
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

def main(argv):
    #desStars = esutil.io.read("des-stars.fits",columns=['mag_auto_g','mag_auto_r','mag_auto_i','mag_auto_z'])
    desStars = getCOSMOS()
    trilStars = esutil.io.read("trilegal_perturbed.fits")
    trilStars = np.random.choice(trilStars,size = 100000)

    desKeep =(  (desStars['mag_auto_g'] < 50) & 
                (desStars['mag_auto_r'] < 50) & 
                (desStars['mag_auto_i'] < 50) & 
                (desStars['mag_auto_z'] < 50)  )


    des  = np.empty(desStars.size, dtype = [('g-r',np.float),('r-i',np.float),('i-z',np.float),('i',np.float)])
    tril = np.empty(trilStars.size, dtype = [('g-r',np.float),('r-i',np.float),('i-z',np.float),('i',np.float)])

    des['g-r'] = desStars['mag_auto_g'] - desStars['mag_auto_r']
    des['r-i'] = desStars['mag_auto_r'] - desStars['mag_auto_i']
    des['i-z'] = desStars['mag_auto_i'] - desStars['mag_auto_z']
    des['i'] = desStars['mag_auto_i']
    des = des[desKeep]

    tril['g-r'] = trilStars['mag_g'] - trilStars['mag_r']
    tril['r-i'] = trilStars['mag_r'] - trilStars['mag_i']
    tril['i-z'] = trilStars['mag_i'] - trilStars['mag_z']
    tril['i'] = trilStars['mag_i']



    wts = reweightMatch(truthSample = des, matchSample = tril, rwt_tags = ['g-r','r-i','i-z'])
    keep = np.random.random(tril.size) * np.max(wts) <= wts
    tril_rwt = tril[keep]

    # do another fill-and-perturb step. This is much like approx. bayesian computation.
    tril2 = perturb(tril_rwt, orig_size = tril.size, sigma = 0.05, tags = ['g-r','r-i','i-z'])
    wts2 = reweightMatch(truthSample = des, matchSample = tril2, rwt_tags = ['g-r','r-i','i-z'])
    keep = np.random.random(tril.size) * np.max(wts2) <= wts2
    tril_rwt2 = tril2[keep]

    # Finally, do a single final perturbation step. This is the catalog we'll ultimately want to use as an input.
    tril3 = perturb(tril_rwt2, orig_size = tril.size, sigma=0.01,tags = ['g-r','r-i','i-z'])
    wts3 = reweightMatch(truthSample= des, matchSample = tril3, rwt_tags = ['g-r','r-i','i-z'])
    keep = np.random.random(tril3.size) * np.max(wts3) <= wts3
    tril3 = tril3[keep]
    tril3 = perturb(tril3, orig_size = tril.size, sigma=0.001, tags = ['g-r','r-i','i-z'])
    tril_out = trilStars.copy()

    tril_out['mag_g'] = tril3['g-r'] + tril3['r-i'] + tril3['i']
    tril_out['mag_r'] = tril3['r-i'] + tril3['i']
    tril_out['mag_i'] = tril3['i']
    tril_out['mag_z'] = tril3['i'] - tril3['i-z']
    esutil.io.write('trilegal-rematched.fits',tril_out,clobber=True)

    trilegal = np.loadtxt("output49704578313.dat")
    tri_r = trilegal[:,14]
    keep = tri_r < 24.5
    trilegal = trilegal[keep,:]
    tri_g_r = trilegal[:,12] - trilegal[:,13]
    tri_r_i = trilegal[:,13] - trilegal[:,14]
    tri_i_z = trilegal[:,14] - trilegal[:,15]

    fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(19,6))
    from matplotlib.colors import LogNorm, Normalize
    x_b = np.linspace(-1,3,100)
    y_b = np.linspace(-1,3,100)
    bContours, xx_b, yy_b = kdEst(tril_rwt2['r-i'],tril_rwt2['g-r'],x_b,y_b)
    #ax1.plot(tril_rwt2['r-i'],tril_rwt2['g-r'],',',lw=0,label='rwt',markersize=0.1,color='blue')
    #stuff = ax1.hist2d(tril_rwt2['r-i'],tril_rwt2['g-r'],bins = (np.linspace(-1,3,100),np.linspace(-1,3,100) ),norm=LogNorm(),cmap=plt.cm.Greys)
    cfset = ax1.contourf(xx_b, yy_b, bContours, cmap='Blues')
    
    ax1.plot(des['r-i'],des['g-r'],',',lw=0,label='des',markersize=0.2,color='red',alpha=0.5)
    ax1.plot(tri_r_i, tri_g_r, ',', markersize=0.3, color='orange',label='trilegal orig',alpha=0.25)
    ax1.set_xlim(-1,2)
    ax1.set_ylim(-1,3)
    ax1.set_xlabel("r-i")
    ax1.set_ylabel("g-r")
    ax1.legend(loc='best',markerscale=10)
    
    x_r = np.linspace(-1,3,100)
    y_r = np.linspace(-2,3,100) 
    rContours, xx_r, yy_r = kdEst(tril_rwt2['i-z'],tril_rwt2['r-i'],x_r,y_r)
    cfset = ax2.contourf(xx_r, yy_r, rContours, cmap='Blues')
    ax2.plot(des['i-z'],des['r-i'],',',label='des',lw=0,markersize=0.2,color='red',alpha=0.55)
    ax2.plot(tri_i_z, tri_r_i, ',', label='trilegal orig.',markersize=0.3, color='orange',alpha=0.25)
    ax2.set_xlim(-1,2)
    ax2.set_ylim(-2,3.)
    ax2.set_ylabel("r-i")
    ax2.set_xlabel("i-z")

    bContours, xx_b, yy_b = kdEst(tril_rwt2['r-i'],tril_rwt2['g-r'],x_b,y_b)
    cfset = ax3.contourf(xx_b, yy_b, bContours, cmap='Blues')
    ax3.plot(des['r-i'],des['g-r'],',',lw=0,label='des',alpha=0.5,markersize=0.5,color='red')
    ax3.legend(loc='best',markerscale=10)
    ax3.set_xlim(-1,3)
    ax3.set_ylim(-1,3)
    ax3.set_xlabel("r-i")
    ax3.set_ylabel("g-r")

    fig.savefig("reweighted_locus_comparison.png")
    print "(iter 1) fraction of original sample kept: ",tril_rwt.size * 1./tril.size
    print "(iter 2) fraction of original sample kept: ",tril_rwt2.size * 1./tril.size
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

