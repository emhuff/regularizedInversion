import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

def buildBadRegionMap(sim, truth, nside=4096, nest = True, magThresh=1., HPIndices=None):
    '''
    Note that here, "truth" really means "truthMatched".
    '''
    npix = hp.nside2npix(nside)
    pixInd = np.arange(npix)
    simInd = hpRaDecToHEALPixel( sim['ra'],sim['dec'], nside=nside, nest= nest)
    magErr = np.abs(truth['mag'] - sim['mag'])
    badObj = magErr > magThresh
    binct_bad = np.bincount(simInd[badObj],minlength=npix)
    binct_tot = np.bincount(simInd, minlength = npix)
    regionMap = binct_bad * 1. / binct_tot
    regionMap[binct_tot == 0] = hp.UNSEEN
    return regionMap

def visualizeHealPixMap(theMap, nest=True, title="map", norm=None)
    
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    nside = hp.npix2nside(theMap.size)
    mapValue = theMap[theMap != hp.UNSEEN]
    indices = np.arange(theMap.size)
    seenInds = indices[theMap != hp.UNSEEN]

    print "Building polygons from HEALPixel map."
    vertices = np.zeros( (seenInds.size, 4, 2) )
    print "Building polygons for "+str(seenInds.size)+" HEALPixels."
    for HPixel,i in zip(seenInds,xrange(seenInds.size)):
        corners = hp.vec2ang( np.transpose(hp.boundaries(nside,HPixel,nest=True) ) )
        # HEALPix insists on using theta/phi; we in astronomy like to use ra/dec.
        vertices[i,:,0] = corners[1] * np.pi / 180.0
        vertices[i,:,1] = 90.0 - corners[0]*180.0/np.pi


    fig, ax = plt.subplots(figsize=(12,12))
    coll = PolyCollection(vertices, array = mapValue, cmap = plt.cm.gray, edgecolors='none')
    ax.add_collection(coll)
    ax.set_title(title)
    ax.autoscale_view()
    fig.colorbar(coll,ax=ax)
    print "Writing to file: "+title+".png"
    fig.savefig(title+".png",format="png")


