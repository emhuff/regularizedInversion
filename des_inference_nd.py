#!/usr/bin/env python
import matplotlib as mpl
#mpl.use('Agg')

import argparse
import matplotlib.pyplot as plt
import cfunc
import lfunc
import mapfunc
import sys
import numpy as np
import healpy as hp
import esutil



def main(argv):
    parser = argparse.ArgumentParser(description = 'Perform magnitude distribution inference on DES data.')
    parser.add_argument('filter',help='filter name',choices=['g','r','i','z','Y'])
    parser.add_argument("-r","--reload",help='reload catalogs from DESDB', action="store_true")
    parser.add_argument("-np","--no_pca",help='Do not use pca-smoothed likelihoods in reconstruction.', action="store_true")
    parser.add_argument("-ns","--nside",help='HEALPix nside for mapmaking',choices=['64','128','256','512','1024'])
    args = parser.parse_args(argv[1:])
    no_pca = args.no_pca
    if args.nside is not None:
        nside  = int(args.nside)
    else:
        nside = 64

    des, sim, truthMatched, truth = cfunc.getStellarityCatalogs(reload = args.reload, band = args.filter)
    truth = truth[truth['mag'] > 15.]
    des =des[des['mag_auto'] > 15.]
    keep = (sim['mag_auto'] > 15.) & (truthMatched['mag'] > 0.)
    sim = sim[keep]
    truthMatched = truthMatched[keep]

    # Choose the quantities to reconstruct.
    truthTags = ('mag','objtype')
    truthMagBins = np.linspace(15,24.5,25)#lfunc.chooseBins(catalog = truthMatched, tag = 'mag')
    truthTypeBins = np.array( [ 0,2, 4])
    truthBins = [truthMagBins, truthTypeBins]
    obsTags = ('mag_auto','modtype')
    obsMagBins = np.linspace(15,24,25) #lfunc.chooseBins(catalog = des, tag = 'mag_auto')
    obsTypeBins = np.array( [ 0,2, 4,6] )
    obsBins = [truthMagBins, truthTypeBins]
    
    
    # Measure the global likelihood function.
    print "Making global likelihood function."
    L = lfunc.makeLikelihoodMatrix( sim= sim, truth=truth, truthMatched = truthMatched, Lcut = 0., ncut = 0.,
                                    obs_bins = obsBins, truth_bins = truthBins, simTag = obsTags, truthTag = truthTags)
    fig, ax = plt.subplots()
    im = ax.imshow(np.arcsinh(L/0.001), origin='lower', cmap=plt.cm.Greys)
    ax.set_xlabel('truth mag/stellarity')
    ax.set_ylabel('obs mag/stellarity')
    fig.savefig("des_mag-stellarity_likelihood-"+args.filter+".png")
    fig.colorbar(im,ax = ax)
    fig.show()
    
    # Do the inversion inference.
    N_sim_obs, _ = np.histogramdd([des['mag_auto'],des['modtype']], bins = obsBins)
    N_obs_plot,_ = np.histogramdd([des['mag_auto'],des['modtype']], bins = truthBins)
    N_sim_truth, _ = np.histogramdd([truth['mag'], truth['objtype']], bins= truthBins)

    obsShape = N_sim_obs.shape
    truthShape = N_sim_truth.shape
    N_sim_obs_flat = np.ravel(N_sim_obs, order='F')
    N_sim_truth_flat = np.ravel(N_sim_truth, order='F')
    A = L.copy()
    lambda_reg = 0.0001
    Ainv = np.dot( np.linalg.pinv(np.dot(A.T, A) + lambda_reg * np.identity(N_sim_truth_flat.size ) ), A.T)
    #Ainv = np.linalg.pinv(A)
    N_est_flat = np.dot(Ainv, N_sim_obs_flat)
    N_est = np.reshape(N_est_flat, truthShape, order='F')
    fig,ax = plt.subplots()
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:])/2. , N_est[:,0],'--', label='galaxies (est.)')
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:])/2. , N_est[:,1],'--', label = 'stars (est)')
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:] )/2. , N_obs_plot[:,0], '.', label='galaxies (obs.)')
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:] )/2. , N_obs_plot[:,1], '.', label='stars (obs.)')
    N_gal_hist, _ = np.histogram(truth['mag'][truth['objtype'] == 1],bins=truthMagBins)
    N_star_hist, _ = np.histogram(truth['mag'][truth['objtype'] == 3], bins=truthMagBins)
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:])/2. , N_gal_hist  , label='galaxies (true)')
    ax.plot( (truthMagBins[0:-1] + truthMagBins[1:])/2., N_star_hist, label='stars (true)')
    ax.legend(loc='best')
    ax.set_yscale('log')
    ax.set_ylim([1,2.*np.max(N_sim_truth)])
    fig.savefig("nd-reconstruction_mag_stellarity-"+args.filter+".png")
    fig.show()
    stop

    


if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
