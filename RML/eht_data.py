#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:52:32 2020

@author: Johanna
"""
import os
import cv2
import glob

# Very important to make sure kernel doesn't die!!!
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.datasets import mnist

import ehtim as eh # eht imaging package

from ehtim.observing.obs_helpers import *
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel # For tv regularization
from scipy.ndimage import median_filter
#import skimage.transform
# import helpers as hp
import csv
import sys
import datetime
import warnings

# mute the verbose warnings
warnings.filterwarnings("ignore")

# Import eht imaging package
import ehtim as eh 

plt.ion()
nsamp = 10000 #100000
npix = 32 
psize = 0

def Prepare_PGD_Data(eht_array='EHT2019', target='sgrA', dataset='mnist', th_noise=False, flux=None, data_augmentation=False, phase_error=False, amp_error=False ):
    '''
    Prepare the EHT training data for proximal gradient descent optimization methods.
    '''
    ###############################################################################
    # Define parameters
    ###############################################################################
    print("Preparing EHT Data...")
    fov_param = 100.0
    flux_label = 1
    blur_param = 0.25
    sefd_param = 1

    tint_sec = 5    # integration time in seconds
    tadv_sec = 600  # advance time between scans
    tstart_hr = 0   # GMST time of the start of the observation
    tstop_hr = 24   # GMST time of the end
    bw_hz = 4e9     # bandwidth in Hz
    
    add_th_noise = th_noise # False if you *don't* want to add thermal error. If there are no sefds in obs_orig it will use the sigma for each data point
    phasecal = not phase_error # True if you don't want to add atmospheric phase error. if False then it adds random phases to simulate atmosphere
    ampcal = not amp_error # True if you don't want to add atmospheric amplitude error. if False then add random gain errors 
    stabilize_scan_phase = False # if true then add a single phase error for each scan to act similar to adhoc phasing
    stabilize_scan_amp = False # if true then add a single gain error at each scan
    jones = False # apply jones matrix for including noise in the measurements (including leakage)
    inv_jones = False # no not invert the jones matrix
    frcal = True # True if you do not include effects of field rotation
    dcal = True # True if you do not include the effects of leakage
    dterm_offset = 0 # a random offset of the D terms is given at each site with this standard deviation away from 1
    dtermp = 0
    
    npix = 32
    
    ###############################################################################
    # Load EHT Array
    ###############################################################################  
    array = '/Users/Johanna/Desktop/Proximal Gradient Descent/arrays/' + eht_array + '.txt'
    eht = eh.array.load_txt(array)
  
    # Define observation field of view
    fov = fov_param * eh.RADPERUAS
    
    # define scientific target
    if target == 'm87':
        ra = 12.513728717168174
        dec = 12.39112323919932
    elif target == 'sgrA':
        ra = 19.414182210498385
        dec = -29.24170032236311

    ##########################################################################
    # Generate the discrete Fourier transform matrix for complex visibilities
    ##########################################################################
    rf = 230e9
    mjd = 57853 # day of observation
    # simim is the prior
    simim = eh.image.make_empty(npix, fov, ra, dec, rf=rf, source='random', mjd=mjd)
    simim.imvec = np.zeros((npix, npix, 1)).reshape((-1, 1))
    
    obs = simim.observe(eht, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz, add_th_noise=add_th_noise, ampcal=ampcal, phasecal=phasecal, 
                    stabilize_scan_phase=stabilize_scan_phase, stabilize_scan_amp=stabilize_scan_amp,
                    jones=jones,inv_jones=inv_jones,dcal=dcal, frcal=frcal, dterm_offset=dterm_offset)
    obs_data = obs.unpack(['u', 'v', 'vis', 'sigma'])
    
    
    uv = np.hstack((obs_data['u'].reshape(-1,1), obs_data['v'].reshape(-1,1)))
    
    # Extract forward model (Discrete Fourier Transform matrix)
    psize = simim.psize
    F_vis = ftmatrix(simim.psize, simim.xdim, simim.ydim, uv, pulse=simim.pulse)
    sigma_vis = obs_data['sigma']
    
    t1 = obs.data['t1']
    t2 = obs.data['t2']
    
    obs1 = obs
    ###############################################################################
    # generate the discrete Fourier transform matrices for closure phases
    ###############################################################################
    
    obs.add_cphase(count='max')
    # Extract forward models for telescopes 1, 2, and 3
    tc1 = obs.cphase['t1']
    tc2 = obs.cphase['t2']
    tc3 = obs.cphase['t3']
    
    sigma_cphase = obs.cphase['sigmacp']

    cphase_map = np.zeros((len(obs.cphase['time']), 3))

    zero_symbol = 10000
    for k1 in range(cphase_map.shape[0]):
        for k2 in list(np.where(obs.data['time']==obs.cphase['time'][k1])[0]):
            if obs.data['t1'][k2] == obs.cphase['t1'][k1] and obs.data['t2'][k2] == obs.cphase['t2'][k1]:
                cphase_map[k1, 0] = k2
                if k2 == 0:
                    cphase_map[k1, 0] = zero_symbol
            elif obs.data['t2'][k2] == obs.cphase['t1'][k1] and obs.data['t1'][k2] == obs.cphase['t2'][k1]:
                cphase_map[k1, 0] = -k2
                if k2 == 0:
                    cphase_map[k1, 0] = -zero_symbol
            elif obs.data['t1'][k2] == obs.cphase['t2'][k1] and obs.data['t2'][k2] == obs.cphase['t3'][k1]:
                cphase_map[k1, 1] = k2
                if k2 == 0:
                    cphase_map[k1, 1] = zero_symbol
            elif obs.data['t2'][k2] == obs.cphase['t2'][k1] and obs.data['t1'][k2] == obs.cphase['t3'][k1]:
                cphase_map[k1, 1] = -k2
                if k2 == 0:
                    cphase_map[k1, 1] = -zero_symbol
            elif obs.data['t1'][k2] == obs.cphase['t3'][k1] and obs.data['t2'][k2] == obs.cphase['t1'][k1]:
                cphase_map[k1, 2] = k2
                if k2 == 0:
                    cphase_map[k1, 2] = zero_symbol
            elif obs.data['t2'][k2] == obs.cphase['t3'][k1] and obs.data['t1'][k2] == obs.cphase['t1'][k1]:
                cphase_map[k1, 2] = -k2
                if k2 == 0:
                    cphase_map[k1, 2] = -zero_symbol

    F_cphase = np.zeros((cphase_map.shape[0], npix*npix, 3), dtype=np.complex64)
    cphase_proj = np.zeros((cphase_map.shape[0], F_vis.shape[0]), dtype=np.float32)
    for k in range(cphase_map.shape[0]):
        for j in range(cphase_map.shape[1]):
            if cphase_map[k][j] > 0:
                if int(cphase_map[k][j]) == zero_symbol:
                    cphase_map[k][j] = 0
                F_cphase[k, :, j] = F_vis[int(cphase_map[k][j]), :]
                cphase_proj[k, int(cphase_map[k][j])] = 1
            else:
                if np.abs(int(cphase_map[k][j])) == zero_symbol:
                    cphase_map[k][j] = 0
                F_cphase[k, :, j] = np.conj(F_vis[int(-cphase_map[k][j]), :])
                cphase_proj[k, int(-cphase_map[k][j])] = -1
    
    ###############################################################################
    # generate the discrete Fourier transform matrices for closure amplitudes
    ###############################################################################
    obs.add_camp(count='max')
    
    # debias = True or False??
    clamparr = obs.c_amplitudes(mode='all', count='max',
                                        vtype='vis', ctype='camp', debias=True, snrcut=0.0)
    

    uv1 = np.hstack((clamparr['u1'].reshape(-1, 1), clamparr['v1'].reshape(-1, 1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1, 1), clamparr['v2'].reshape(-1, 1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1, 1), clamparr['v3'].reshape(-1, 1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1, 1), clamparr['v4'].reshape(-1, 1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']
    
    mask = []
    # shape: (4, 2022, npix**2)
    F_camp = (ftmatrix(simim.psize, simim.xdim, simim.ydim, uv1, pulse=simim.pulse, mask=mask),
          ftmatrix(simim.psize, simim.xdim, simim.ydim, uv2, pulse=simim.pulse, mask=mask),
          ftmatrix(simim.psize, simim.xdim, simim.ydim, uv3, pulse=simim.pulse, mask=mask),
          ftmatrix(simim.psize, simim.xdim, simim.ydim, uv4, pulse=simim.pulse, mask=mask)
          )
    
    # shape: (2022)
    sigma_camp = obs.camp['sigmaca']
        
    ##########################################################################
    # Get training and testing data from dataset
    ##########################################################################
    xdata = []
    pad_width = 2
    if (dataset == 'fashion' or dataset == 'all'):
        npix = 32
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        xdata_train = 1.0*x_train[[k%60000 for k in range(int(nsamp*0.7))]]
        # # Adjust fov from 100 to 160
        # xdata_train = np.pad(xdata_train, ((0,0), (66,66), (66,66)), 'constant')  # get to 160x160
        xdata_train = np.pad(xdata_train, ((0,0), (pad_width,pad_width), (pad_width,pad_width)), 'constant')
        xdata_train = xdata_train[..., np.newaxis]/255
      
        xdata_test = 1.0*x_train[[k%60000 for k in range(int(nsamp*0.3))]]
        xdata_test = np.pad(xdata_test, ((0,0), (pad_width,pad_width), (pad_width,pad_width)), 'constant')  # get to 160x160
        xdata_test = xdata_test[..., np.newaxis]/255
        
        fashion_data = np.concatenate([xdata_train, xdata_test], 0).reshape((-1, npix*npix))
        # Adjust input flux ratio
        if flux != None:
            x_flux = np.sum(np.abs(fashion_data), axis=1)
            x_flux = x_flux.reshape((len(x_flux), 1))
            fashion_data = (flux/x_flux)*np.abs(fashion_data)
            
        if dataset == 'all' :
            xdata += list(fashion_data)
            print('Adding fashion dataset... length = ', len(fashion_data))
        else:
            xdata = fashion_data
            
    if (dataset == 'mnist' or dataset == 'all'):
        npix = 32
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
        
        xdata_train = 1.0*x_train_mnist[[k%60000 for k in range(int(nsamp*0.7))]]
        xdata_train = np.pad(xdata_train, ((0,0), (pad_width,pad_width), (pad_width,pad_width)), 'constant')  # get to 160x160
        xdata_train = xdata_train[..., np.newaxis]/255
        
        xdata_test = 1.0*x_test_mnist[[k%60000 for k in range(int(nsamp*0.3))]]
        xdata_test = np.pad(xdata_test, ((0,0), (pad_width,pad_width), (pad_width,pad_width)), 'constant')  # get to 32x32
        xdata_test = xdata_test[..., np.newaxis]/255
        
        mnist_data = np.concatenate([xdata_train, xdata_test], 0).reshape((-1, npix*npix))
        # Adjust input flux ratio
        if flux != None:
            x_flux = np.sum(np.abs(mnist_data), axis=1)
            x_flux = x_flux.reshape((len(x_flux), 1))
            mnist_data = (flux/x_flux)*np.abs(mnist_data)
            
        if dataset == 'all' :
            xdata += list(mnist_data)
            print('Adding mnist dataset... length = ', len(mnist_data))
        else:
            xdata = mnist_data
            
    if (dataset == 'bh_data' or dataset == 'all'):
        # Load full dataset (note: bh_sim_data.npy is a massive file)
        bh_sim_data = np.load('/Users/Johanna/Desktop/Neural Network/bh_sim_data.npy', allow_pickle=True).item()
        bh_data = bh_sim_data['image']
        
        # resize images to 32 x 32 and fov = 100
        bh_data = np.array(bh_data)
        bh_data_reshape = []
        for i in range(len(bh_data)):
            #bh_data_reshape.append(cv2.resize(bh_data[i].reshape((160, 160)).astype('float32'), (32, 32), interpolation = cv2.INTER_CUBIC).flatten())
            bh_img = eh.image.make_empty(160, 160, ra, dec, rf=rf, source='random', mjd=mjd)
            bh_img.imvec = bh_data[i].flatten()
            bh_img_reshape = bh_img.regrid_image(100, 32)
            bh_data_reshape.append(bh_img_reshape.imvec)
        bh_data = np.array(bh_data_reshape).reshape((-1, npix*npix))
        
        # Adjust input flux ratio
        if flux != None:
            x_flux = np.sum(np.abs(bh_data), axis=1)
            x_flux = x_flux.reshape((len(x_flux), 1))
            bh_data = (flux/x_flux)*np.abs(bh_data)

        if dataset == 'all' :
            xdata += list(bh_data)
            print("Adding black hole dataset... length = ", len(bh_data))
        else:
            xdata = bh_data

    # Pack DFT Matrices
    F_matrices = {'F_vis': F_vis, 'F_cphase': F_cphase, 'F_camp': F_camp}
    # Pack sigmas
    sigmas = {'sigma_vis': sigma_vis, 'sigma_cphase': sigma_cphase, 'sigma_camp': sigma_camp}

                  
    return xdata, obs1, simim, F_matrices, sigmas, [t1, t2], [tc1, tc2, tc3]

