#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:22:16 2020

@author: Johanna

This script (1) extract the forward model (Fourier transform matricies) and
(2) implements regularized maximum likelihood method.
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
from scipy.signal import correlate2d
import skimage.transform
from sklearn.model_selection import ParameterGrid
# import helpers as hp
import csv
import sys
import datetime
import warnings

# Import function for loading dataset and measurements
from eht_data import Prepare_PGD_Data
# Import utilities for computing data terms, losses, and gradients
from data_term_functions import *
# Import regularizers
from regularizers import *

# mute the verbose warnings
warnings.filterwarnings("ignore")

plt.ion()
nsamp = 10000 #100000
npix = 32 # 160, 32(depends on dataset)
psize = 0

###############################################################################
# Helper Functions for Computing Chi^2, NRMSE, and Random/Gaussian Priors
###############################################################################
def total_chisq(Z, data_terms, alpha, F_matrices, sigmas):
    '''
        Returns the weighted sum of loss of data products and regularization terms.
    '''
    vis, amp, cphase, camp, lgcamp = 0, 0, 0, 0, 0
    if 'vis' in data_terms:
        vis = data_terms['vis']
    if 'amp' in data_terms:
        amp = data_terms['amp']
    if 'cphase' in data_terms:
        cphase = data_terms['cphase']
    if 'camp' in data_terms:
        camp = data_terms['camp']
    if 'lgcamp' in data_terms:
        lgcamp = data_terms['lgcamp']
    # Unpack F matrices
    F_vis = F_matrices['F_vis']
    F_cphase = F_matrices['F_cphase']
    F_camp = F_matrices['F_camp']
    
    # Unpack sigmas
    sigma_vis = sigmas['sigma_vis']
    sigma_cphase = sigmas['sigma_cphase']
    sigma_camp = sigmas['sigma_camp']
    
    # compute sum of coefficients
    total = np.sum(alpha)
    
    # Compute weighted sum of losses of data products 
    total_chisq = 0
    if 'vis' in data_terms:
        chisq = chisq_vis(vis, Z, F_vis, sigma_vis)
        total_chisq += alpha[0]/total * chisq
    if 'amp' in data_terms:
        chisq = chisq_amp(amp, Z, F_vis, sigma_vis)
        total_chisq += alpha[0]/total * chisq
    if 'cphase' in data_terms:
        chisq = chisq_cphase(cphase, Z, F_cphase, sigma_cphase)
        total_chisq += alpha[1]/total * chisq
    if 'camp' in data_terms:
        chisq = chisq_camp(camp, Z, F_camp, sigma_camp)
        total_chisq += alpha[2]/total * chisq
    if 'lgcamp' in data_terms:
        chisq = chisq_lgcamp(lgcamp, Z, F_camp, sigma_camp)
        total_chisq += alpha[2]/total * chisq

    return total_chisq

def NRMSE(X, Z):
    ''' 
    Returns the normalized root-mean-square error between two images A and B
    '''
    num = np.sqrt(np.sum(np.square(X - Z)))
    denom = np.sqrt(np.sum(np.square(X)))
    return (num/denom)

def random_Z(seed=None):
    # Initialize predicted image with random pixels
    np.random.seed(seed)
    
    Z = np.random.rand(X.shape[0])
        
    # Equalize flux to target image
    Z_flux = np.sum(np.abs(Z))
    Z = (flux/Z_flux)*np.abs(Z)
    
    return Z
        
def gauss_Z(empty_img, flux, seed=None):
    np.random.seed(seed)
    
    img = empty_img
    img.imvec = np.zeros((npix, npix, 1)).reshape((-1, 1))
    fwhm = 200*eh.RADPERUAS
    gauss_Z = img.add_gauss(flux=flux, beamparams= [fwhm, fwhm, 0, 0, 0],)
    
    # gauss_Z.display()
    
    return gauss_Z.imvec


###############################################################################
# Regularized Maximum Likelihood
###############################################################################
def RML(X, Z, data_terms, F_matrices, sigmas, max_epochs, 
                          alpha, reg, beta, verbose=False, early_stopping=True,
                          min_epochs=10, stopping_condition=1, patience=10, 
                          th_noise=False):
    '''
        Parameters:
            X - true image
            Z - initial predicted image (random or gaussian)
            F_matrices - Fourier transform matrices for all data terms: F_vis, F_cphase, F_camp
            sigmas - sigma arrays for all data terms: sigma_vis, sigma_cphase, sigma_camp
            max_epochs - maximum number of iterations of gradient descent
            alpha - array of weights for data terms, amplitude term followed by cphase term
            reg - array of regularizers to use as strings  
            beta - dictionary of weights for regularizers
            verbose - if True, then print losses and chi^2 per epoch
            early_stopping - if True, stop PGD once stopping condition is reached
            min_epochs - minimum number of iterations of gradient descent
            stopping_condition - minimum change in chi^2 after 10 epochs
            patience - number of epochs allowed of increasing chisq before early stopping
            th_noise - if True, then data includes thermal noise
            
        Returns:
            Z - final predicted image (1D array)
            losses - MSE per epoch
            chisq_vals - Chi^2 of each data term per epoch (dictionary)
    '''
    print('PATIENCE = ', patience)
    # Unpack data terms
    flux = data_terms['flux']
    prior = Z
    vis, amp, cphase, camp, lgcamp = 0, 0, 0, 0, 0
    if 'vis' in data_terms:
        vis = data_terms['vis']
    if 'amp' in data_terms:
        amp = data_terms['amp']
    if 'cphase' in data_terms:
        cphase = data_terms['cphase']
    if 'camp' in data_terms:
        camp = data_terms['camp']
    if 'lgcamp' in data_terms:
        lgcamp = data_terms['lgcamp']
    
    # Unpack F matrices
    F_vis = F_matrices['F_vis']
    F_cphase = F_matrices['F_cphase']
    F_camp = F_matrices['F_camp']
    
    # Unpack sigmas
    sigma_vis = sigmas['sigma_vis']
    sigma_cphase = sigmas['sigma_cphase']
    sigma_camp = sigmas['sigma_camp']
    
    # Train
    chisq_vals = {'total': [], 'vis': [], 'amp': [], 'cphase': [], 'camp': [],
                  'lgcamp': []}
    Z_copy = Z.copy()   # store best Z in case of early stopping
    prev_chisq = 0
    losses = [] # NRMSE
    prev_loss = 0 
    epoch = 1
    while (epoch < max_epochs):
        Z_grad = 0
        
        # Compute gradients of data terms
        gradients = "" # Store mean gradients used for updating Z_grad
        if 'vis' in data_terms:
            vis_grad = compute_vis_grad(vis, Z, F_vis)
            vis_grad =  np.dot(alpha[0], vis_grad)
            Z_grad += vis_grad
            gradients += ' vis_grad: {:0.3e}'.format(np.mean(vis_grad))
        if 'amp' in data_terms:
            amp_grad = compute_amp_grad(amp, Z, F_vis, sigma_vis)
            amp_grad = np.dot(alpha[0], amp_grad)
            Z_grad += amp_grad
            gradients += ' amp_grad: {:.3e}'.format(np.mean(amp_grad))
        if 'cphase' in data_terms:
            cphase_grad = compute_cphase_grad(cphase, Z, F_cphase, sigma_cphase, npix)
            cphase_grad = np.dot(alpha[1], cphase_grad)
            Z_grad += cphase_grad
            gradients += ' cphase_grad: {:.3e}'.format(np.mean(cphase_grad))
        if 'camp' in data_terms:
            camp_grad = compute_camp_grad(camp, Z, F_camp, sigma_camp)
            camp_grad = np.dot(alpha[2], camp_grad)
            Z_grad += camp_grad
            gradients += ' camp_grad: {:.3e}'.format(np.mean(camp_grad))
        if 'lgcamp' in data_terms:
            lgcamp_grad = compute_lgcamp_grad(lgcamp, Z, F_camp, sigma_camp)
            lgcamp_grad = np.dot(alpha[2], lgcamp_grad)
            Z_grad += lgcamp_grad
            gradients += ' lgcamp_grad: {:.3e}'.format(np.mean(lgcamp_grad))
            
        # Update predicted image
        Z = Z - Z_grad
        R_grad = 0
        
        # Regularization
        R = []
        regs = "" # Store mean regularizers 
        if("L1" in reg):
            L1_grad = L1(Z, prior)
            R_grad += beta['L1']*L1_grad
            regs += ' L1_grad: {:.3e}'.format(np.mean(beta['L1']*L1_grad))
        if("TV" in reg):
            tv_grad = TV(Z, npix)
            R_grad += beta['TV']*tv_grad
            regs += ' tv_grad: {:.3e}'.format(np.mean(beta['TV']*tv_grad))
        if("TSV" in reg):
            tsv_grad = TSV(Z, npix)
            R_grad += beta['TSV']*tsv_grad 
            regs += ' tsv_grad: {:.3e}'.format(np.mean(beta['TSV']*tsv_grad))
        if("entropy" in reg):
            ent_grad = simple_entropy(Z, prior, flux)
            R_grad += beta['entropy']*ent_grad
            regs += ' ent_grad: {:.3e}'.format(np.mean(beta['entropy']*ent_grad))
        if("tot_flux" in reg):
            R_grad, R, flux_grad = tot_flux(Z, Z_grad,flux, R, beta['tot_flux'])
            regs += ' flux_grad: {:.3e}'.format(np.mean(flux_grad))
        if("gauss" in reg):
            R_gauss, gauss_grad = gauss(Z, psize)
            R_grad += beta['gauss'] * gauss_grad
            regs += ' gauss_grad: {:.3e}'.format(np.mean(beta['gauss']*gauss_grad))
        if(epoch % 5000 == 0 and "gauss_blur" in reg):
            sigma = (beta['gauss_blur'], beta['gauss_blur'])
            Z = gaussian_filter(Z.reshape(npix, npix), sigma=sigma, order=0)
            Z = Z.flatten()
            
        # Update predicted image
        Z = Z - R_grad
        
        # Cutoff smallest pixel values to zero
        # MNIST: try with 1e-6!
        Z = np.maximum(np.zeros(Z.shape), Z - 1e-6)
        
        # Normalize Z between 0 and 1
        Z = np.maximum(np.zeros(np.shape(Z)), Z)
        Z = np.minimum(np.ones(np.shape(Z)), Z)
        
        
        # Show image every 100 epochs
        if(verbose and epoch % 100 == 0):
            pred_img = Z.reshape(npix, npix)
            plt.figure()
            plt.imshow(pred_img)
            plt.title('Epoch: ' + str(epoch))
            plt.show()
        
        # Compute NRMSE and Chi^2 of each data term
        curr_loss = NRMSE(X, Z)
        losses.append(curr_loss)
        curr_chisq = total_chisq(Z, data_terms, alpha, F_matrices, sigmas)
        chisq_vals['total'].append(curr_chisq)
        if 'vis' in data_terms:
            vis_chisq = chisq_vis(vis, Z, F_vis, sigma_vis)
            chisq_vals['vis'].append(vis_chisq)
        if 'amp' in data_terms:
            amp_chisq = chisq_amp(amp, Z, F_vis, sigma_vis)
            chisq_vals['amp'].append(amp_chisq)
        if 'cphase' in data_terms:
            cphase_chisq = chisq_cphase(cphase, Z, F_cphase, sigma_cphase)
            chisq_vals['cphase'].append(cphase_chisq)
        if 'camp' in data_terms:
            camp_chisq = chisq_camp(camp, Z, F_camp, sigma_camp)
            chisq_vals['camp'].append(camp_chisq)
        if 'lgcamp' in data_terms:
            lgcamp_chisq = chisq_lgcamp(lgcamp, Z, F_camp, sigma_camp)
            chisq_vals['lgcamp'].append(lgcamp_chisq)
        
        
        if verbose and (epoch % 10 == 0 or epoch == 1) :
            print(epoch, " NRMSE: ", round(curr_loss,6), ' chisq = ', round(curr_chisq, 3))
            #print(round((chisq_vals['total'][epoch-10] - curr_chisq), 3))
            
        # Early stopping 1: Change in chisq < stopping_condition for ten epochs
        if early_stopping and epoch > min_epochs and (np.abs(chisq_vals['total'][epoch-10] - curr_chisq) < stopping_condition):
            print("Early Stop at Epoch ", epoch)
            print("Early Stopping Condition 1: Change in chisq only ", (np.abs(chisq_vals['total'][epoch-10] - curr_chisq)), " for ten epochs")
            break
        # Early stopping 2: Change in chisq increasing for (patience) # epochs
        if early_stopping and epoch > patience+1 and (chisq_vals['total'][epoch-patience] - curr_chisq < 0):
            print("Early Stop at Epoch ", epoch)
            print("Early Stopping Condition 2: Chisq increased by ", chisq_vals['total'][epoch-patience] - curr_chisq, " over ", patience, 'epochs!')
            break
        
        Z_copy = Z.copy()
        prev_chisq = curr_chisq
        prev_loss = curr_loss
        epoch += 1
    
    # Normalize final flux
    Z_flux = np.sum(np.abs(Z))
    Z = (flux/Z_flux)*np.abs(Z)
    
    # Print final chi^2 and data term values
    if verbose:
        print()
        print("Epochs: ", epoch, ' \\\\')
        print("NRMSE: ", losses[-1], ' \\\\')
        if 'vis' in data_terms:
            vis_chisq = chisq_vis(vis, Z, F_vis, sigma_vis)
            print('vis chisq: ', vis_chisq, ' \\\\')
            chisq_vals['vis'].append(vis_chisq)
        if 'amp' in data_terms:
            amp_chisq = chisq_amp(amp, Z, F_vis, sigma_vis)
            print('amp chisq: ', amp_chisq, ' \\\\')
            chisq_vals['amp'].append(amp_chisq)
        if 'cphase' in data_terms:
            cphase_chisq = chisq_cphase(cphase, Z, F_cphase, sigma_cphase)
            print('cphase chisq: ', cphase_chisq, ' \\\\')
            chisq_vals['cphase'].append(cphase_chisq)
        if 'camp' in data_terms:
            camp_chisq = chisq_camp(camp, Z, F_camp, sigma_camp)
            print('camp chisq: ', camp_chisq, ' \\\\')
            chisq_vals['camp'].append(camp_chisq)
        if 'lgcamp' in data_terms:
            lgcamp_chisq = chisq_lgcamp(lgcamp, Z, F_camp, sigma_camp)
            print('lgcamp chisq: ', lgcamp_chisq, ' \\\\')
            chisq_vals['lgcamp'].append(lgcamp_chisq)
        print()

    return Z, losses, chisq_vals


###############################################################################
# Image Reconstruction Functions: Run regularized maximum likelihood using different
# data term combinations.
###############################################################################
def recon_vis(X, Z, flux, vis, F_matrices, sigmas, verbose=True, params=None):
    '''
        Run image reconstruction using amp, lgcamp, and clphase quantities
        from the original image. Returns the predicted image.
    '''
    max_epochs = 5000
    
    # Optimization parameters
    alpha = [1e-5]
    data_terms = {'flux': flux, 'vis': vis}
    reg = ['entropy', 'TSV', 'L1']
    beta = {'entropy': 1e-8, 'TSV': 1e-12, 'L1':1e-8}
    early_stop = 1e-7
    
    if (params != None):
        alpha, reg, beta, early_stop, max_epochs = params[0], params[1], params[2], params[3], params[4]
    
    Z1, losses, chisq_vals = RML(X, Z, data_terms, F_matrices, sigmas,
                                      max_epochs, alpha, reg, beta,
                                      verbose=verbose, early_stopping=True,
                                      min_epochs=50, stopping_condition=early_stop)
    
    return Z1, losses, chisq_vals
    
def recon_cphase(X, Z, flux, amp, cphase, F_matrices, sigmas, th_noise=False, verbose=True, params=None):
    '''
        Run image reconstruction using amp, lgcamp, and clphase quantities
        from the original image. Returns the predicted image.
    '''
    Z1, losses, chisq_vals = [], [], []
    
    # Default parameters, if params array is not provided
    alpha =  [1e-8, 1e-5] # amp, cphase 
    data_terms = {'flux': flux, 'cphase': cphase, 'amp': amp}
    reg = ['entropy', 'TSV', 'L1', 'gauss_blur']
    beta = {'entropy': 1e-8, 'TSV': 1e-12, 'L1':3e-8, 'gauss_blur': 0.5}
    early_stop = 1e-2
    max_epochs = 5000
    if th_noise:
        early_stop = 1e-8
    
    if (params != None):
            alpha, reg, beta, early_stop, max_epochs, patience = params[0], params[1], params[2], params[3], params[4], params[5]
    
    Z, losses, chisq_vals = RML(X, Z, data_terms, F_matrices, sigmas,
                                    max_epochs, alpha, reg, beta,
                                    verbose=verbose, early_stopping=True,
                                    min_epochs=50, stopping_condition=early_stop, 
                                    patience=patience, th_noise=th_noise)
    
  
    return Z, losses, chisq_vals

def recon_camp(X, Z, flux, amp, camp, cphase, F_matrices, sigmas, th_noise, verbose=True):
    '''
        Run image reconstruction using amp, lgcamp, and clphase quantities
        from the original image. Returns the predicted image.
    '''
    max_epochs = 5000
    
    Z1, losses, chisq_vals = [], [], []
    
    if th_noise:
        alpha =  [1e-8, 1e-5, 1e-18] # amp, cphase, camp
        data_terms = {'flux': flux, 'amp': amp, 'cphase': cphase, 'camp': camp}
        
        reg = ['entropy', 'TSV', 'L1', 'gauss_blur']
        beta = {'entropy': 1e-8, 'TSV': 1e-12, 'L1':3e-8, 'gauss_blur': 0.5}
        
        Z1, losses, chisq_vals = RML(X, Z, data_terms, F_matrices, sigmas,
                                          max_epochs, alpha, reg, beta,
                                          verbose=verbose, early_stopping=False,
                                          min_epochs=50, stopping_condition=1e-5)

    else:
        alpha =  [1e-8, 1e-5, 1e-8] # amp, cphase, camp
        data_terms = {'flux': flux, 'amp': amp, 'cphase': cphase, 'camp': camp}
        
        reg = ['entropy', 'TSV', 'L1']
        beta = {'entropy': 1e-5, 'TSV': 1e-4, 'L1':1e-5}
        
        Z1, losses, chisq_vals = RML(X, Z, data_terms, F_matrices, sigmas,
                                          max_epochs, alpha, reg, beta,
                                          verbose=True, early_stopping=True,
                                          min_epochs=10, stopping_condition=1e-1)
    
    return Z1, losses, chisq_vals
    
def recon_lgcamp(X, Z, flux, amp, lgcamp, cphase, F_matrices, sigmas, th_noise, verbose=True):
    '''
        Run image reconstruction using amp, lgcamp, and clphase quantities
        from the original image. Returns the predicted image.
    '''
    max_epochs = 5000
    
    if th_noise:
        alpha =  [1e-8, 1e-5, 1e-6] # amp, cphase, camp
        data_terms = {'flux': flux, 'amp': amp, 'cphase': cphase, 'lgcamp': lgcamp}
        
        reg = ['entropy', 'TSV', 'L1', 'gauss_blur']
        beta = {'entropy': 1e-8, 'TSV': 1e-12, 'L1':1e-8, 'gauss_blur': 0.5}
        
        Z1, losses, chisq_vals = RML(X, Z, data_terms, F_matrices, sigmas,
                                          max_epochs, alpha, reg, beta,
                                          verbose=verbose, early_stopping=False,
                                          min_epochs=50, stopping_condition=1e-4)

    else:
        alpha =  [1e-8, 1e-5, 1e-6] # amp, cphase, lgcamp
        data_terms = {'flux': flux, 'cphase': cphase, 'lgcamp': lgcamp, 'amp': amp}
        
        reg = ['entropy', 'TSV', 'L1']
        beta = {'entropy': 1e-3, 'TSV': 1e-3, 'L1':1e-4}
        
        Z1, losses, chisq_vals = RML(X, Z, data_terms, F_matrices, sigmas,
                                          max_epochs, alpha, reg, beta,
                                          verbose=True, early_stopping=True,
                                          min_epochs=10, stopping_condition=1)
    
    return Z1, losses, chisq_vals

###############################################################################
# Testing Hyperparameters Functions with Different Data Terms
###############################################################################
def test_hyperparameters_vis(X, Z, flux, vis, F_matrices, sigmas, res, simim, grid=None):
    # Create target image
    target_img = simim
    target_img.imvec = X.flatten()
    
    # Create parameter grid, in case none provided
    if grid is None:
        amp_vals = np.linspace(5e-7, 5e-5, 5)
        ent_vals = np.linspace(5e-8, 5e-6, 5)
        early_stop = [1e-10, 1e-9, 1e-8]
        max_epochs = [10000]
        param_grid = {'amp_val': amp_vals, 'ent_val': ent_vals,
                  'early_stop': early_stop, 'max_epochs': max_epochs}
        grid = ParameterGrid(param_grid)

    # Run through parameter combinations and store predicted images
    i = 1
    print("Running through hyperparameter combinations...")
    recon_imgs = []
    losses = []
    for params in grid:
        alpha = [params['amp_val']]
        reg = ['entropy']
        beta = {'entropy': params['ent_val']}
        early_stop, max_epochs = params['early_stop'], params['max_epochs']
        params_arr = [alpha, reg, beta, early_stop, max_epochs]
        
        Z, recon_losses, chisq_vals = recon_vis(X, Z, flux, vis, F_matrices, sigmas, verbose=False, params=params_arr)
        
        recon_imgs.append(Z)
        losses = [recon_losses]
        print("     params ", i, "/", len(grid))
        i += 1
        
    return recon_imgs, losses

def test_hyperparameters_cphase(X, Z, flux, vis_amp, cphase, F_matrices, sigmas, res, simim, grid=None):
    # Create target image
    target_img = simim
    target_img.imvec = X.flatten()
    
    # Create parameter grid, in case none provided
    if grid is None:
        # Set variable values
        amp_vals = np.linspace(5e-9, 5e-8, 5)
        cphase_vals = np.linspace(5e-6, 5e-5, 5)
        reg = ['entropy', 'TSV', 'L1', 'gauss_blur']
        beta = {'entropy': params['ent_val'], 'TSV': params['tsv_val'], 
                'L1': params['l1_val'], 'gauss_blur': params['gauss_val']}
        early_stop = [1e-7, 1e-6, 1e-5, 1e-4]
        max_epochs = [1000]
        param_grid = {'amp_val': amp_vals, 'cphase_val': cphase_vals,
                      'early_stop': early_stop, 'max_epochs': max_epochs}
        grid = ParameterGrid(param_grid)
    
    # Run through parameter combinations and store predicted images
    i = 1
    print("Running through hyperparameter combinations...")
    recon_imgs = []
    losses = []
    for params in grid:
        alpha = [params['amp_val'], params['cphase_val']]
        reg = ['entropy', 'TSV', 'L1', 'gauss_blur']
        beta = {'entropy': 1e-8, 'TSV': 1e-12, 'L1':3e-8, 'gauss_blur': 0.5}
        early_stop, max_epochs = params['early_stop'], params['max_epochs']
        params_arr = [alpha, reg, beta, early_stop, max_epochs]
        
        Z, recon_losses, chisq_vals = recon_cphase(X, Z, flux, vis_amp, cphase, F_matrices, sigmas, verbose=False, params=params_arr)
        
        recon_imgs.append(Z)
        losses = [recon_losses]
        print("     params ", i, "/", len(grid))
        i += 1
        
    return recon_imgs, losses

###############################################################################
#  Main Function
###############################################################################
if __name__ == '__main__':    
    '''
        Main Function
    '''

    # Set observation variables
    eht_array = 'EHT2019'
    target = 'm87'      #'sgrA'#'both'#
    dataset = 'bh_data'   #'fashion', 'mnist', 'bh_data'
    flux=1
    th_noise = True    # add thermal noise
    
    # Get observation data
    xdata, obs, simim, F_matrices, sigmas = Prepare_PGD_Data(eht_array, target, dataset, th_noise, flux=flux)
    
    #obs.plotall('uvdist','amp')
    # Select a sample image 
    # MNIST: (idx, digit), (1, "0"), (6, "1"), (5, "2"),  (7, "3"), 
            #              (18, "6"), (15, "7"), (17, "8"), (4, "9")
    # BH_DATA: 0, 2500, 3003
    idx = 0
    X = xdata[idx]
    
    X_img = simim
    X_img.imvec = X.reshape((-1, 1))
    
    pred_img = X.reshape(npix, npix)
    plt.figure()
    plt.imshow(pred_img)
    plt.title('Initial Image')
    plt.colorbar()
    plt.show()
    
    # Compute and pack data terms of target image
    flux = X_img.total_flux() # In Jy, equivalent to np.sum(X)
    vis = compute_vis(X, F_matrices['F_vis'])
    amp = compute_amp(X, F_matrices['F_vis'])
    cphase = compute_cphase(X, F_matrices['F_cphase'])
    camp = compute_camp(X, F_matrices['F_camp'])
    lgcamp = compute_lgcamp(X, F_matrices['F_camp'])
    
    save = False
    filename = "test"

    # Command line arguments provided
    if (len(sys.argv) > 1):
        # Save image, obs, and summary
        save = True
        filename = sys.argv[1]
    
    # Gaussian prior
    Z = gauss_Z(simim, flux)
    pred_img = Z_recon2.reshape(npix, npix)
    plt.figure()
    plt.imshow(pred_img)
    plt.title('Reconstructed Image (Amplitudes and Closure Phases)')
    plt.colorbar()
    plt.show()
    
    # Z = pre_img.imvec 
    # pre_img.display()
    
    # Reconstruction 1: Complex Visibilities
    # print("Reconstruction 1: Complex Visibilities")
    # Z_recon1, losses1, chisq_vals1 = recon_vis(X, Z, flux, vis, F_matrices, sigmas)

    # pred_img = Z_recon1.reshape(npix, npix)
    # plt.figure()
    # plt.imshow(pred_img)
    # plt.title('Reconstructed Image (Complex Visibilities)')
    # plt.colorbar()
    # plt.show()
    
    # Reconstruction 2: Amplitudes and Closure Phases
    # tune_hyperparameters(Z)
    print("Reconstruction: Amplitudes and Closure Phases")
    Z_recon2, losses2, chisq_vals2 = recon_cphase(X, Z, flux, amp, cphase, F_matrices, sigmas, th_noise)
    
    pred_img = Z_recon2.reshape(npix, npix)
    plt.figure()
    plt.imshow(pred_img)
    plt.title('Reconstructed Image (Amplitudes and Closure Phases)')
    plt.colorbar()
    plt.show()

    # Reconstruction 3: Closure Phases, Closure Phases, Amplitudes
    # print("Reconstruction 3: Closure Phases, Closure Phases, Amplitudes")
    # Z_recon3, losses3, chisq_vals3 = recon_camp(X, Z, flux, amp, camp, cphase, F_matrices, sigmas, th_noise)
    
    # pred_img = Z_recon3.reshape(npix, npix)
    # plt.figure()
    # plt.imshow(pred_img)
    # plt.title('Reconstructed Image (Closure Amplitudes)')
    # plt.colorbar()
    # plt.show()
    
    # Reconstruction 4: Log Closure Phases, Closure Phases, Amplitudes
    # print("Reconstruction 4: Log Closure Phases, Closure Phases, Amplitudes")
    # Z_recon4, losses4, chisq_vals4 = recon_lgcamp(X, Z, flux, amp, lgcamp, cphase, F_matrices, sigmas, th_noise, verbose=True)
    
    # pred_img = Z_recon4.reshape(npix, npix)
    # plt.figure()
    # plt.imshow(pred_img)
    # plt.title('Reconstructed Image (Log Closure Amplitudes)')
    # plt.colorbar()
    # plt.show()
    
    # pred_img = xdata[0].reshape(32, 32)
    # plt.figure()
    # plt.imshow(pred_img)
    # plt.title('Resized Black Hole Image')
    # plt.colorbar()
    # plt.show()
    
    # Plot losses
    plt.figure()
    plt.plot(losses[10:len(losses)], '-')
    plt.title("NRMSE vs. Epoch")
    plt.show()

    # Save Image, Obs, and Image Summary
    # if save:
    #     img = simim
    #     img.imvec = Z_recon1.reshape((-1, 1))
    #     img.save_fits(filename + ".fits")
    #     obs.save_txt(filename+"_obs.txt")


