#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:37:47 2020

@author: Johanna
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel # For tv regularization
import skimage.transform

def L1 (Z, prior):
    epsilon = 1e-16
    num = Z / np.sqrt(Z**2 + epsilon)
    denom = np.sqrt(prior**2 + epsilon) + epsilon

    l1grad = num / denom
    return l1grad

def L2 (Z, Z_grad, R=0, beta=1e-6):
    epsilon = 1e-16
    num = Z / (Z**2 + epsilon)
    denom = prior**2 + epsilon + epsilon

    l1grad = - num / denom
    return lgrad

def TV (Z, npix):    
    nx, ny = npix, npix
    epsilon = 1e-12
    norm = 1#psize
    im = Z.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    out = -np.sum(np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2) + epsilon)

    return out/norm

def TSV (Z, npix):
    I = Z.reshape(npix, npix)
   
    # Compute gradient in x and y-directions
    x_grad = sobel(I,axis=0,mode='constant').reshape(npix**2)
    y_grad = sobel(I,axis=1,mode='constant').reshape(npix**2)
    
    # Compute L2-norm of gradient
    eps = 1e-16
    tsv_norm = sum(np.square(eps**2 + np.square(x_grad) + np.square(y_grad)))
    
    # Compute TV gradient
    tsv_grad = ((np.abs(x_grad) + np.abs(y_grad)) / tsv_norm).reshape(npix**2)
    
    return tsv_grad

def simple_entropy(Z, prior, flux):
    # Mask: indices where imvec > 0
    Z = np.array(Z)
    mask = Z > np.zeros(len(Z)) 
    
    nprior = np.array([prior[i] if (mask[i] and prior[i] > 0) else 1 for i in range(len(prior))])
    nimvec = np.array([Z[i] if (mask[i] and prior[i] > 0) else 1 for i in range(len(Z))])
  
    entropy = -np.sum(nimvec*np.log(nimvec/nprior)) / flux
    
    entropy_grad = (np.log(nimvec/nprior) - 1)/ flux

    return entropy_grad

def tot_flux(Z, Z_grad, flux, R=0, beta=100):
    # Compute gradient of total image flux (known a priori)
    tot_flux = -(np.sum(Z) - flux)**2
    flux_grad = -2*beta*(np.sum(Z) - flux)
    Z_grad += flux_grad 
    R.append((beta, tot_flux))
    
    return Z_grad, R, flux_grad


def gauss (Z, psize):
    pass
    # #major, minor and PA are all in radians
    # phi, major, minor = 1.0, 1.0, 1.0
    # xdim, ydim = npix, npix

    # #computing eigenvalues of the covariance matrix
    # lambda1 = (minor**2.)/(8.*np.log(2.))
    # lambda2 = (major**2.)/(8.*np.log(2.))

    # #now compute covariance matrix elements from user inputs

    # sigxx_prime = lambda1*(np.cos(phi)**2.) + lambda2*(np.sin(phi)**2.) 
    # sigyy_prime = lambda1*(np.sin(phi)**2.) + lambda2*(np.cos(phi)**2.)
    # sigxy_prime = (lambda2 - lambda1)*np.cos(phi)*np.sin(phi) 

    # #we get the dimensions and image vector     
    # im = Z.reshape(xdim, ydim)
    # xlist, ylist = np.meshgrid(range(xdim),range(ydim))
    # xlist = xlist - (xdim-1)/2.0
    # ylist = ylist - (ydim-1)/2.0

    # xx = xlist * psize
    # yy = ylist * psize

    # #the centroid parameters
    # x0 = np.sum(xx*im) / np.sum(im)
    # y0 = np.sum(yy*im) / np.sum(im)

    # #we calculate the elements of the covariance matrix of the image 
    # sigxx = (np.sum((xx - x0)**2.*im)/np.sum(im))
    # sigyy = (np.sum((yy - y0)**2.*im)/np.sum(im))
    # sigxy = (np.sum((xx - x0)*(yy- y0)*im)/np.sum(im))

    # #now we compute the gradients of all quantities 
    # #gradient of centroid 
    # dx0 = ( xx -  x0) / np.sum(im)
    # dy0 = ( yy -  y0) / np.sum(im)

    # #gradients of covariance matrix elements 
    # dxx = (( (xx - x0)**2. - 2.*(xx - x0)*dx0*im ) - sigxx ) / np.sum(im) 

    # dyy = ( ( (yy - y0)**2. - 2.*(yy - y0)*dx0*im ) - sigyy ) / np.sum(im) 	

    # dxy = ( ( (xx - x0)*(yy - y0) - (yy - y0)*dx0*im - (xx - x0)*dy0*im ) - sigxy ) / np.sum(im) 

    # #gradient of the regularizer
    # drgauss = ( 2.*(sigxx - sigxx_prime)*dxx + 2.*(sigyy - sigyy_prime)*dyy + 4.*(sigxy - sigxy_prime)*dxy )
    # drgauss = drgauss/(major**2. * minor**2.)
    # drgauss = -drgauss.reshape(-1)
    
    # return drgauss, drgauss











    