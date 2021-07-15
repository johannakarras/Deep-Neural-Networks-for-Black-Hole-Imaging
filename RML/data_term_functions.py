#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:35:10 2020

@author: Johanna
"""
import numpy as np

###############################################################################
# Complex Visibility Functions
###############################################################################

def compute_vis(X, F):
    vis = np.matmul(X, np.transpose(F)).astype(np.complex64)
    return vis

def compute_vis_grad(vis, Z, F):
    Z_vis = compute_vis(Z, F)
    grad = -np.matmul(np.conjugate(F.T), vis - Z_vis)
    return grad.real

def chisq_vis(vis, Z, F, sigma):
    ''' 
        Compute mean chi-squared of visibilities of Z.
    '''
    samples = compute_vis(Z, F)
    chisq = np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))
    return chisq

###############################################################################
# Visibility Amplitude Functions
###############################################################################
    
def compute_amp(X, F):
    ''' Given an image X and DFT matrix F, compute and return its 
        visibility amplitude. '''
    amp = np.abs(np.dot(F, X))
    return amp

def compute_amp_grad(amp, Z, A, sigma):
    ''' 
        Compute gradient of visibility amplitude.
    '''
    i1 = np.dot(A, Z)
    amp_samples = np.abs(i1)

    pp = ((amp - amp_samples) * amp_samples) / (sigma**2) / i1
    out = (-2.0/len(amp)) * np.real(np.dot(pp, A))
    return out

def chisq_amp(amp, Z, F, sigma):
    ''' Compute and return chi-squared of amplitude between X and Z. '''
    amp_Z = compute_amp(Z, F)
    chisq = np.sum(np.abs((amp - amp_Z)/sigma)**2)/len(amp)
    return chisq 

###############################################################################
# Closure Phase Functions
###############################################################################

def compute_cphase(X, F_cphase):
    ''' Given an image X and the DFT matrices from three baselines,
        compute and return its closure phase. '''
    # Get fourier matrices of each baseline 
    A1 = F_cphase[:, :, 0]
    A2 = F_cphase[:, :, 1]
    A3 = F_cphase[:, :, 2]
    
    X = np.array(X)
    
    # Compute observed closure phase of image
    vis1 = np.matmul(X.reshape((1,-1)), np.transpose(A1)).astype(np.complex64)
    vis2 = np.matmul(X.reshape((1,-1)), np.transpose(A2)).astype(np.complex64)
    vis3 = np.matmul(X.reshape((1,-1)), np.transpose(A3)).astype(np.complex64)
    
    cphase = np.angle(vis1 * vis2 * vis3) 
    
    return cphase

def compute_cphase_grad(cphase, Z, F_cphase, sigma, npix):
    ''' 
        Compute gradient of closure phase chi-squared
        
        cphase : closure phase of true image 
        Z : predicted image vector
        F_cphase : 3 DFT matrices from three baselines in a closure triangle
    '''
    # Get fourier matrices of each baseline 
    A1 = F_cphase[:, :, 0]
    A2 = F_cphase[:, :, 1]
    A3 = F_cphase[:, :, 2]
    
    i1 = np.matmul(Z.reshape((1,-1)), np.transpose(A1)).astype(np.complex64)
    i2 = np.matmul(Z.reshape((1,-1)), np.transpose(A2)).astype(np.complex64)
    i3 = np.matmul(Z.reshape((1,-1)), np.transpose(A3)).astype(np.complex64)
    cphase_samples = np.angle(i1 * i2 * i3)
    
    pref = np.sin(cphase - cphase_samples)/(sigma**2)
    pt1  = pref/i1
    pt2  = pref/i2
    pt3  = pref/i3
    out  = -(2.0/len(cphase)) * np.imag(np.dot(pt1, A1) + np.dot(pt2, A2) + np.dot(pt3, A3))
    
    return out.reshape(npix**2)

def chisq_cphase(cphase, Z, F_cphase, sigma_cphase):
    """Closure Phase reduced chi-squared loss."""
    cphase_samples = compute_cphase(Z, F_cphase)
    chisq= (2.0/len(cphase)) * np.sum((1.0 - np.cos(cphase-cphase_samples))/(sigma_cphase**2))
    return chisq 
 
###############################################################################
# Closure Amplitude Functions
###############################################################################
   
def compute_camp(X, Amatrices):
    '''
        Compute closure amplitude of image vector X.
    '''
    i1 = np.dot(Amatrices[0], X)
    i2 = np.dot(Amatrices[1], X)
    i3 = np.dot(Amatrices[2], X)
    i4 = np.dot(Amatrices[3], X)
    
    camp = np.abs((i1 * i2)/(i3 * i4))
    return camp

def compute_camp_grad(camp, Z, Amatrices, sigma):
    """
    The gradient of the closure amplitude chi-squared
    
    camp: Closure amplitudes of true image
    Z: Predicted image vector
    Amatrices: DFT matrices of four baselines
    """
    i1 = np.dot(Amatrices[0], Z)
    i2 = np.dot(Amatrices[1], Z)
    i3 = np.dot(Amatrices[2], Z)
    i4 = np.dot(Amatrices[3], Z)
    camp_samples = np.abs((i1 * i2)/(i3 * i4))

    pp = ((camp - camp_samples) * camp_samples)/(sigma**2)
    pt1 = pp/i1
    pt2 = pp/i2
    pt3 = -pp/i3
    pt4 = -pp/i4
    out = (np.dot(pt1, Amatrices[0]) +
           np.dot(pt2, Amatrices[1]) +
           np.dot(pt3, Amatrices[2]) +
           np.dot(pt4, Amatrices[3]))

    return (-2.0/len(camp)) * np.real(out)
    
def chisq_camp(camp, Z, Amatrices, sigma):
    """Closure Amplitudes reduced chi-squared loss."""

    i1 = np.dot(Amatrices[0], Z)
    i2 = np.dot(Amatrices[1], Z)
    i3 = np.dot(Amatrices[2], Z)
    i4 = np.dot(Amatrices[3], Z)
    camp_samples = np.abs((i1 * i2)/(i3 * i4))

    chisq = np.sum(np.abs((camp - camp_samples)/sigma)**2)/len(camp)
    return chisq 

 
###############################################################################
# Log Closure Amplitude Functions
###############################################################################
   
def compute_lgcamp(X, Amatrices):
    ''' Compute log closure amplitude of image vector X '''
    a1 = np.abs(np.dot(Amatrices[0], X))
    a2 = np.abs(np.dot(Amatrices[1], X))
    a3 = np.abs(np.dot(Amatrices[2], X))
    a4 = np.abs(np.dot(Amatrices[3], X))
    
    lgcamp = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
    return lgcamp

def compute_lgcamp_grad(lgcamp, Z, Amatrices, sigma):
    """The gradient of the Log closure amplitude chi-squared"""

    i1 = np.dot(Amatrices[0], Z)
    i2 = np.dot(Amatrices[1], Z)
    i3 = np.dot(Amatrices[2], Z)
    i4 = np.dot(Amatrices[3], Z)
    lgcamp_samples = (np.log(np.abs(i1)) +
                         np.log(np.abs(i2)) - 
                         np.log(np.abs(i3)) -
                         np.log(np.abs(i4)))

    pp = (lgcamp - lgcamp_samples) / (sigma**2)
    pt1 = pp / i1
    pt2 = pp / i2
    pt3 = -pp / i3
    pt4 = -pp / i4
    out = (np.dot(pt1, Amatrices[0]) +
           np.dot(pt2, Amatrices[1]) +
           np.dot(pt3, Amatrices[2]) +
           np.dot(pt4, Amatrices[3]))

    return (-2.0/len(lgcamp)) * np.real(out)

def chisq_lgcamp(lgcamp, X, Amatrices, sigma):
    """Log Closure Amplitudes reduced chi-squared"""

    a1 = np.abs(np.dot(Amatrices[0], X))
    a2 = np.abs(np.dot(Amatrices[1], X))
    a3 = np.abs(np.dot(Amatrices[2], X))
    a4 = np.abs(np.dot(Amatrices[3], X))

    samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
    chisq = np.sum(np.abs((lgcamp - samples)/sigma)**2) / (len(lgcamp))
    return chisq 












