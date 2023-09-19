#!/bin/env/python
#! -*- coding: utf-8 -*-

import numpy
from SBB.Numpy_extra.numpy_extra import symetrize,find_nearest_A_to_a
from SBB.Math_extra.Math_extra import fourier_transform
from SBB.Phys import Tunnel_Junction 

from SBB.AutoCorr.Deprecated import window_after_2ns

def binV2_to_A2(S2,R_acq,mv_per_bin):
    """
    Used to convert SII(t)[bin_V**2] to SII(t)[A**2]
    """
    return S2*(mv_per_bin*1.0e-3)**2/(R_acq**2)
    
def SII_dc_of_t_to_spectrum(S2,dt):
    S2_windowed       = window_after_2ns(S2)
    S2_sym            = symetrize (S2_windowed)
    return numpy.abs(fourier_transform(S2_sym,dt))
    
def compute_Ith(f,f_max,Te,eps=0.01,R_jct=50.0):
    """
    Computes the threshold currents above which SII = eI in good approximation for each frequencies.

    This function intended to help with the use of compute_G_and_SII_amp

    Parameters
    ----------
    f[Hz]       : float or array
    f_max[Hz]   : float
        frequencies above this value will get the same threshold as this frequency
    Te [K]      : float
        expected temperature of the electron of the junction
    eps [~]     : float
        Error on the convergence of Seq
    R_jct [Ohm] : float
    """
    V_th = Noise_Theory_Junction.V_th(f,Te=Te,epsilon=eps)
    _,f_max_idx = find_nearest_A_to_a(f_max,f)
    f_max_idx = f_max_idx[0]
    V_th[f_max_idx:] = V_th[f_max_idx]
    return V_th/R_jct

def compute_noiseTemp(SII,G,R=50.0):
    """
    Used to compute the amplifier's noise temperature
    Converts SII[A**2/Hz] in K

    Last update
    -----------
    Created
    I choose the factor 2.0 instead of 4.0 in Jonhson-Nyquist Noise
    """
    return SII*R/(2.0*C.k*G)

def compute_SII_sym_and_antisym(SII,axis=-1,interlacing=False):
    """
        Suposing SII's abscisse is negative on the first half of axis and positive on the second half
            Ex: SII(Vdc) where Vdc = [-5,-4,-3,-2,-1,0,0,1,2,3,4,5]
    """
    SII         = SII.swapaxes(axis,-1)
    
    L       = SII.shape[-1]
    if L%2 or ((L%4)and interlacing):
        raise Exception("SII.shape[axis] is not valid for symertization")
    l = L//2
    shape       = SII.shape 
    shape_sym   = shape[:-1] + (l,)
    slice_pos = slice(L//2,None)
    slice_neg = slice(None,L//2)
    if interlacing :
        S2_sym  = (SII[...,slice_pos] + SII[...,slice_neg].reshape( shape[:-1]+(l//2,2))[...,::-1,:].reshape(shape_sym) )/2.0 
        S2_anti = (SII[...,slice_pos] - SII[...,slice_neg].reshape( shape[:-1]+(l//2,2))[...,::-1,:].reshape(shape_sym) )/2.0
    else :
        S2_sym  = (SII[...,slice_pos] + SII[...,slice_neg][...,::-1] )/2.0 
        S2_anti = (SII[...,slice_pos] - SII[...,slice_neg][...,::-1] )/2.0
    
    return S2_sym.swapaxes(axis,-1),S2_anti.swapaxes(axis,-1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    