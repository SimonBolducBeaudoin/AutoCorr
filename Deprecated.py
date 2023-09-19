#!/bin/env/python
#! -*- coding: utf-8 -*-

import numpy
#from SBB.Utilities.General_tools import *

from scipy import constants as const
_e      = const.e           # Coulomb
_h      = const.h           # Joule second
_kb     = const.Boltzmann   # J/K (joules per kelvin)

def window_after_2ns(S2):
    """
        Damping everything more than 2 ns
        At a sampling rate of 0.03125 it means everything after the 64th point
        
        This will need to be rewritten if used with another aquisition card...
    """ 
    def damp(x,epsilon,x_0):
        return numpy.exp((-1)*epsilon*(x-x_0))
    def compute_epsilon(red,after_lenght):
        return -numpy.log(1.0/red)/(after_lenght)
    red = 1000
    L_0 = 65
    epsilon = compute_epsilon(red,after_lenght=L_0)
    shape = S2.shape
    len = shape[-1]
    out = numpy.zeros(shape)
    out = S2
    for index in range(L_0,len):
        out[...,index] = S2[...,index]*damp(index,epsilon,L_0-1)
    return out

class Tunnel_junction(object):
    """
       This class embeds the logic associated with the tunnel junctionhttps://www.youtube.com/feed/subscriptions
       It contains usefull static methods
       I'm using JO's phd thesis as a reference for the SII definitions
        Todos : 
            - Add imperfections in the junciton parasitic capacitance ect.
            - Add photoexcitation behavior
        Bugs :
    """
    __version__     = { 'Tunnel_junction'  : 0.5 }
    def __init__(self,R_jct):
        self.R_jct = R_jct
    @staticmethod
    def get_I_jct_max(**options):
        I_jct_max = 20.0e-6 # [A]
        return I_jct_max
    @staticmethod
    def coth(x):
        """ Not defined in numpy """
        return 1.0/numpy.tanh(x)
    @staticmethod
    def cothc(x):
        """ x*coth(x) """
        ret             = x/numpy.tanh(x)
        ret[numpy.isnan(ret)] = 1.
        return ret
    @staticmethod
    def Vsquare_to_K(SII,Z_jct):
        """ 
        Converts from V**2/Hz to Kelvin
        Si on replie les fréquences negatives sur les fréquences positives 
        on prend le facteur 1/2k (i.e. l'abscisse n'existe pas sur les fréquence negative)
        Sinon on prend 1/4K
        """
        return SII/(2.0*Z_jct*_kb)
    @staticmethod
    def K_to_Vsquare(SII,Z_jct):
        """ 
        Converts from Kelvin to V**2/Hz
        """
        return (2.0*Z_jct*_kb)*SII
    @staticmethod
    def _to_K(SII,Te):
        """
        Converts a SII with no unit to Kelvin
        """
        return Te*SII
    @staticmethod
    def _to_Vsquare(SII,Z_jct,Te):
        """
        Converts a SII with no unit to V**2/Hz
        """
        return (2.0*Z_jct*_kb*Te)*SII
    @staticmethod
    def _to_Asquare(SII,Z_jct,Te):
        """
        Converts a SII with no unit to A**2/Hz
        """
        return (2.0*_kb*Te)*SII/Z_jct
    @staticmethod
    def V_th(f,Te=None,epsilon=0.01):
        """
            Return the V_th for which e*V_th =  h f
            
            f : float or 1D array
            
            When Te != None
                Returns a more conservative value for V_th
                for which SII_eq = asymtotic value + epsilon
        """
        if Te :
            cst = numpy.arctanh(1.0/(1.0+epsilon)) # coth(cst) = 1.01
            return numpy.max(numpy.array([  numpy.abs( cst*2.0*_kb*Te - _h*f ) ,  numpy.abs( cst*2.0*_kb*Te + _h*f ) ]),axis=0)/_e
        else :
            return _h*f/_e
    @staticmethod
    def SII_eq(E,Te):
        """
        Compute SII_eq [no unit]
        
        Parameters
        ----------
        E : numpy array 
            [J]
        Te : numpy array 
            [K]
        """
        return Tunnel_junction.cothc(E/(2.0*_kb*Te))
    @staticmethod
    def SII_eq_Vsquare(E,R,Te):
        """
        Computes SII_eq in [V**2/Hz]
        """
        Seq     =  Tunnel_junction.SII_eq(E,Te)
        return Tunnel_junction._to_Vsquare(Seq,R,Te)
    @staticmethod
    def SII_dc(I,f,Te,R):
        """
        Compute SII_dc [no units]
        Parameters
        ----------
        I : numpy array
            Current through the junction [A]
        f : numpy array
            Frequency [Hz]
        Te : float 
            Electronic temperature [K]
        R : float
            R_jct [Ohm]
        Notes
        -----
        We assume that the junction is polarized in current and therefore
        R_s is needed to convert to the reel junction tension.
        """
        eV,hf   = numpy.meshgrid(_e*I*R,_h*f,indexing='ij')
        SII_eq  = Tunnel_junction.SII_eq
        return (SII_eq(eV-hf,Te)+SII_eq(eV+hf,Te))/2.0
    @staticmethod
    def SII_dc_V2(I,f,Te,R):
        """
        Compute SII_dc in [V**2/Hz]
        See Also
        --------
        SII_dc : Parameters
        """
        sii     = Tunnel_junction.SII_dc(I,f,Te,R)
        return Tunnel_junction._to_Vsquare(sii,R,Te)
    @staticmethod
    def SII_dc_A2(I,f,Te,R):
        """
        Compute SII_dc in [V**2/Hz]
        See Also
        --------
        SII_dc : Parameters
        """
        sii     = Tunnel_junction.SII_dc(I,f,Te,R)
        return Tunnel_junction._to_Asquare(sii,R,Te)
    @staticmethod
    def SII_dc_V2_amplified(I,f,Te,R,G,b):
        """
        Model for SII [V**2/Hz] with dc polarisation only
        A constant parameter "b" (amplification noise) is added and a gain parameter "G"
        See Also
        --------
        SII_dc_V2 : Parameters
        """
        return G*(Tunnel_junction.SII_dc_V2(I,f,Te,R)) + b
    @staticmethod
    def SII_dc_A2_amplified(I,f,Te,R,G,b):
        """
        Model for SII [V**2/Hz] with dc polarisation only
        A constant parameter "b" (amplification noise) is added and a gain parameter "G"
        See Also
        --------
        SII_dc_A2 : Parameters
        """
        return G*(Tunnel_junction.SII_dc_A2(I,f,Te,R)) + b
    @staticmethod
    def _decorator_fix_f_R(f,R,func,amplified=True):
        """
        Fixes f and R 
        
        Parameters
        ----------
        f,R  : float
        func : method
            the full fitting model
        
        See Also
        --------
        SII_dc_A2_amplified
        
        Returns
        -------
        out : a method
            Parameters
            ----------
            I : numpy array 
                [A]
            tuple :
                The fitting parameters
        """
        def out(I,tup): #tup=(Te, G, b)
            Te = tup[0]
            G=tup[1]
            b=tup[2]
            return func(I,f,Te,R,G,b)[:,0] if amplified else func(I,f,Te,R)[:,0]
        return out
    @staticmethod
    def _decorator_fix_I(I,func):
        """
            Compatible with SII_dc_A2
        """
        def out(f,tup): #tup=(Te,R)
            Te=tup[0]
            R=tup[1]
            return func(I,f,Te,R)[0,:]
        return out
    @staticmethod
    def _decorator_fix_I_R(I,R,func):
        """
            Compatible with SII_dc_A2
        """
        def out(f,tup): #tup=Te
            Te=tup[0]
            return func(I,f,Te,R)[0,:]
        return out
