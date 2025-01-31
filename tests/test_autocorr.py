#!/bin/env/python
#! -*- coding: utf-8 -*-

import numpy as np
from SBB.AutoCorr.autocorr import autocorr_cyclo , autocorr_cyclo_m
from matplotlib.pyplot import *
import time

from SBB.Omp_extra.omp_extra import set_num_threads

from python_translation import *

from fractions import Fraction

#set_num_threads(1)

# Parameters
R = 32
F = 6

Period = Fraction(F,R).denominator

nb_fft = 1
l_fft = 1024
delta = 5 # Offset to avoid zero variance
timesteps = 2**30# Number of time steps

dt = 1/R  # Time step size

# Generate the data vector
time = np.r_[0:Period*2]*dt
Nchunk = timesteps//(Period*2)
time = np.tile(time, Nchunk)
np.random.seed(42)
# data = np.array([
    # int(np.random.normal(0, np.sqrt(abs(1000*np.sin(F*2*np.pi * t)) + delta))) for t in time
# ], dtype=np.int16)

if len(time) > 2**20:
    base_length = 2**20
    base_data = np.array([
        int(np.random.normal(0, np.sqrt(abs(1000*np.sin(F*2*np.pi * t)) + delta))) 
        for t in np.linspace(0, time[-1], base_length)
    ], dtype=np.int16)

    repeats = len(time) // base_length + 1  # Ensure enough repetitions
    data = np.tile(base_data, repeats)[:len(time)]  # Repeat and truncate
else:
    data = np.array([
        int(np.random.normal(0, np.sqrt(abs(1000*np.sin(F*2*np.pi * t)) + delta))) 
        for t in time
    ], dtype=np.int16)


acorr_m_1 = autocorr_cyclo(data,F,R,nb_fft  ,l_fft  ,Mmax=-1,norm="forward")
acorr_m_2 = autocorr_cyclo(data,F,R,nb_fft*2,l_fft  ,Mmax=-1,norm="forward")
acorr_m_3 = autocorr_cyclo(data,F,R,nb_fft  ,l_fft//2,Mmax=-1,norm="forward")
acorr_m_4 = autocorr_cyclo_py_Vpy(data,F,R,l_fft    ,Mmax=-1,norm="forward")

# Visualizing the result
fig,axs = subplots(2,1)
time_slice= slice(None,1000)
for i in range(min(Nchunk,1000)):
    axs[0].plot(data[i*Period:(i+1)*Period],marker='none',ls='-')

ms = np.r_[np.r_[:len(acorr_m_1)//2+1] , np.r_[-len(acorr_m_1)//2+1:0]]
for m,acorr in enumerate(acorr_m_1) :
    axs[1].plot(np.fft.fftfreq(l_fft,dt),abs(acorr),label="{}".format(ms[m]))

fig.legend() 

# Different nb_fft
fig,axs = subplots(1,1)
time_slice= slice(None,1000)

ms = np.r_[np.r_[:len(acorr_m_1)//2+1] , np.r_[-len(acorr_m_1)//2+1:0]]
for m,(acorr1,acorr2) in enumerate(zip(acorr_m_1,acorr_m_2)) :
    axs.plot(np.fft.fftfreq(l_fft,dt),abs(acorr1-acorr2),label="{}".format(ms[m]))

fig.legend() 

fig,axs = subplots(2,1)
time_slice= slice(None,1000)

ms = np.r_[np.r_[:len(acorr_m_1)//2+1] , np.r_[-len(acorr_m_1)//2+1:0]]
for m,(acorr1,acorr2) in enumerate(zip(acorr_m_1,acorr_m_2)) :
    axs[0].plot(np.fft.fftfreq(l_fft,dt),abs(acorr1),label="{}".format(ms[m]))
    axs[1].plot(np.fft.fftfreq(l_fft,dt),abs(acorr2),label="{}".format(ms[m]))

fig.legend() 


# Different l_fft
fig,axs = subplots(2,1)
time_slice= slice(None,1000)

ms = np.r_[np.r_[:len(acorr_m_1)//2+1] , np.r_[-len(acorr_m_1)//2+1:0]]
for m,(acorr1,acorr3) in enumerate(zip(acorr_m_1,acorr_m_3)) :
    axs[0].plot(np.fft.fftfreq(l_fft,dt),abs(acorr1),label="{}".format(ms[m]))
    axs[1].plot(np.fft.fftfreq(l_fft//2,dt),abs(acorr3)                         )

fig.legend() 

# Implementation python vs C++
fig,axs = subplots(2,1)
time_slice= slice(None,1000)

ms = np.r_[np.r_[:len(acorr_m_1)//2+1] , np.r_[-len(acorr_m_1)//2+1:0]]
for m,(acorr1,acorr4) in enumerate(zip(acorr_m_1,acorr_m_4)) :
    if m > 1 :
        break
    axs[0].plot(np.fft.fftfreq(l_fft,dt),abs(acorr1),label="{}".format(ms[m]))
    axs[1].plot(np.fft.fftfreq(l_fft,dt),abs(acorr4)                         )

fig.legend() 

# Implementation python vs C++ différence pour m=0 et m=1
fig,axs = subplots(2,1)
time_slice= slice(None,1000)

ms = np.r_[np.r_[:len(acorr_m_1)//2+1] , np.r_[-len(acorr_m_1)//2+1:0]]
for m,(acorr1,acorr4) in enumerate(zip(acorr_m_1,acorr_m_4)) :
    if m > 1 :
        break
    axs[0].plot(np.fft.fftfreq(l_fft,dt),abs(acorr1),label="{}".format(ms[m]))
    axs[1].plot(np.fft.fftfreq(l_fft,dt),abs(acorr1-acorr4)                         )
    

fig.legend() 