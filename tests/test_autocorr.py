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
l_fft = 2048
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

# Visualizing the result ##########################################################
acorr_m_1 = autocorr_cyclo(data,F,R,nb_fft  ,l_fft  ,Mmax=-1)

fig,axs = subplots(2,1)
time_slice= slice(None,1000)
for i in range(min(Nchunk,1000)):
    axs[0].plot(data[i*Period:(i+1)*Period],marker='none',ls='-')


ms = np.r_[np.r_[:len(acorr_m_1)//2+1] , np.r_[-len(acorr_m_1)//2+1:0]]
for m,acorr in enumerate(acorr_m_1) :
    axs[1].plot(np.fft.fftfreq(l_fft,dt),abs(acorr),label="{}".format(ms[m]))

axs[0].set_title("Input data")
axs[0].set_xlabel("t")
axs[1].set_title(r"C++ $\gamma_m(f)$")
axs[1].set_xlabel("f")
fig.suptitle("Visualizing the result")
fig.legend(title="m=") 

# Implementation python vs C++ ####################################################
acorr_m_1 = autocorr_cyclo(data,F,R,nb_fft  ,l_fft  ,Mmax=-1,norm="forward")
acorr_m_4 = autocorr_cyclo_py_Vpy(data,F,R,l_fft    ,Mmax=-1,norm="forward")

fig,axs = subplots(3,1)
time_slice= slice(None,1000)

ms = np.r_[np.r_[:len(acorr_m_1)//2+1] , np.r_[-len(acorr_m_1)//2+1:0]]
for m,(acorr1,acorr4) in enumerate(zip(acorr_m_1,acorr_m_4)) :
    if m > 1 :
        break
    axs[0].plot(np.fft.fftfreq(l_fft,dt),abs(acorr1),label="{}".format(ms[m]))
    axs[1].plot(np.fft.fftfreq(l_fft,dt),abs(acorr4)                         )
    axs[2].plot(np.fft.fftfreq(l_fft,dt),abs(acorr4-acorr1)                  )
    
axs[0].set_title(r"C++ $\gamma_m(f)$")
axs[0].set_xlabel("f")
axs[1].set_title(r"python$")
axs[1].set_xlabel("f")
axs[2].set_title(r" diff$")
axs[2].set_xlabel("f")
fig.suptitle("Implementation python vs C++")
fig.legend(title="m=") 

# Implementation python vs C++ nb_fft=2 ####################################################
acorr_m_1 = autocorr_cyclo(data,F,R,2  ,l_fft  ,Mmax=-1,norm="forward")
acorr_m_4 = autocorr_cyclo_py_Vpy(data,F,R,l_fft    ,Mmax=-1,norm="forward")

fig,axs = subplots(3,1)
time_slice= slice(None,1000)

ms = np.r_[np.r_[:len(acorr_m_1)//2+1] , np.r_[-len(acorr_m_1)//2+1:0]]
for m,(acorr1,acorr4) in enumerate(zip(acorr_m_1,acorr_m_4)) :
    if m > 1 :
        break
    axs[0].plot(np.fft.fftfreq(l_fft,dt),abs(acorr1),label="{}".format(ms[m]))
    axs[1].plot(np.fft.fftfreq(l_fft,dt),abs(acorr4)                         )
    axs[2].plot(np.fft.fftfreq(l_fft,dt),abs(acorr4-acorr1)                  )
    
axs[0].set_title(r"C++ $\gamma_m(f)$")
axs[0].set_xlabel("f")
axs[1].set_title(r"python")
axs[1].set_xlabel("f")
axs[2].set_title(r" diff")
axs[2].set_xlabel("f")
fig.suptitle("Implementation python vs C++ nb_fft=2")
fig.legend(title="m=") 

# nb_fft #########################################################################
nb_fft_1 = 1
acorr_m_1 = autocorr_cyclo(data, F, R, nb_fft_1, l_fft)
nb_fft_2 = 2
acorr_m_2 = autocorr_cyclo(data, F, R, nb_fft_2, l_fft)

fig,axs = subplots(3,1)
time_slice= slice(None,1000)

ms = np.r_[np.r_[:len(acorr_m_1)//2+1] , np.r_[-len(acorr_m_1)//2+1:0]]
for m,(acorr1,acorr2) in enumerate(zip(acorr_m_1,acorr_m_2)) :
    if m > 1 :
        break
    axs[0].plot(np.fft.fftfreq(l_fft,dt),abs(acorr1),label="{}".format(ms[m]))
    axs[1].plot(np.fft.fftfreq(l_fft,dt),abs(acorr2)                         )
    axs[2].plot(np.fft.fftfreq(l_fft,dt),abs(acorr2-acorr1)                  )
    
axs[0].set_title(r"C++ Testing nb_fft (shouldn't change anything)")
axs[0].set_xlabel("f")
axs[1].set_title(r"python")
axs[1].set_xlabel("f")
fig.suptitle("Implementation python vs C++")
fig.legend(title="m=") 

# C++ autocorr_cyclo_m vs python implementation ####################################################
acorr_py = autocorr_cyclo_py_Vpy(data,F,R,l_fft    ,Mmax=-1,norm="forward")

fig,axs = subplots(3,1)
time_slice= slice(None,1000)

ms = np.r_[np.r_[:len(acorr_m_1)//2+1] , np.r_[-len(acorr_m_1)//2+1:0]]
for m,acorr1 in enumerate(acorr_py) :
    if m > 1 :
        break
    acorr_m  = autocorr_cyclo_m(data,F,R,nb_fft  ,l_fft  ,m=m,norm="forward")
    axs[0].plot(np.fft.fftfreq(l_fft,dt),abs(acorr1),label="{}".format(ms[m]))
    axs[1].plot(np.fft.fftfreq(l_fft,dt),abs(acorr_m)                         )
    axs[2].plot(np.fft.fftfreq(l_fft,dt),abs(acorr_m-acorr1)                  )
    
axs[0].set_title(r"python")
axs[0].set_xlabel("f")
axs[1].set_title(r"C++ autocorr_cyclo_m")
axs[1].set_xlabel("f")
axs[2].set_title(r" diff")
axs[2].set_xlabel("f")
fig.suptitle("C++ autocorr_cyclo_m vs python implementation")
fig.legend(title="m=") 

# C++ autocorr_cyclo_m nb_fft = 2 vs python implementation ####################################################
acorr_py = autocorr_cyclo_py_Vpy(data,F,R,l_fft    ,Mmax=-1,norm="forward")

fig,axs = subplots(3,1)
time_slice= slice(None,1000)

ms = np.r_[np.r_[:len(acorr_m_1)//2+1] , np.r_[-len(acorr_m_1)//2+1:0]]
for m,acorr1 in enumerate(acorr_py) :
    if m > 1 :
        break
    acorr_m  = autocorr_cyclo_m(data,F,R,2  ,l_fft  ,m=m,norm="forward")
    axs[0].plot(np.fft.fftfreq(l_fft,dt),abs(acorr1),label="{}".format(ms[m]))
    axs[1].plot(np.fft.fftfreq(l_fft,dt),abs(acorr_m)                         )
    axs[2].plot(np.fft.fftfreq(l_fft,dt),abs(acorr_m-acorr1)                  )
    
axs[0].set_title(r"python")
axs[0].set_xlabel("f")
axs[1].set_title(r"C++ autocorr_cyclo_m")
axs[1].set_xlabel("f")
axs[2].set_title(r" diff")
axs[2].set_xlabel("f")
fig.suptitle("C++ autocorr_cyclo_m nb_fft = 2 vs python implementation")
fig.legend(title="m=") 

# autocorr_cyclo vs  autocorr_cyclo_m #########################################################################

acorr = autocorr_cyclo(data, F, R, nb_fft, l_fft)
    
fig,axs = subplots(3,1)
time_slice= slice(None,1000)

ms = np.r_[np.r_[:len(acorr_m_1)//2+1] , np.r_[-len(acorr_m_1)//2+1:0]]

for m in range(len(acorr)//2):
    acorr1 = acorr[m]
    acorr_m = autocorr_cyclo_m(data, F, R, nb_fft, l_fft, m=m)
    acorr2 = acorr_m
    
    axs[0].plot(np.fft.fftfreq(l_fft,dt),abs(acorr1),label="{}".format(ms[m]))
    axs[1].plot(np.fft.fftfreq(l_fft,dt),abs(acorr2)                         )
    axs[2].plot(np.fft.fftfreq(l_fft,dt),abs(acorr2-acorr1)                  )
    
axs[0].set_title(r"autocorr_cyclo")
axs[0].set_xlabel("f")
axs[1].set_title(r"autocorr_cyclo_m")
axs[1].set_xlabel("f")
fig.suptitle("autocorr_cyclo vs autocorr_cyclo_m")
fig.legend(title="m=") 