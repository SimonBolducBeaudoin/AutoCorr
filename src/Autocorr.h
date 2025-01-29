#pragma once


#include <Multi_array.h>
#include <fftw3.h>
#include <omp_extra.h>
#include <complex>
#include <cstdint> // unint_t
typedef std::complex<double> complex_d;
typedef std::complex<float> complex_f;
typedef unsigned int uint;

Multi_array<complex_d, 2> 
autocorr_cyclo (Multi_array<int16_t, 1> &data, 
				uint64_t i_F, 
				uint nb_fft, 
				uint l_fft,
				int Mmax=-1);
