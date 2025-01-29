#pragma once

#include "Autocorr.h"

#include <complex>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string> 

namespace py = pybind11;
using namespace pybind11::literals;

typedef std::complex<double> complex_d;

typedef py::array_t<complex_d, py::array::c_style> np_complex_d;
typedef py::array_t<double, py::array::c_style> np_double;
typedef py::array_t<int16_t, py::array::c_style> np_int16;

np_complex_d autocorr_cyclo_py(np_int16 &np_data, 
								uint64_t F, 
								uint64_t R, 
								uint nb_fft, 
								uint l_fft,
								int Mmax=-1,
								std::string norm="backward"
								); 
void init_autocorr(py::module &m);
