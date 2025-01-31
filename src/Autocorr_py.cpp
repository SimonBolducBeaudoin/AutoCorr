#include "Autocorr_py.h"

np_complex_d autocorr_cyclo_py(np_int16 &np_data, 
								uint64_t F, 
								uint64_t R, 
								uint nb_fft, 
								uint l_fft,
								int Mmax,
								std::string norm) 
{
	// Vérifier que la fréquence F est exactement sur la grille des fréquences représenté par
	// la transformé de Fourrier discrète .
	if (F*l_fft%R != 0) {
		throw std::runtime_error("F must be exactly reprented by the DFT's frequency spectrum.");
	}
	// Compute the index of the pump
	uint i_F = F*l_fft/R ;

	py::buffer_info data_buf = np_data.request();	
	if (data_buf.size < nb_fft*l_fft) {
        throw std::runtime_error("data length too small.");
    }
	// Cast data to multi array
	Multi_array<int16_t, 1> data = Multi_array<int16_t, 1>::numpy_share(np_data);
	
	Multi_array<complex_d, 2> gamma = autocorr_cyclo(data,i_F,nb_fft,l_fft,Mmax);
  
	// Normalization follows numpy.fft's convention
	double inv_norm;
	if (norm == "backward") { 
		inv_norm = 1.0 ;
	}
	else if (norm == "ortho" ){ 
		inv_norm = 1.0/std::sqrt(l_fft) ;
	}
	else if (norm == "forward"){ 
		inv_norm = 1.0/l_fft ;
	}
	else { 
		throw std::runtime_error("Invalid norm value ortddho; should be 'backward','ortho' or 'forward'.");
	}
	
	for (uint j = 0; j < gamma.get_n_j(); j++) {
		for (uint i = 0; i < gamma.get_n_i(); i++) {
			gamma(j,i) *= inv_norm ;
		}
	}
	
	// Casting a copy to a numpy array and pass to python 
	// without copying the bulk of the data the data
	return gamma.move_py();
}

np_complex_d autocorr_cyclo_m_py(np_int16 &np_data, 
								uint64_t F, 
								uint64_t R, 
								uint nb_fft, 
								uint l_fft,
								uint m,
								std::string norm) 
{
	// Vérifier que la fréquence F est exactement sur la grille des fréquences représenté par
	// la transformé de Fourrier discrète .
	if (F*l_fft%R != 0) {
		throw std::runtime_error("F must be exactly reprented by the DFT's frequency spectrum.");
	}
	// Compute the index of the pump
	uint i_F = F*l_fft/R ;

	py::buffer_info data_buf = np_data.request();	
	if (data_buf.size < nb_fft*l_fft) {
        throw std::runtime_error("data length too small.");
    }
	// Cast data to multi array
	Multi_array<int16_t, 1> data = Multi_array<int16_t, 1>::numpy_share(np_data);
	
	Multi_array<complex_d, 1> gamma = autocorr_cyclo_m(data,i_F,nb_fft,l_fft,m);
  
	// Normalization follows numpy.fft's convention
	double inv_norm;
	if (norm == "backward") { 
		inv_norm = 1.0 ;
	}
	else if (norm == "ortho" ){ 
		inv_norm = 1.0/std::sqrt(l_fft) ;
	}
	else if (norm == "forward"){ 
		inv_norm = 1.0/l_fft ;
	}
	else { 
		throw std::runtime_error("Invalid norm value ortddho; should be 'backward','ortho' or 'forward'.");
	}
	
	for (uint i = 0; i < gamma.get_n_i(); i++) {
		gamma(i) *= inv_norm ;
	}
	
	// Casting a copy to a numpy array and pass to python 
	// without copying the bulk of the data the data
	return gamma.move_py();
}

void init_autocorr(py::module &m) {
  m.def("autocorr_cyclo", &autocorr_cyclo_py,
		"data"_a.noconvert(), 
		"F"_a.noconvert(), 
		"R"_a.noconvert(), 
		"nb_fft"_a.noconvert(),
		"l_fft"_a.noconvert(),
		"Mmax"_a.noconvert()=-1,
		"norm"_a.noconvert()="backward"
		);
	m.def("autocorr_cyclo_m", &autocorr_cyclo_m_py,
		"data"_a.noconvert(), 
		"F"_a.noconvert(), 
		"R"_a.noconvert(), 
		"nb_fft"_a.noconvert(),
		"l_fft"_a.noconvert(),
		"m"_a.noconvert()=0,
		"norm"_a.noconvert()="backward"
		);
}
