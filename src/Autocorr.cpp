#include "Autocorr.h"

Multi_array<complex_d, 2> 
autocorr_cyclo (Multi_array<int16_t, 1> &data, 
				uint64_t i_F, 
				uint nb_fft, 
				uint l_fft,
				int Mmax)
{
	// Comput M and N
	int M ;
	int N ;
	uint i_f_max ;
	
	if (l_fft%2 == 0){
		M = (l_fft/2 -1)/i_F ;
		N = (2*(l_fft/2 -1))/i_F ;
		i_f_max = l_fft/2 - 1 ; // index of the last positive frequency of fftfreq
	}
	else{
		M = ((l_fft-1)/2)/i_F ;
		N = (l_fft-1)/i_F ;
		i_f_max = (l_fft-1)/2; 
	}
	if (Mmax <= 0){
		Mmax = N ;
	}
	Mmax = std::min(N,Mmax);
	M    = std::min(M,Mmax);
	N    = std::min(N,Mmax);
	
	// Cast to double
	Multi_array<double, 2> gs(nb_fft, l_fft, fftw_malloc, fftw_free);
	// rfft result
	Multi_array<complex_d, 2> hs( nb_fft, l_fft/2+1, fftw_malloc, fftw_free);
	// Allocate frency space accumulator
	Multi_array<complex_d, 2> gamma( 2*N+1, l_fft, fftw_malloc, fftw_free);
	for (int j = 0; j < 2*N+1; j++) {
		for (uint i = 0; i < l_fft; i++) {
			gamma(j,i) = 0 ;
		}
	}

	// FFTW plans
	fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
	int n[] = {(int)l_fft};
    fftw_plan r2c_plan = fftw_plan_many_dft_r2c(1,    // rank == 1D transform
                                    n,      //  list of dimensions
                                    nb_fft, // howmany (to do many ffts on the same core)
                                    gs(0), // input
                                    NULL,                                       // inembed
                                    1,                                          // istride
                                    l_fft,                                      // idist
                                    reinterpret_cast<fftw_complex *>(hs(0)),                      // output
                                    NULL,                                       //  onembed
                                    1,                                          // ostride
                                    l_fft / 2 + 1,		                        // odist
                                    FFTW_EXHAUSTIVE);
    fftw_export_wisdom_to_filename("FFTW_Wisdom.dat");
	
	uint Nchunk = data.get_n_i()/(nb_fft*l_fft);
	
	for (uint i_chunk = 0; i_chunk < Nchunk; i_chunk++) {
		uint stride = nb_fft*l_fft*i_chunk ;
		for (uint i = 0; i < nb_fft*l_fft; i++) {
			gs(0, i) = (double)data[stride + i];
		}
		fftw_execute_dft_r2c(r2c_plan, (double*)gs(0),reinterpret_cast<fftw_complex *>(hs(0)));
		
		for (uint j = 0; j < nb_fft; j++) {
			// m = 0 
			for (uint i = 0; i <= (l_fft-1)/2; i++) {
				gamma(0,i) += std::norm(hs(j,i));
			}
			// m >= M
			
			uint l_half ;
			int m = 1 ;
			for (; m <= M; m++) {
				l_half = (i_F*m)/2 + (i_F*m)%2  ;
				for (uint i = 0; i <= l_half; i++) {
					gamma(m,i) += hs(j,i)*hs(j,i_F*m-i);
				}
				for (uint i = i_F*m+1; i <= i_f_max; i++) {
					gamma(m,i) += hs(j,i)*std::conj(hs(j,i-i_F*m));
				}
			}
			for (; m <= N; m++) {
				l_half = (i_F*m)/2 + (i_F*m)%2  ;
				for (uint i = i_F*m-i_f_max; i <= l_half; i++) {
					gamma(m,i) += hs(j,i)*hs(j,i_F*m-i);
				}
			}
		}
    }
	
	// Symmetrize
	for (uint i = (l_fft-1)/2 + 1; i < l_fft; i++) {
		// m = 0 
		gamma(0,i) = gamma(0,l_fft-i);
	}
	int m = 1;
	uint l_half ;
	for (; m <= M; m++) {
		l_half = (i_F*m)/2 + (i_F*m)%2  ;
		for (uint i = l_half+1; i <= i_F*m; i++) {
			gamma(m,i) = gamma(m,i_F*m-i);
		}
		for (uint i = i_f_max+1+i_F*m; i < l_fft; i++) {
			gamma(m,i) = gamma(m,l_fft-1-i+i_F*m);
		}
	}
	
	for (; m <= N; m++) {
		l_half = (i_F*m)/2 + (i_F*m)%2  ;
		for (uint i = l_half+1; i <= i_f_max; i++) {
			gamma(m,i) = gamma(m,i_F*m-i);
		}
	}
	
	// gamma_m[i]=gamma_-m[-i]
	for (int j = N; j < 2*N+1; j++) {
		for (uint i = i_f_max+1; i < l_fft; i++) {
			gamma(j,i) = gamma(2*N+1-j,l_fft-i) ;
		}
	}
	
	// Reduce
	// We could transform it into direct space before returning 
	// or we could leave it in frequency representation
	
	fftw_destroy_plan(r2c_plan);
	
	// To get the average we need to divide by the number of fft made
	double inv_norm = 1.0/(Nchunk*nb_fft);
	for (int j = 0; j < 2*N+1; j++) {
		for (uint i = 0; i < l_fft; i++) {
			gamma(j,i) *= inv_norm ;
		}
	}
	
	return gamma ; // Move semantics
}
