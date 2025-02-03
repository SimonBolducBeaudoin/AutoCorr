#include "Autocorr.h"
#include <iostream>
#include <chrono>

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
	if (Mmax < 0){
		Mmax = N ;
	}
	Mmax = std::min(N,Mmax);
	M    = std::min(M,Mmax);
	N    = std::min(N,Mmax);
	
	int n_threads= omp_get_max_threads();
	
	// rfft result
	Multi_array<complex_d, 3> hs(n_threads, nb_fft, l_fft/2+1, fftw_malloc, fftw_free);
	// Allocate frency space accumulator
	Multi_array<complex_d, 3> gamma(n_threads, 2*N+1, l_fft, fftw_malloc, fftw_free);
	// Allocate output
	Multi_array<complex_d, 2> gamma_out( 2*N+1, l_fft, fftw_malloc, fftw_free);
	
	// FFTW plans
	fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
	int n[] = {(int)l_fft};
    fftw_plan r2c_plan = fftw_plan_many_dft_r2c(1,    // rank == 1D transform
                                    n,      //  list of dimensions
                                    nb_fft, // howmany (to do many ffts on the same core)
                                    (double*)hs(0), // input
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
	
	#pragma omp parallel
	{
		manage_thread_affinity();
		int this_thread = omp_get_thread_num();
		
		#pragma omp for simd collapse(3)
		for (int k = 0; k < n_threads; k++) {
			for (int j = 0; j < 2*N+1; j++) {
				for (uint i = 0; i < l_fft; i++) {
					gamma(k,j,i) = 0 ;
				}
			}
		}
		#pragma omp for simd collapse(2)
		for (int j = 0; j < 2*N+1; j++) {
			for (uint i = 0; i < l_fft; i++) {
				gamma_out(j,i) = 0 ;
			}
		}
		
		#pragma omp for
		for (uint i_chunk = 0; i_chunk < Nchunk; i_chunk++) {
			// uint stride = nb_fft*l_fft*i_chunk ;
			for (uint j = 0; j < nb_fft; j++) {
				for (uint i = 0; i < l_fft; i++) {
					((double*)(hs(this_thread,j)))[i] = (double)data[nb_fft*l_fft*i_chunk + l_fft*j + i];
				}
			}
			
			fftw_execute_dft_r2c(r2c_plan, (double*)hs(this_thread,0),reinterpret_cast<fftw_complex *>(hs(this_thread,0)));
			
			for (uint j = 0; j < nb_fft; j++) {
				// m = 0 
				for (uint i = 0; i <= (l_fft-1)/2; i++) {
					gamma(this_thread,0,i) += std::norm(hs(this_thread,j,i));
				}
				//// m >= M
				uint l_half ;
				int m = 1 ;
				for (; m <= M; m++) {
					l_half = (i_F*m)/2 + (i_F*m)%2  ;
					for (uint i = 0; i <= l_half; i++) {
						gamma(this_thread,m,i) += hs(this_thread,j,i)*hs(this_thread,j,i_F*m-i);
					}
					for (uint i = i_F*m+1; i <= i_f_max; i++) {
						gamma(this_thread,m,i) += hs(this_thread,j,i)*std::conj(hs(this_thread,j,i-i_F*m));
					}
				}
				for (; m <= N; m++) {
					l_half = (i_F*m)/2 + (i_F*m)%2  ;
					for (uint i = i_F*m-i_f_max; i <= l_half; i++) {
						gamma(this_thread,m,i) += hs(this_thread,j,i)*hs(this_thread,j,i_F*m-i);
					}
				}
			}
		}
		
		#pragma omp for simd collapse(2)
		for (int j = 0; j < 2*N+1; j++) {
			for (uint i = 0; i < l_fft; i++) {
				for (int k = 0; k < n_threads; k++) {
					gamma_out(j,i) += gamma(k,j,i) ;
				}
			}
		}
	}
		
	// Symmetrize
	for (uint i = (l_fft-1)/2 + 1; i < l_fft; i++) {
		// m = 0 
		gamma_out(0,i) = gamma_out(0,l_fft-i);
	}
	int m = 1;
	uint l_half ;
	for (; m <= M; m++) {
		l_half = (i_F*m)/2 + (i_F*m)%2  ;
		for (uint i = l_half+1; i <= i_F*m; i++) {
			gamma_out(m,i) = gamma_out(m,i_F*m-i);
		}
		for (uint i = i_f_max+1+i_F*m; i < l_fft; i++) {
			// BAD
			gamma_out(m,i) = gamma_out(m,l_fft-1-i+i_F*m);
		}
	}
	
	for (; m <= N; m++) {
		l_half = (i_F*m)/2 + (i_F*m)%2  ;
		for (uint i = l_half+1; i <= i_f_max; i++) {
			gamma_out(m,i) = gamma_out(m,i_F*m-i);
		}
	}
	
	// gamma_out_m[i]=gamma_out_-m[-i]
	for (int j = N; j < 2*N+1; j++) {
		for (uint i = i_f_max+1; i < l_fft; i++) {
			gamma_out(j,i) = gamma_out(2*N+1-j,l_fft-i) ;
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
			gamma_out(j,i) *= inv_norm ;
		}
	}
		
	return gamma_out ; // Move semantics
}


Multi_array<complex_d, 1> 
autocorr_cyclo_m (Multi_array<int16_t, 1> &data, 
				uint64_t i_F, 
				uint nb_fft, 
				uint l_fft,
				uint m)
{	
	// Comput M and N
	uint M ;
	uint N ;
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
	
	if (m > N) {
		throw std::runtime_error("Vallue error : m is bigger then N (the maximal index that fits in the bandwidth)");
	}
	
	// Check that m is valid
	
	int n_threads= omp_get_max_threads();
	
	// rfft result
	Multi_array<complex_d, 3> hs(n_threads, nb_fft, l_fft/2+1, fftw_malloc, fftw_free);
	// Allocate frency space accumulator
	Multi_array<complex_d, 2> gamma(n_threads, l_fft, fftw_malloc, fftw_free);
	// Allocate output
	Multi_array<complex_d, 1> gamma_out(l_fft, fftw_malloc, fftw_free);
	
	// FFTW plans
	fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
	int n[] = {(int)l_fft};
    fftw_plan r2c_plan = fftw_plan_many_dft_r2c(1,    // rank == 1D transform
                                    n,      //  list of dimensions
                                    nb_fft, // howmany (to do many ffts on the same core)
                                    (double*)hs(0), // input
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
	
	if (m==0){	
		#pragma omp parallel
		{
			manage_thread_affinity();
			int this_thread = omp_get_thread_num();
			
			#pragma omp for simd collapse(2) nowait
			for (int k = 0; k < n_threads; k++) {
				for (uint i = 0; i < l_fft; i++) {
					gamma(k,i) = 0 ;
				}
			}
			#pragma omp for simd		
			for (uint i = 0; i < l_fft; i++) {
				gamma_out(i) = 0 ;
			}
			
			#pragma omp for
			for (uint i_chunk = 0; i_chunk < Nchunk; i_chunk++) {
				for (uint j = 0; j < nb_fft; j++) {
					for (uint i = 0; i < l_fft; i++) {
						((double*)hs(this_thread,j))[i] = (double)data[nb_fft*l_fft*i_chunk + l_fft*j + i];
					}
				}
				
				fftw_execute_dft_r2c(r2c_plan, (double*)hs(this_thread,0),reinterpret_cast<fftw_complex *>(hs(this_thread,0)));
				
				for (uint j = 0; j < nb_fft; j++) {
					for (uint i = 0; i <= (l_fft-1)/2; i++) {
						gamma(this_thread,i) += std::norm(hs(this_thread,j,i));
					}
				}
			}
			#pragma omp for 
			for (uint i = 0; i < l_fft; i++) {
				for (int k = 0; k < n_threads; k++) {
					gamma_out(i) += gamma(k,i) ;
				}
			}
		}
		// Symmetrize
		for (uint i = (l_fft-1)/2 + 1; i < l_fft; i++) {
			gamma_out(i) = gamma_out(l_fft-i);
		}
	}
	else if (m<=M){	
		#pragma omp parallel
		{
			manage_thread_affinity();
			int this_thread = omp_get_thread_num();
			
			#pragma omp for simd collapse(2) nowait
			for (int k = 0; k < n_threads; k++) {
				for (uint i = 0; i < l_fft; i++) {
					gamma(k,i) = 0 ;
				}
			}
			#pragma omp for simd		
			for (uint i = 0; i < l_fft; i++) {
				gamma_out(i) = 0 ;
			}
			
			#pragma omp for
			for (uint i_chunk = 0; i_chunk < Nchunk; i_chunk++) {
				for (uint j = 0; j < nb_fft; j++) {
					for (uint i = 0; i < l_fft; i++) {
						((double*)hs(this_thread,j))[i] = (double)data[nb_fft*l_fft*i_chunk + l_fft*j + i];
					}
				}
				
				fftw_execute_dft_r2c(r2c_plan, (double*)hs(this_thread,0),reinterpret_cast<fftw_complex *>(hs(this_thread,0)));
				
				for (uint j = 0; j < nb_fft; j++) {
					uint l_half = (i_F*m)/2 + (i_F*m)%2  ;
					for (uint i = 0; i <= l_half; i++) {
						gamma(this_thread,i) += hs(this_thread,j,i)*hs(this_thread,j,i_F*m-i);
					}
					for (uint i = i_F*m+1; i <= i_f_max; i++) {
						gamma(this_thread,i) += hs(this_thread,j,i)*std::conj(hs(this_thread,j,i-i_F*m));
					}
				}
			}
			#pragma omp for 
			for (uint i = 0; i < l_fft; i++) {
				for (int k = 0; k < n_threads; k++) {
					gamma_out(i) += gamma(k,i) ;
				}
			}
		}
		// Symmetrize
		uint l_half = (i_F*m)/2 + (i_F*m)%2  ;
		for (uint i = l_half+1; i <= i_F*m; i++) {
			gamma_out(i) = gamma_out(i_F*m-i);
		}
		for (uint i = i_f_max+1+i_F*m; i < l_fft; i++) {
			gamma_out(i) = gamma_out(l_fft-1-i+i_F*m);
		}
	}
	else {	//  M < m < N 
		#pragma omp parallel
		{
			manage_thread_affinity();
			int this_thread = omp_get_thread_num();
			
			#pragma omp for simd collapse(2) nowait
			for (int k = 0; k < n_threads; k++) {
				for (uint i = 0; i < l_fft; i++) {
					gamma(k,i) = 0 ;
				}
			}
			#pragma omp for simd		
			for (uint i = 0; i < l_fft; i++) {
				gamma_out(i) = 0 ;
			}
			
			#pragma omp for
			for (uint i_chunk = 0; i_chunk < Nchunk; i_chunk++) {
				for (uint j = 0; j < nb_fft; j++) {
					for (uint i = 0; i < l_fft; i++) {
						((double*)hs(this_thread,j))[i] = (double)data[nb_fft*l_fft*i_chunk + l_fft*j + i];
					}
				}
				
				fftw_execute_dft_r2c(r2c_plan, (double*)hs(this_thread,0),reinterpret_cast<fftw_complex *>(hs(this_thread,0)));
				
				for (uint j = 0; j < nb_fft; j++) {
					uint l_half = (i_F*m)/2 + (i_F*m)%2  ;
					for (uint i = i_F*m-i_f_max; i <= l_half; i++) {
						gamma(this_thread,i) += hs(this_thread,j,i)*hs(this_thread,j,i_F*m-i);
					}
				}
			}
			#pragma omp for 
			for (uint i = 0; i < l_fft; i++) {
				for (int k = 0; k < n_threads; k++) {
					gamma_out(i) += gamma(k,i) ;
				}
			}
		}
		// Symmetrize
		uint l_half = (i_F*m)/2 + (i_F*m)%2  ;
		for (uint i = l_half+1; i <= i_f_max; i++) {
			gamma_out(i) = gamma_out(i_F*m-i);
		}
	}
		
	// Reduce
	// We could transform it into direct space before returning 
	// or we could leave it in frequency representation
	
	fftw_destroy_plan(r2c_plan);
	
	// To get the average we need to divide by the number of fft made
	double inv_norm = 1.0/(Nchunk*nb_fft);
	for (uint i = 0; i < l_fft; i++) {
		gamma_out(i) *= inv_norm ;
	}
		
	return gamma_out ; // Move semantics
}
