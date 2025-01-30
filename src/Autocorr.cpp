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
	auto start1 = std::chrono::high_resolution_clock::now();
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
	
	auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    std::cout << "Time taken for first section: " << elapsed1.count() << " seconds\n";
	
	
	start1 = std::chrono::high_resolution_clock::now();
	
	// Cast to double
	Multi_array<double, 3>    gs(n_threads, nb_fft, l_fft, fftw_malloc, fftw_free);
	// rfft result
	Multi_array<complex_d, 3> hs(n_threads, nb_fft, l_fft/2+1, fftw_malloc, fftw_free);
	// Allocate frency space accumulator
	Multi_array<complex_d, 3> gamma(n_threads, 2*N+1, l_fft, fftw_malloc, fftw_free);
	// Allocate output
	Multi_array<complex_d, 2> gamma_out( 2*N+1, l_fft, fftw_malloc, fftw_free);
	
	end1 = std::chrono::high_resolution_clock::now();
    elapsed1 = end1 - start1;
    std::cout << "Time taken for Allocation: " << elapsed1.count() << " seconds\n";
	
	start1 = std::chrono::high_resolution_clock::now();
	
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
	
	end1 = std::chrono::high_resolution_clock::now();
    elapsed1 = end1 - start1;
    std::cout << "Time taken for FFTW stuff: " << elapsed1.count() << " seconds\n";
	
	#pragma omp parallel
	{
		manage_thread_affinity();
		int this_thread = omp_get_thread_num();
		
		std::chrono::time_point<std::chrono::high_resolution_clock> s0,s1,s2;
		std::chrono::time_point<std::chrono::high_resolution_clock> end0  ,end1  ,end2   ;
		std::chrono::duration<double> el0(0.0)   ,el1(0.0)   ,el2(0.0)   ;
		
		#pragma omp single
		{
			start1 = std::chrono::high_resolution_clock::now();
		}
		
		#pragma omp for simd collapse(3)
		for (int k = 0; k < n_threads; k++) {
			for (int j = 0; j < 2*N+1; j++) {
				for (uint i = 0; i < l_fft; i++) {
					gamma(k,j,i) = 0 ;
				}
			}
		}
				
		#pragma omp single
		{
			end1 = std::chrono::high_resolution_clock::now();
			elapsed1 = end1 - start1;
			std::cout << "Time taken for zeroing gamma: " << elapsed1.count() << " seconds\n";
			
			start1 = std::chrono::high_resolution_clock::now();
		}
		
		#pragma omp for
		for (uint i_chunk = 0; i_chunk < Nchunk; i_chunk++) {
			// uint stride = nb_fft*l_fft*i_chunk ;
			s0 = std::chrono::high_resolution_clock::now();
			for (uint j = 0; j < nb_fft; j++) {
				for (uint i = 0; i < l_fft; i++) {
					gs(this_thread,j, i) = (double)data[nb_fft*l_fft*i_chunk + nb_fft*j + i];
				}
			}
			end0 = std::chrono::high_resolution_clock::now();
			el0 += end0 - s0 ;
			
			s1 = std::chrono::high_resolution_clock::now();
			fftw_execute_dft_r2c(r2c_plan, (double*)gs(this_thread,0),reinterpret_cast<fftw_complex *>(hs(this_thread,0)));
			end1 = std::chrono::high_resolution_clock::now();
			el1 += end1 - s1 ;
			
			s2 = std::chrono::high_resolution_clock::now();
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
			end2 = std::chrono::high_resolution_clock::now();
			el2 += end2 - s2 ;
		}
		
		#pragma omp single
		{
			end1 = std::chrono::high_resolution_clock::now();
			elapsed1 = end1 - start1;
			std::cout << "Time taken for main loop: " << elapsed1.count() << " seconds\n";
			std::cout << "Time casting int16 to double: " << el0.count() << " seconds\n";
			std::cout << "Time fft : " << el1.count() << " seconds\n";
			std::cout << "Time Ms : " << el2.count() << " seconds\n";
			
			start1 = std::chrono::high_resolution_clock::now();
		}
		
		#pragma omp for simd collapse(2)
		for (int j = 0; j < 2*N+1; j++) {
			for (uint i = 0; i < l_fft; i++) {
				for (int k = 0; k < n_threads; k++) {
					gamma_out(j,i) += gamma(k,j,i) ;
				}
			}
		}
		
		#pragma omp single
		{
			end1 = std::chrono::high_resolution_clock::now();
			elapsed1 = end1 - start1;
			std::cout << "Time taken for parallel reduction: " << elapsed1.count() << " seconds\n";
		}
	}
	
	start1 = std::chrono::high_resolution_clock::now();
	
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
	
	end1 = std::chrono::high_resolution_clock::now();
    elapsed1 = end1 - start1;
    std::cout << "Time taken for the rest: " << elapsed1.count() << " seconds\n";
	
	return gamma_out ; // Move semantics
}
