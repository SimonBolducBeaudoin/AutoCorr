import unittest
import numpy as np
from SBB.AutoCorr.autocorr import autocorr_cyclo, autocorr_cyclo_m
from python_translation import *
from fractions import Fraction
from SBB.Omp_extra.omp_extra import set_num_threads


class TestAutocorrCyclo(unittest.TestCase):
    
    def setUp(self):
        # Parameters for the test
        self.R = 32
        self.F = 6
        self.Period = Fraction(self.F, self.R).denominator
        self.timesteps = 2**20  # Number of time steps
        self.dt = 1/self.R  # Time step size
        
        self.time = np.r_[0:self.Period*2]*self.dt
        self.Nchunk = self.timesteps // (self.Period*2)
        self.time = np.tile(self.time, self.Nchunk)
        
        # Generate the random data
        np.random.seed(42)
        self.data = np.array([
            int(np.random.normal(0, np.sqrt(abs(1000*np.sin(self.F*2*np.pi * t)) + 5))) for t in self.time
        ], dtype=np.int16)
        
        # Fixed parameters for the autocorr function
        self.l_fft = 1024
        self.nb_fft = 8 
        
        
        self.i_F = self.F*self.l_fft//self.R ;
        
        if self.l_fft%2==0 :
            self.M = (self.l_fft//2 -1)//self.i_F 
            self.N = (2*(self.l_fft//2 -1))//self.i_F 
        else :
            self.M = ((self.l_fft-1)//2)//self.i_F 
            self.N = (self.l_fft-1)//self.i_F 
    
    def test_A_autocorr_cyclo_vs_python(self):
        acorr_1 = autocorr_cyclo(self.data, self.F, self.R, 1, self.l_fft)
        acorr_2 = autocorr_cyclo_py_Vpy(self.data, self.F, self.R, self.l_fft)
        
        # testing only m={0,1}
        np.testing.assert_allclose(abs(acorr_1[[0,1]]), abs(acorr_2[[0,1]]),rtol=1e-12, err_msg="C++ didn't give the same result as python algorithm.")
        
    def test_B_nb_fft_2_vs_python(self):
        acorr_1 = autocorr_cyclo(self.data, self.F, self.R, 2, self.l_fft)
        acorr_2 = autocorr_cyclo_py_Vpy(self.data, self.F, self.R, self.l_fft)
        
        # Assert that the outputs are the same (this will depend on the expected behavior of autocorr_cyclo)
        np.testing.assert_allclose(abs(acorr_1[[0,1]]), abs(acorr_2[[0,1]]),rtol=1e-12, err_msg="C++ didn't give the same result as python algorithm.")
             
    def test_C_autocorr_cyclo_and_autocorr_cyclo_m_have_same_output(self):
        acorr = autocorr_cyclo(self.data, self.F, self.R, self.nb_fft, self.l_fft)
        for m in range(self.N):
            acorr_m = autocorr_cyclo_m(self.data, self.F, self.R, self.nb_fft, self.l_fft, m=m)
            np.testing.assert_allclose(abs(acorr[m]), abs(acorr_m),rtol=1e-14, err_msg="The results for autocorr_cyclo and autocorr_cyclo_m are different for m={}.".format(m))

if __name__ == '__main__':
    unittest.main()
