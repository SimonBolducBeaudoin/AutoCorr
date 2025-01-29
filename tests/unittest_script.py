import unittest
import numpy as np
from SBB.AutoCorr.autocorr import autocorr_cyclo
from fractions import Fraction

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
        self.l_fft = 2048
        self.Mmax = -1
        
    def test_autocorr_same_output_for_different_nb_fft(self):
        # Test for nb_fft = 1
        nb_fft_1 = 1
        acorr_m_1 = autocorr_cyclo(self.data, self.F, self.R, nb_fft_1, self.l_fft, Mmax=self.Mmax)
        
        # Test for nb_fft = 2
        nb_fft_2 = 2
        acorr_m_2 = autocorr_cyclo(self.data, self.F, self.R, nb_fft_2, self.l_fft, Mmax=self.Mmax)
        
        # Assert that the outputs are the same (this will depend on the expected behavior of autocorr_cyclo)
        np.testing.assert_array_equal(acorr_m_1, acorr_m_2, err_msg="The autocorrelation results for different nb_fft values do not match.")

if __name__ == '__main__':
    unittest.main()
