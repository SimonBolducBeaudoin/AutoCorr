import numpy as np
from scipy.fft import rfft

def autocorr_cyclo_py_Vpy(np_data: np.ndarray, F: int, R: int, l_fft: int, Mmax: int=-1, norm: str="backward"):
    """
    Python equivalent of the C++ autocorr_cyclo_py function.

    Parameters:
    np_data (np.ndarray): Input data array (should be of dtype int16).
    F (int): Frequency.
    R (int): Sampling rate.
    l_fft (int): FFT length.
    Mmax (int): Maximum M value.
    norm (str): Normalization method ('backward', 'ortho', or 'forward').

    Returns:
    np.ndarray: Computed autocorrelation values.
    """

    # Verify that F is exactly on the discrete Fourier transform grid
    if (F * l_fft) % R != 0:
        raise ValueError("F must be exactly represented by the DFT's frequency spectrum.")

    # Compute the index of the pump
    i_F = (F * l_fft) // R

    # Check that the input data is large enough
    if np_data.size < l_fft:
        raise ValueError("data length too small.")

    # Ensure correct dtype
    if np_data.dtype != np.int16:
        raise TypeError("Input data must be of dtype np.int16.")

    # Call the equivalent autocorr_cyclo function (assumed to exist in Python)
    gamma = autocorr_cyclo_Vpy(np_data, i_F, l_fft, Mmax)

    # Normalization following NumPy FFT conventions
    if norm == "backward":
        inv_norm = 1.0
    elif norm == "ortho":
        inv_norm = 1.0 / np.sqrt(l_fft)
    elif norm == "forward":
        inv_norm = 1.0 / l_fft
    else:
        raise ValueError("Invalid norm value; should be 'backward', 'ortho', or 'forward'.")

    # Apply normalization
    gamma *= inv_norm

    # Return the numpy array
    return gamma



def autocorr_cyclo_Vpy(data: np.ndarray, i_F: int, l_fft: int, Mmax: int):
    """
    Compute the cyclic autocorrelation of the input signal.

    Parameters:
    data (np.ndarray): Input data array (should be 1D and of dtype int16).
    i_F (int): Index of frequency.
    l_fft (int): FFT length.
    Mmax (int): Maximum M value.

    Returns:
    np.ndarray: Computed cyclic autocorrelation values.
    """

    # Compute M, N, and i_f_max
    if l_fft % 2 == 0:
        M = (l_fft // 2 - 1) // i_F
        N = (2 * (l_fft // 2 - 1)) // i_F
        i_f_max = l_fft // 2 - 1
    else:
        M = ((l_fft - 1) // 2) // i_F
        N = (l_fft - 1) // i_F
        i_f_max = (l_fft - 1) // 2

    if Mmax <= 0:
        Mmax = N

    Mmax = min(N, Mmax)
    M = min(M, Mmax)
    N = min(N, Mmax)

    # Frequency-space accumulator
    gamma = np.zeros((2 * N + 1, l_fft), dtype=np.complex128)

    # Number of data chunks
    Nchunk = len(data) // (l_fft)

    for i_chunk in range(Nchunk):
        stride = l_fft * i_chunk
        gs = data[stride: stride + l_fft].astype(np.float64)
        hs = rfft(gs)
        
        # gamma[0, :l_fft//2+1] += np.abs(hs[:]) ** 2
        gamma[0, :l_fft//2] += np.abs(hs[:-1]) ** 2

        # m >= M
        for m in range(1, M + 1):
            l_half = (i_F * m) // 2 + ((i_F * m) % 2)
            gamma[m, :l_half + 1]       += hs[:l_half + 1] * hs[ i_F*m :i_F*m -l_half-1:-1]
            gamma[m, i_F*m+1:i_f_max+1] += hs[i_F*m+1:i_f_max+1] * hs[np.r_[i_F*m+1:i_f_max+1]-i_F*m].conjugate()
            # didn't do the rest

    # m = 0
    gamma[0,(l_fft-1)//2+1:l_fft] = gamma[0,l_fft-((l_fft-1)//2+1):0:-1]
    
    # m >= M
    
    for m in range(1, M + 1):
        l_half = (i_F * m) // 2 + ((i_F * m) % 2)
        # gamma[m,l_half+1:i_F*m+1]      = gamma[m,i_F*m-(l_half+1):-1:-1]
        gamma[m,l_half+1:i_F*m+1]      = gamma[m,0:i_F*m-l_half][::-1]
        gamma[m,i_f_max+1+i_F*m:l_fft] = gamma[m,l_fft-1-(i_f_max+1+i_F*m)+i_F*m:l_fft-1-(l_fft)+i_F*m:-1]
    
    # Normalize
    gamma *= 1.0 / (Nchunk)

    return gamma
