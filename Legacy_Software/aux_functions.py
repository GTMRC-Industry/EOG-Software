from scipy.signal import butter, lfilter
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

## FUNCTION FOR SIGNAL DIFFERENCING
def diff(signal,lag):
    """
    PARAMS
    1.Signal
    2.Lag (in number of samples)
    """
    #Discrete derivation
    sol = np.zeros_like(signal)
    #Keep in mind first n samples are discarded
    sol[lag:]=signal[lag:]-signal[:-lag]
    return sol
#High pass filter design in python
def hpf(signal,fs=1200,fc=0.05,order=4): 
    #Calculate coefficients
    b,a=butter(order,fc/(fs/2),btype='high')
    #Apply filter, NOTE: filtfilt removes delay but it does NOT work on live signals
    filtered_signal=lfilter(b,a,signal)
    return filtered_signal
#Apply matlab filter in python
def hpf_m(signal): 
    #NOTE: .mat file needs to be in same directory
    #Extract b and a from .mat file
    data=loadmat("HPF_coefficients.mat")
    if 'b' not in data or 'a' not in data:
        raise ValueError("File .mat does not contain b and a")
    b=np.squeeze(data['b'])
    print(b)
    a=np.squeeze(data['a'])
    print(a)

    filtered_signal=lfilter(b,a,signal)
    return filtered_signal
#Plot spectrum 
def plot_fft(signal, fs):
    N = len(signal) 
    f = np.fft.fftfreq(N, d=1/fs)  #frequency vec
    fft_signal = np.fft.fft(signal)
    
    # Shift 
    f_shifted = np.fft.fftshift(f)
    fft_shifted = np.fft.fftshift(np.abs(fft_signal)) / N  # normalize 
    
    plt.figure()
    plt.plot(f_shifted, fft_shifted)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('FFT of the signal')
    plt.grid(True)
    plt.show()

#ENTER MORE FUNCTIONS HERE ...


#Test code here if needed
if __name__=="__main__":
    #Testing of baseline removers
    fs = 1200           
    t = np.arange(0, 10, 1/fs)  

    # Signal example
    f_baja = 0.05          # low frequency component
    f_alta = 10         # high frequency component 
    signal = np.sin(2*np.pi*f_baja*t) + 0.5*np.sin(2*np.pi*f_alta*t)

    # Original
    plt.figure()
    plt.plot(t, signal)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Testing signal')

    # Filter
    signal_filtrada=hpf_m(signal) 

    # Filtered signal
    plt.figure()
    plt.plot(t, signal_filtrada)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Filtered signal')
    plt.show()
    plot_fft(signal_filtrada,fs)
    pass