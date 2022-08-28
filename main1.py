import numpy as np
import matplotlib.pyplot as plt


fs = 44100#sampling frequency
CHUNK = 10000
signal_time = 20 # seconds

def sine(freq,fs,secs):
    data = np.arange(fs*secs)/(fs*1.0)
    wave = np.sin(freq*2*np.pi*data)
    return wave

a1 = sine(150,fs,120)
a2 = sine(300,fs,120)

signal = a1+a2

def sine2(freq, fs, start, stop):
    data = np.arange(fs*stop)/(fs*1.0)
    data[fs*start] = 0
    wave = np.sin(freq)

a3 = sine2(150, fs, 0, 120)
a4 = sine2(300, fs, 60, 120)

afft = np.abs(np.fft.fft(signal[0:CHUNK]))
freqs = np.linspace(0,fs,CHUNK)[0:int(fs/2)]
spectrogram_chunk = freqs/np.amax(freqs*1.0)

# Plot spectral analysis
plt.plot(freqs[0:250],afft[0:250])
plt.show()

number_of_chunks = 1000

# Empty spectrogram
Spectrogram = np.zeros(shape = [CHUNK,number_of_chunks])

for i in range(number_of_chunks):
    afft = np.abs(np.fft.fft(signal[i*CHUNK:(1+i)*CHUNK]))
    freqs = np.linspace(0,fs,CHUNK)[0:int(fs/2)]
    #plt.plot(spectrogram_chunk[0:250],afft[0:250])
    #plt.show()
    spectrogram_chunk = afft/np.amax(afft*1.0)
    #print(signal[i*CHUNK:(1+i)*CHUNK].shape)
    try:
        Spectrogram[:,i]=spectrogram_chunk
    except:
        break


import cv2
Spectrogram = Spectrogram[0:250,:]
cv2.imshow('spectrogram',np.uint8(255*Spectrogram/np.amax(Spectrogram)))
cv2.waitKey()
cv2.destroyAllWindows()
