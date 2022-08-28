import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

# the spectrogram scale can also be lambda[pi] over n(samples)
# in that case, lambda=Pi=f_sample/2


# Fixing random state for reproducibility
np.random.seed(19680801)

dt = 0.000005
t = np.arange(0.0, 2.5, dt)
f1 = 40
# s1 = np.sin(2 * np.pi * 100 * t**2)
s2 = np.sin(2 * np.pi * f1 * t)
s3 = np.sin(2 * np.pi * 2 * f1 * t)
# s3 = 2 * np.sin(2 * np.pi * 300 * t)
# s4 = 2 * np.sin(2 * np.pi * 400 * t)

# create a transient "chirp"
# s1[t <= 0] = s1[4 <= t] = 0
s2[t <= 0] = s2[t[-1]//2 <= t] = 0
s3[t <= t[-1]//2] = s2[t[-1] <= t] = 0
# s3[t <= 6] = s3[10 <= t] = 0
# s4[t <= 15] = s4[20 <= t] = 0

# add some noise into the mix
nse = 0.01 * np.random.random(size=len(t))

# x = s1 + s2 + s3 + s4 + nse  # the signal
x = s2 + s3 + nse  # the signal
NFFT = 1024  # the length of the windowing segments
Fs = int(1.0 / dt)  # the sampling frequency

##############################
# fft computation
##############################
CHUNK = t.shape[0] // 1000
afft = np.abs(fft(x[:CHUNK]))[:CHUNK//2] * 2 / CHUNK
freqs = fftfreq(CHUNK, 1/Fs)[:CHUNK//2]
# afft = np.abs(np.fft.fft(signal[0:CHUNK]))
# freqs = np.linspace(0,Fs,CHUNK)[0:int(Fs/2)]
# spectrogram_chunk = freqs/np.amax(freqs*1.0)

# Plot fft analysis
# plt.plot(freqs,afft)
# plt.xlim(0, 100)
# plt.show()

number_of_chunks = x.shape[0]//CHUNK

# Empty spectrogram
Spectrogram = np.zeros(shape = [CHUNK,number_of_chunks])

for i in range(number_of_chunks):
    afft = np.abs(np.fft.fft(x[i*CHUNK:(1+i)*CHUNK]))
    freqs = np.linspace(0,Fs,CHUNK)[0:int(Fs/2)]
    #plt.plot(spectrogram_chunk[0:250],afft[0:250])
    #plt.show()
    spectrogram_chunk = afft/np.amax(afft*1.0)
    #print(signal[i*CHUNK:(1+i)*CHUNK].shape)
    try:
        Spectrogram[:,i]=spectrogram_chunk
    except:
        break
#######################################
# END FFT COMPUTATION
#######################################

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
ax1.plot(t, x)
ax1.set_ylabel("Signal [1]")
ax1.set_xlabel("Time [s]")
Pxx, freqs, bins, im = ax2.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=900, scale_by_freq=False)
ax2.set_ylim(0,100)
# The `specgram` method returns 4 objects. They are:
# - Pxx: the periodogram
# - freqs: the frequency vector
# - bins: the centers of the time bins
# - im: the matplotlib.image.AxesImage instance representing the data in the plot
ax2.set_ylabel("Freq [Hz]")
ax2.set_xlabel("Time [s]")
plt.tight_layout()
plt.show()