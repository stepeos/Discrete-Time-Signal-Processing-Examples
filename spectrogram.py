#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

# the spectrogram scale can also be lambda[pi] over n(samples)
# in that case, if lambda is pi, thus Pi=f_sample/2


# Fixing random state for reproducibility
np.random.seed(2022)

dt = 0.0005
t = np.arange(0.0, 2.5, dt)
Fs = int(1.0 / dt)  # the sampling frequency
f1 = Fs/2/4
s1 = 0.25 * np.sin(2 * np.pi * 3 * f1 * t)
s2 = np.sin(2 * np.pi * f1 * t)
s3 = np.sin(2 * np.pi * 2 * f1 * t)

# create a transient "chirp"
s1[t <= 0] = s1[4 <= t] = 0
s2[t <= 0] = s2[t[-1]<= t] = 0
s3[t <= t[-1]//2] = s2[t[-1] <= t] = 0

# add some noise into the mix
nse = 0.01 * np.random.random(size=len(t))

x = s1 + s2 + s3 + nse  # the signal

##############################
# fft computation
##############################
CHUNK = t.shape[0] // 15
# afft = np.abs(fft(x[-CHUNK:]))[:CHUNK//2] * 2 / CHUNK
# freqs = fftfreq(CHUNK, 1/Fs)[:CHUNK//2]
# Plot fft analysis
# plt.plot(freqs,afft)
# plt.xlim(0, 100)
# plt.show()

number_of_chunks = x.shape[0]//CHUNK

# Empty spectrogram
Spectrogram = np.zeros((number_of_chunks, CHUNK//2))

for i in range(number_of_chunks):
    afft = np.abs(fft(x[i*CHUNK:(i+1)*CHUNK]))[:CHUNK//2] * 2 / CHUNK
    freqs = fftfreq(CHUNK, 1/Fs)[:CHUNK//2]
    spectrogram_chunk = 10 * np.log10(afft/np.amax(afft*1.0))
    try:
        Spectrogram[i] = spectrogram_chunk
    except:
        break
lamb = freqs / (0.5 * Fs)

#######################################
# END FFT COMPUTATION
#######################################

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
ax1.plot(t, x)
ax1.set_ylabel("Signal [1]")
ax1.set_xlabel("Time [s]")
ax1.set_xlim(t[0], t[-1])
# The `specgram` method returns 4 objects. They are:
# - Pxx: the periodogram
# - freqs: the frequency vector
# - bins: the centers of the time bins
# - im: the matplotlib.image.AxesImage instance representing the data in the plot
Pxx, freqs_, bins, im = ax2.specgram(x, NFFT=CHUNK, Fs=Fs, noverlap=CHUNK//4*3, scale_by_freq=False)
ax2.set_ylim(0,Fs/2)
ax2.set_ylabel("Freq [Hz]")
ax2.set_xlabel("Time [s]")
ax3.pcolormesh(np.arange(t[CHUNK//2],t[-1], t[-1]/Spectrogram.shape[0]), lamb, Spectrogram.T, shading='auto')
# ax3.set_xlim(t[0], t[-1])
ax3.set_ylim(0, 1)
ax3.set_ylabel(f"$\lambda/\pi$  @Fs={Fs:.0f}Hz")
plt.tight_layout()
plt.show()
# %%
