import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

from scipy.io.wavfile import write

# Audio clip properties
# sampling rate = 16000 Hz , this is quantization in time
# bits per sample = 16, this is quantization in amplitude

spf = wave.open('helloworld.wav', 'r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
print("numpy signal:", signal.shape)

plt.plot(signal)
plt.title("Hello world without echo")
plt.show()

delta = np.array([1., 0., 0.])
noecho = np.convolve(signal, delta)
print("noecho signal:", noecho.shape)
assert(np.abs(noecho[:len(signal)] - signal).sum() < 0.000001)

noecho = noecho.astype(np.int16) # make sure you do this, otherwise, you will get VERY LOUD NOISE
write('noecho.wav', 16000, noecho)

filt = np.zeros(16000)
filt[0] = 1
filt[4000] = 0.6
filt[8000] = 0.3
filt[12000] = 0.2
filt[15999] = 0.1
out = np.convolve(signal, filt)

out = out.astype(np.int16) # make sure you do this, otherwise, you will get VERY LOUD NOISE
write('out.wav', 16000, out)

plt.plot(out)
plt.title("Hello world with small echo")
plt.show()

