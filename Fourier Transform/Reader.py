import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

f = pyedflib.EdfReader("PN01-1.edf")
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))

for i in np.arange(n):
     sigbufs[i, :] = f.readSignal(i)

print(sigbufs)
print(signal_labels)

signal = []
for i in range(0, len(sigbufs)):
        print(sigbufs[i])
        signal[0:90000] = sigbufs[i][10000:100000]
        tf.signal.fft(signal)
        plt.plot(signal)
        plt.show()