import numpy as np
import pylab as pl
import scipy.signal as signal
import time

t0 = time.time()

nx = 128
print(nx)
a = np.random.rand(nx,nx)
b = np.random.rand(nx,nx)

convolved = signal.convolve(a,b,mode="same",method="direct")

print(time.time()-t0)
