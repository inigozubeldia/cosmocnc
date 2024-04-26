import numpy as np
import pylab as pl
import cnc
import time

number_counts = cnc.cluster_number_counts()
number_counts.initialise()

for i in range(0,10):

    t0 = time.time()

    number_counts.log_lik_unbinned()
    number_counts.reinitialise_computation()

    t1 = time.time()

    print("Time total",t1-t0)
