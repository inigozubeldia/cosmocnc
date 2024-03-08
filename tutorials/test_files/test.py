import numpy as np
import pylab as pl
import cnc
import time

number_counts = cnc.cluster_number_counts()
number_counts.initialise()

t0 = time.time()

number_counts.get_number_counts()

t1 = time.time()

print("Time total number counts",time.time()-t0)

log_lik_binned = number_counts.get_lik_binned()

t11 = time.time()

print("Time binned",t11-t1)

#pl.plot(number_counts.obs_select_vec,number_counts.abundance_matrix[5,:])

#number_counts.get_loglik_data()

t2 = time.time()

print("Time cluster data likelihood",t2-t11)

print("Time total likelihood evaluation",time.time()-t0)

n_z = number_counts.n_z
n_q = number_counts.n_obs

z = number_counts.redshift_vec
q = number_counts.obs_select_vec

n_tot = number_counts.n_tot

np.savetxt("n_z.txt",np.c_[z,n_z])
np.savetxt("n_q.txt",np.c_[q,n_q])

print("N tot",n_tot)

pl.plot(z,n_z)
pl.plot(z,np.zeros(len(z)))
pl.axvline(x=0.,color="k",linestyle="dashed")
pl.xlabel("$z$")
pl.ylabel("$dN/dz$")
pl.show()

pl.semilogx(q,n_q)
pl.plot(q,np.zeros(len(q)))
pl.xlabel("$q$")
pl.ylabel("$dN/dq$")
pl.axvline(x=6.,color="k",linestyle="dashed")
pl.show()
