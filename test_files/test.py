import numpy as np
import pylab as pl
# if __name__ ==  '__main__':
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

number_counts.get_loglik_data(observables=["q_mmf3"])

t2 = time.time()

print("Time cluster data likelihood",t2-t11)

print("Time total likelihood evaluation",time.time()-t0)

n_z = number_counts.n_z
n_q = number_counts.n_obs

z = number_counts.redshift_vec
q = number_counts.obs_select_vec

print(n_z)
print(n_q)

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
