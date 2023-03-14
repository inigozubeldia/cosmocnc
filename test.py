import numpy as np
import cosmo
import hmf
import pylab as pl
import cnc
import cat
import time

#cosmology = cosmo.cosmology_model()
#halo_mass_function = hmf.halo_mass_function(cosmology=cosmology,M_min=1e12,M_max=1e15)

#z = 0.5
#m,dndm = halo_mass_function.eval_hmf(z,log=True)

#print(dndm)

#pl.loglog(np.exp(m),dndm)
#pl.show()

cosmology = cosmo.cosmology_model()

catalogue_planck = cat.cluster_catalogue()
n_tot = len(catalogue_planck.catalogue["q"])
n_no_z = len(np.where(catalogue_planck.catalogue["z"] < 0)[0])

print("Clusters Planck",n_tot,n_no_z)

t0 = time.time()

number_counts = cnc.cluster_number_counts(cosmology=cosmology)
number_counts.get_number_counts()

print("Time total",time.time()-t0)

n_z = number_counts.n_z
n_q = number_counts.n_obs

z = number_counts.redshift_vec
q = number_counts.obs_select_vec

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
