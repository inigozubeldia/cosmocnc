import numpy as np
# import cosmo
# import hmf
import pylab as pl
import cnc as cnc
# import cat
import time
# import sr

#cosmology = cosmo.cosmology_model()
#halo_mass_function = hmf.halo_mass_function(cosmology=cosmology,M_min=1e12,M_max=1e15)

#z = 0.5
#m,dndm = halo_mass_function.eval_hmf(z,log=True)

#print(dndm)

#pl.loglog(np.exp(m),dndm)
#pl.show()

scaling_relations = {"q_mmf3_mean":cnc.sr.scaling_relations(observable="q_mmf3_mean"),"m_lens":cnc.sr.scaling_relations(observable="m_lens")}
scaling_relations["q_mmf3_mean"].initialise_scaling_relation()

cosmology = cnc.cosmo.cosmology_model()

catalogue_planck = cnc.cat.cluster_catalogue(catalogue_name="Planck_MMF3_cosmo")
n_tot = len(catalogue_planck.catalogue["q_mmf3_mean"])
n_no_z = len(np.where(catalogue_planck.catalogue["z"] < 0)[0])

n_m_lens = len(np.where(catalogue_planck.catalogue["m_lens"] > 0)[0])

print("Clusters Planck",n_tot,n_no_z)
print("N m lens",n_m_lens)

t0 = time.time()

number_counts = cnc.cluster_number_counts(cosmology=cosmology,catalogue=catalogue_planck,scaling_relations=scaling_relations)
number_counts.get_number_counts()

t1 = time.time()

print("Time total number counts",time.time()-t0)

number_counts.get_loglik_data(observables=["q_mmf3_mean","m_lens"])

t2 = time.time()

print("Time cluster data likelihood",t2-t1)

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
