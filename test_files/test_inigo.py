import numpy as np
import pylab as pl
import cnc
import time
number_counts = cnc.cluster_number_counts()
cnc_params = number_counts.cnc_params
cnc_params["priors"] = False
number_counts.cnc_params["catalogue_params"]["downsample"] = True
cnc_params["data_lik_from_abundance"] = False
cnc_params["number_cores_abundance"] = 8
cnc_params["number_cores_data"] = 12
cnc_params["number_cores_stacked"] = 1
cnc_params["likelihood_type"] =  "unbinned"
cnc_params["cluster_catalogue"] = "SPT2500d"
cnc_params["observables"] = [["xi","Yx","WLMegacam","WLHST"]]
#cnc_params["observables"] = [["xi"]]
cnc_params["obs_select"] = "xi"
cnc_params["stacked_likelihood"] = False
cnc_params["stacked_data"] = 0
cnc_params["cosmology_tool"] = "classy_sz"
cnc_params["scalrel_type_deriv"] = "numerical"
cnc_params["n_points"] = 4096
cnc_params["n_obs_select"] = 4096
cnc_params["n_points_data_lik"] =  64
cnc_params["n_z"] = 50
cnc_params["M_min"] = 5e13
cnc_params["M_max"] = 5e15
cnc_params["sigma_mass_prior"] = 7.
cnc_params["parallelise_type"] = "redshift"
cnc_params["z_min"] = 0.25
cnc_params["z_max"] = 2.
cnc_params['obs_select_min'] = 5.
cnc_params["z_errors"] = True
cnc_params["n_z_error_integral"] = 10
number_counts.initialise()
t0 = time.time()
n = 10
bias_vec = np.linspace(0.1,1.5,n)
sigma_8_vec = np.linspace(0.75,0.82,n)
Om0_vec = np.linspace(0.25,0.35,n)
bias_cmb_lens_vec = np.linspace(0.8,1.1,n)
sigma_cmblens_vec = np.linspace(0.01,0.3,n)
corr_vec = np.linspace(-0.9,0.9,n)
a_lens_vec = np.linspace(9,11.,n)
sigma_sz = np.linspace(0.05,0.3,n)
h_vec = np.linspace(0.2,0.75,n)
alpha_vec = np.linspace(1.,1.9,n)
log10_Y_star_vec = np.linspace(-0.23,-0.15,n)
false_vec = np.linspace(0.,0.05,n)
lik_vec = np.zeros(n)
a = sigma_8_vec
for i in range(0,n):
    t0 = time.time()
    cosmo_params = number_counts.cosmo_params
    scal_rel_params = number_counts.scal_rel_params
    #scal_rel_params["f_false_detection"] = false_vec[i]
    print(cosmo_params)
    cosmo_params["h"] = 0.68
#    cosmo_params["Om0"] = Om0_vec[i]
#    scal_rel_params["alpha"] = alpha_vec[i]
#
#    scal_rel_params["bias_sz"] = bias_vec[i]
#    scal_rel_params["sigma_lnp"] = sigma_cmblens_vec[i]
#    scal_rel_params["bias_cmblens"] = bias_cmb_lens_vec[i]
#    scal_rel_params["a_lens"] = a_lens_vec[i]
#    scal_rel_params["corr_lnq_lnp"] = corr_vec[i]
#    scal_rel_params["sigma_lnq"] = sigma_sz[i]
#    scal_rel_params["log10_Y_star"] = log10_Y_star_vec[i]
#    cosmo_params["h"] = h_vec[i]
    number_counts.update_params(cosmo_params,scal_rel_params)
    lik_vec[i] = number_counts.get_log_lik()
    print(i,lik_vec[i])
    print("time",time.time()-t0)
print(lik_vec)
lik_vec = np.exp(lik_vec-np.max(lik_vec))
pl.plot(a,lik_vec)
pl.legend()
# pl.savefig("/home/iz221/cnc/figures/test_unbiased_spt.pdf")
pl.show()
