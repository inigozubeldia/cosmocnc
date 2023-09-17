
import cnc as cnc
import classy_sz
import numpy as np
number_counts = cnc.cluster_number_counts()
cnc_params = number_counts.cnc_params
#### set params here
cnc_params_new = {
    # "number_cores": 1,
    # "number_cores_hmf": 1,
    'cosmology_tool': "classy_sz",

    "n_points": 2**7,#2**7, #number of points in which the mass function at each redshift (and all the convolutions) is evaluated
    "M_min": 1e13,
    "M_max": 1e16,
    "hmf_type": "Tinker08",
    "mass_definition": "200c",
    "hmf_type_deriv": "numerical",
    "power_spectrum_type": "cosmopower",

    "obs_select_min": 6.,
    "obs_select_max": 100.,
    "n_obs_select": 500,
    "z_min": 0.01,
    "z_max": 1.01,
    "n_z": 500,

    # "obs_select": "q_mmf3_mean", #"q_mmf3_mean",
    # "n_patches": 1,
    "cov_patch_dependent":False,
    "obs_select_uncorrelated":False,
    "all_layers_uncorrelated":False, #if True, all observables have uncorrelated scatter
    "last_layer_uncorrelated":False, #if True, it means that the last layer of the observables is uncorrelated
    "first_layer_power_law":False,
    # "obs_mass": ["q_mmf3_mean"],
    # "observables": [["q_mmf3_mean"]],

    # "cluster_catalogue":"Planck_MMF3_cosmo",

    'observables': [["q_act"]],
    'obs_select': "q_act",
    'cluster_catalogue' : 'act',


    "cosmo_amplitude_parameter": "A_s",
    "bins_edges_z": np.linspace(0.01,1.01,11),
    "bins_edges_obs_select": np.exp(np.linspace(np.log(6.),np.log(60),6)),
    "non_validated_clusters": False,
    "binned_lik_type":"z_and_obs_select",

    "stacked_likelihood": False,
    # "stacked_data": ["p_zc19_stacked"], #list of stacked data
    # "compute_stacked_cov": False,
    #
    # #Parms to compute mass calibration likelihood in an alternative way
    #
    "likelihood_cal_alt": False,
    # # "observables_cal": ["p_zc19"],
    #
    # #Priors
    #
    # "priors": False,
    # "theta_mc_prior": False,
    "catalogue_params":{"downsample":False},

    "hmf_calc": "classy_sz",

    "number_cores_hmf": 1,
    "number_cores_abundance": 1,
    "number_cores_data": 1,
    "number_cores_stacked":1,
    "parallelise_type": "patch",
    "scalrel_type_deriv": "numerical",
    'apply_obs_cutoff' : True,
    "abundance_integral_type":"fft",
    "compute_abundance_matrix":True,
    # 'm_ncdm'

    'tenToA0' : 1.9e-5,
    'B0 ': 0.08,
    'SZmPivot' : 4.25e14,
    'bias_sz' : 1.,
    'sigma_lnq' : 0.173,
    # 'sigma_lnq' : 0.
# 'A_ym'  : 1.9e-05,
# 'B_ym'  : 0.08,
'C0' : 0.,

    "Om0":0.3096,
    "Ob0":0.04897,
    "h":0.6766,
    "A_s":1.9687e-9,
    "n_s":0.96,
    "m_nu":0.06,
    'tau_reio':  0.0561,
    "dof":3.,

}
for k,v in cnc_params_new.items():
    cnc_params[k] = v
number_counts.cnc_params = cnc_params
number_counts.initialise()

cosmo_params = number_counts.cosmo_params
scal_rel_params = number_counts.scal_rel_params
# print(cosmo_params)

# cosmo_params["H0"] = 68.
for k,v in cnc_params_new.items():
    cosmo_params[k] = v
    scal_rel_params[k] = v

number_counts.get_number_counts()
number_counts.get_abundance_matrix()
# log_lik_binned = number_counts.get_lik_binned()
dndz = number_counts.n_z
dndq = number_counts.n_obs
cnc_zs = number_counts.redshift_vec
cnc_qs = number_counts.obs_select_vec
