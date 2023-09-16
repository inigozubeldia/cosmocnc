import cnc as cnc
import classy_sz
import numpy as np
number_counts = cnc.cluster_number_counts()

cnc_params = number_counts.cnc_params

#### set params here
cnc_params_new = {

    'number_cores_abundance' : 10,
    'number_cores_hmf' : 10,
    'number_cores_data' : 10,

    'cosmology_tool': "classy_sz",
    "mass_definition": "200c",
    "hmf_calc": "classy_sz",
    # "hmf_calc": "cnc",

    'observables': [["q_act"]],
    'obs_select': "q_act",
    'z_errors' : False,
    'cluster_catalogue' : 'act',
    'scalrel_type_deriv' : "numerical",

    "M_min": 5e13,
    "M_max": 5e15,

    'z_min' : 0.0,
    'z_max' : 2.,
    'bins_edges_z': np.linspace(0.,2.,21), ## this needs to be set ! why ??

    'obs_select_min': 5.,
    'obs_select_max': 100.,
    'bins_edges_obs_select': np.exp(np.linspace(np.log(5.),np.log(100),6)),
    'dof': 3.,
    'h' : 0.68,
    'sigma_8' : 0.81,
    'n_s' : 0.965,
    'Om0' : 0.31,
    'Ob0' : 0.049,

    'tenToA0' : 1.9e-5,
    'B0 ': 0.08,
    'SZmPivot' : 4.25e14,
    'bias_sz' : 1.,
    'sigma_lnq' : 0.2
}
for k,v in cnc_params_new.items():
    cnc_params[k] = v
number_counts.cnc_params = cnc_params

number_counts.initialise()

#
cosmo_params = number_counts.cosmo_params
scal_rel_params = number_counts.scal_rel_params
print(cosmo_params)

# cosmo_params["H0"] = 68.
for k,v in cnc_params_new.items():
    cosmo_params[k] = v
    scal_rel_params[k] = v

number_counts.update_params(cosmo_params,scal_rel_params)
# # print()

number_counts.get_number_counts()
# number_counts.get_log_lik()
exit(0)

number_counts.get_abundance_matrix()
# log_lik_binned = number_counts.get_lik_binned()
dndz = number_counts.n_z
dndq = number_counts.n_obs
cnc_zs = number_counts.redshift_vec
cnc_qs = number_counts.obs_select_vec
print("dndz:",dndz)
print("dndq:",dndq)
print("cnc_zs:",cnc_zs)
print("cnc_qs:",cnc_qs)
