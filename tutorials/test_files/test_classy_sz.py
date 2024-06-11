import numpy as np
import cosmocnc as cosmocnc


number_counts = cosmocnc.cluster_number_counts()
number_counts.cnc_params.update(
{
"path_to_cosmopower_organization" : "/Users/boris/Work/CLASS-SZ/SO-SZ/cosmopower-organization/",
"compute_abundance_matrix" : True,
"cluster_catalogue" : "SO_sim_" + str(0),
"observables" : [["q_so_sim"]],
"obs_select" : "q_so_sim",
"data_lik_from_abundance" : True,
"number_cores_hmf" : 1,
"number_cores_abundance" : 8,
"number_cores_data" : 16,
"obs_select_min" : 5.,
"obs_select_max" : 200.,

"parallelise_type" : "redshift",
"scalrel_type_deriv" : "numerical",

"z_max" : 3.,
"z_min" : 0.01,
"n_z" : 5000,

"M_min" : 1e13,
"M_max" : 1e16,
"n_points" : 1024*32, #64*4#2**13, ##number of points in which the mass function at each redshift (and all the convolutions) is evaluated

"apply_obs_cutoff" : False,
    
'cosmology_tool' : "classy_sz",
    
    
"class_sz_ndim_redshifts" : 500,
"class_sz_ndim_masses" : 100,  # when using cosmopower this is automatically fixed. 
"class_sz_concentration_parameter" : "B13",
"class_sz_hmf": "T08M500c", 
"hmf_calc": "classy_sz"
})
number_counts.scal_rel_params.update(
{
"dof" : 0.,
"bias_sz" : 0.8,
}
)

number_counts.cosmo_params.update(
{
'h':0.7,
}
)



print("initializing")
number_counts.initialise()
print("initialized")

print("getting number counts")
number_counts.get_number_counts()
print("got number counts")
