import numpy as np
import pylab as pl
import cosmocnc as cnc
import time

number_counts = cnc.cluster_number_counts()

data_lik_from_abundances = [True,False]

labels = ["Forward","Backward"]

for i in range(0,len(data_lik_from_abundances)):

    number_counts.cnc_params["path_to_cosmopower_organization"] = "/Users/boris/Work/CLASS-SZ/SO-SZ/SBI/cosmopower-organization/"

    number_counts.cnc_params["compute_abundance_matrix"] = True
    number_counts.cnc_params["cluster_catalogue"] = 'SPT2500d'
    # number_counts.cnc_params["observables"] = [["xi","Yx","WLMegacam","WLHST"]] 
    number_counts.cnc_params["observables"] = [["xi"]] 
    number_counts.cnc_params["obs_select"] = 'xi'

    number_counts.cnc_params["data_lik_from_abundance"] = data_lik_from_abundances[i]
    number_counts.cnc_params["number_cores_hmf"] = 1
    number_counts.cnc_params["number_cores_abundance"] = 1
    number_counts.cnc_params["number_cores_data"] = 8

    number_counts.cnc_params["obs_select_min"] = 5.
    number_counts.cnc_params["obs_select_max"] = 47.

    number_counts.cnc_params["n_points"] = 1024*64 #64*4#2**13, ##number of points in which the mass function at each redshift (and all the convolutions) is evaluated
    number_counts.cnc_params["n_obs_select"] = number_counts.cnc_params["n_points"]
    number_counts.cnc_params["parallelise_type"] = "redshift"


    number_counts.cnc_params["scalrel_type_deriv"] = "numerical"


    number_counts.cnc_params["z_max"] = 3.
    number_counts.cnc_params["z_min"] = 0.25
    number_counts.cnc_params["n_z"] = 100

    number_counts.cnc_params["M_min"] = 1e13
    number_counts.cnc_params["M_max"] = 1e16

    number_counts.cnc_params["apply_obs_cutoff"] = False

    number_counts.scal_rel_params["dof"] = 0.
    number_counts.scal_rel_params["q_cutoff"] = 2. ## see section 3.1.1 of https://arxiv.org/pdf/1812.01679.pdf



    number_counts.cnc_params["abundance_integral_type"] = "fft"

    number_counts.cnc_params["likelihood_type"] = "unbinned"
    number_counts.cnc_params["cosmology_tool"] = "classy_sz"

    number_counts.cnc_params["class_sz_ndim_redshifts"] = 500
    number_counts.cnc_params["class_sz_ndim_masses"] = 100
    number_counts.cnc_params["class_sz_concentration_parameter"] = "B13"

    number_counts.cnc_params["class_sz_hmf"] = "T08M500c"
    number_counts.cnc_params["hmf_calc"] = "classy_sz"


    number_counts.initialise()

    log_lik = number_counts.get_log_lik()

    print(labels[i])
    print("log lik",log_lik)
    print("")
