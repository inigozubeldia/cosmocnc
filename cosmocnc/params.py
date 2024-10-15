import numpy as np
from .config import *
from .cnc import *
from .cosmo import *
from .hmf import *

cnc_params_default = {

    "survey_sr": "/home/iz221/cnc/surveys/survey_sr_so_sim.py", #File where the survey scaling relations are defined
    "survey_cat": "/home/iz221/cnc/surveys/survey_cat_so_sim.py", #File where the survey catalogue(s) are defined

    #Number of cores

    "number_cores_hmf": 1,
    "number_cores_abundance": 1,
    "number_cores_data": 8,
    "number_cores_stacked":8,

    "parallelise_type": "patch", #"patch" or "redshift"

    #Precision parameters

    "n_points": 4096, ##number of points in which the mass function at each redshift (and all the convolutions) is evaluated
    "n_z": 50,
    "n_points_data_lik": 128, #number of points for the computation of the cluster data part of the likelihood
    "sigma_mass_prior": 5.,
    "downsample_hmf_bc": 1,


    #Observables and catalogue

    "load_catalogue": True,
    "likelihood_type": "unbinned", #"unbinned", "binned", or "extreme_value"
    "obs_select": "q_so_sim", #"q_mmf3_mean",
    "observables": [["q_so_sim"],["p_so_sim"]],
    "cluster_catalogue":"SO_sim_0",#"Planck_MMF3_cosmo",
    "data_lik_from_abundance":True, #if True, and if the only observable is the selection observable,
    "data_lik_type":"backward_convolutional", #"backward_convolutional" or "direct_integral". Note that "direct_integral" only works with one correlation set
    "abundance_integral_type":"fft", #"fft" or "direct"
    "compute_abundance_matrix":False, #only true if the abundance matrix is needed
    "catalogue_params":{"downsample":True},
    "apply_obs_cutoff":False,
    "get_masses":False,
    "delta_m_with_ref":False,

    #Range of abundance observables

    "obs_select_min": 6.,
    "obs_select_max": 100.,
    "z_min": 0.01,
    "z_max": 1.01,

    #cosmology and hmf parameters

    "cosmology_tool": "classy_sz", #"astropy" or "classy_sz"
    "M_min": 5e13,
    "M_max": 5e15,
    "hmf_calc": "cnc", #"cnc", "hmf", "MiraTitan", or "classy_sz"
    "hmf_type": "Tinker08",
    "mass_definition": "500c",
    "hmf_type_deriv": "numerical", #"analytical" or "numerical"
    "power_spectrum_type": "cosmopower",
    "cosmo_amplitude_parameter": "sigma_8", #"sigma_8" or "A_s"
    "Hubble_parameter": "h", # "H0" or "h"
    "cosmo_param_density": "critical", #"physical" or "critical"
    "scalrel_type_deriv": "analytical", #"analytical" or "numerical"
    "sigma_scatter_min": 1e-5,
    "interp_tinker": "linear", #"linear" or "log"


    "cosmo_model": "lcdm", # redundancy taken care of in cosmo.py
    "class_sz_cosmo_model": "lcdm", # lcdm, mnu, neff, wcdm, ede

    "class_sz_ndim_redshifts" : 100,
    "class_sz_ndim_masses" : 100,  # when using emulators this is automatically fixed.
    "class_sz_concentration_parameter" : "B13",
    "class_sz_output": 'mPk,m500c_to_m200c,m200c_to_m500c',
    "class_sz_hmf": "T08M500c", # M500 or T08M500c for Tinker et al 208 HMF defined at m500 critical.

    #Redshift errors parameters

    "z_errors": False,
    "n_z_error_integral": 100,
    "z_error_sigma_integral_range": 4.,
    "z_error_min": 1e-5, #minimum z std for which an integral over redshift in the cluster data term is performed (if "z_errors" = True)
    "z_bounds": False, #redshift bounds if there's no redshift measurement by the redshift is bounded (as in, e.g., SPT)

    "convolve_nz": False,
    "sigma_nz": 0.,

    #False detections

    "non_validated_clusters": False, #True if there are clusters which aren't validated. If so, a distribution for their selection obsevable pdf must be provided

    #Binned likelihood params

    "binned_lik_type":"z_and_obs_select", #can be "obs_select", "z", or "z_and_obs_select"
    "bins_edges_z": np.linspace(0.01,1.01,11),
    "bins_edges_obs_select": np.exp(np.linspace(np.log(6.),np.log(60),6)),

    #Stacked likelihood params

    "stacked_likelihood": False,
    "stacked_data": ["p_zc19_stacked"], #list of stacked data
    "compute_stacked_cov": True,

    #Priors

    "priors": False,
    "theta_mc_prior": False,


    # Verbose:
    "cosmocnc_verbose": "none" # none, minimal or extensive

    }

scal_rel_params_ref = {
#Planck
"alpha":1.79,
"beta":0.66,
"log10_Y_star":-0.19,
"sigma_lnq":0.173,
"bias_sz":0.8, #a.k.a. 1-b
"sigma_lnmlens":0.2,
"sigma_mlens":0.5,
"bias_lens":1.,
"dof":0.,
"bias_cmblens":0.92,
"sigma_lnp":0.22,
"corr_lnq_lnp":0.,
"a_lens":1.,
"f_false_detection":0.0, #N_F / (N_F + N_T) fraction of false detections to total detections
"f_true_validated":1.,#fraction of true clusters which have been validated
"q_cutoff":0.,

#SZiFi Planck

"alpha_szifi":1.1233,#1233, #1.1233 ?
"A_szifi": -4.3054, #Arnaud values, respectively
"sigma_lnq_szifi": 0.173,

#SPT
# spt style lkl:
"A_sz": 5.1,
"B_sz": 1.75,
"C_sz": 0.5,

"A_x": 6.5,
"B_x": 0.69,
"C_x": -0.25,

"sigma_lnYx":0.255, # 'Dx' in Bocquet's code
"dlnMg_dlnr" : 0.,

'WLbias' : 0.,
'WLscatter': 0.,

'HSTbias': 0.,
'HSTscatterLSS':0.,

'MegacamBias': 0.,
'MegacamScatterLSS': 0.,

'corr_xi_Yx': 0.1, # 'rhoSZX' in Bocquet's code
'corr_xi_WL': 0.1, # 'rhoSZWL' in Bocquet's code
'corr_Yx_WL': 0.1,  # 'rhoWLX' in Bocquet's code

'SZmPivot': 3e14
}

scaling_relation_params_default = {

#Planck

"alpha":1.79,
"beta":0.66,
"log10_Y_star":-0.19,
"sigma_lnq":0.173,
"bias_sz":0.62, #a.k.a. 1-b
"sigma_lnmlens":0.2,
"sigma_mlens":0.5,
"bias_lens":1.,
"dof":0.,
"bias_cmblens":0.92,
"sigma_lnp":0.22,
"corr_lnq_lnp":0.77,
"a_lens":1.,
"f_false_detection":0.0, #N_F / (N_F + N_T) fraction of false detections to total detections
"f_true_validated":1.,#fraction of true clusters which have been validated
"q_cutoff":0.,

#SZiFi Planck

"alpha_szifi":1.1233, #1.1233 ? True value in synthetic catalogues is 1.1233, for some reason
"A_szifi": -4.3054, #Arnaud values, respectively
"sigma_lnq_szifi": 0.173,

#SPT
# spt style lkl:
"A_sz": 5.1,
"B_sz": 1.75,
"C_sz": 0.5,

"A_x": 6.5,
"B_x": 0.69,
"C_x": -0.25,

"sigma_lnYx":0.255, # 'Dx' in Bocquet's code
"dlnMg_dlnr" : 0.,

'WLbias' : 0.,
'WLscatter': 0.,

'HSTbias': 0.,
'HSTscatterLSS':0.,

'MegacamBias': 0.,
'MegacamScatterLSS': 0.,

'corr_xi_Yx': 0.1, # 'rhoSZX' in Bocquet's code
'corr_xi_WL': 0.1, # 'rhoSZWL' in Bocquet's code
'corr_Yx_WL': 0.1,  # 'rhoWLX' in Bocquet's code

'SZmPivot': 3e14,

#ACT

"A0": np.log10(1.9e-5),
"B0": 0.08,
"C0": 0.,
"sigma_lnq_act": 0.2,

#Planck DES Y3:

"lnb_wl_sigma": 0., #prior: unit standard deviation, mean 0
"b_wl_m": 1.029,
"s_wl_m": -0.226,
}

cosmo_params_default = {

"Om0":0.315,
"Ob0":0.04897,
"Ob0h2":0.04897*0.674**2,
"Oc0h2":(0.315-0.04897)*0.674**2,
"h":0.674,
"A_s":2.08467e-09, #if amplitude_parameter == "sigma_8", this is overriden by the value given to "sigma_8" in this dictionary
"n_s":0.96,
"m_nu":0.06, #m_nu is sum of the three neutrino masses
"sigma_8":0.811, #if amplitude_paramter == "A_s", this is overriden; the amplitude is taken by the value given to "A_s" in this dictionary
"tau_reio": 0.0544,
"w0": -1.,
"Onu0": 0.00141808,
"N_eff": 3.046,

"k_cutoff": 1e8,
"ps_cutoff": 1,

}
