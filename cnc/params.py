import numpy as np
from .config import *
from .cnc import *
from .cosmo import *
from .hmf import *

cnc_params_default = {

    #Number of cores

    "number_cores_hmf": 1,
    "number_cores_abundance": 1,
    "number_cores_data": 1,

    "parallelise_type": "patch", #"patch" or "redshift"

    #Precision parameters

    "n_points": 2048, #2**13, ##number of points in which the mass function at each redshift (and all the convolutions) is evaluated
    "n_obs_select": 2048,# 2**13,
    "n_z": 50,
    "n_points_data_lik": 64, #number of points for the computation of the cluster data part of the likelihood
    "sigma_mass_prior": 5.,

    #Observables and catalogue

    "likelihood_type": "unbinned", #"unbinned", "binned", or "extreme_value"
    "obs_select": "q_mmf3_mean", #"q_mmf3_mean",
    # "observables": [["q_mmf3_mean","p_zc19"]],
    "observables": [["q_mmf3_mean"]],
    "cluster_catalogue":"zc19_simulated_12",#"0.0119647",#
    # "cluster_catalogue":"Planck_MMF3_cosmo",#"Planck_MMF3_cosmo",
    #"cluster_catalogue":"q_mlens_simulated",
    "data_lik_from_abundance":True, #if True, and if the only observable is the selection observable,

    #Range of abundance observables

    "obs_select_min": 6.,
    "obs_select_max": 100.,
    "z_min": 0.01,
    "z_max": 1.01,

    #cosmology and hmf parameters

    "cosmology_tool": "astropy", #"astropy" or "classy_sz"
    "M_min": 5e13,
    "M_max": 5e15,
    "hmf_calc": "cnc", #"cnc", "hmf", or "MiraTitan"
    "hmf_type": "Tinker08",
    "mass_definition": "500c",
    "hmf_type_deriv": "numerical", #"analytical" or "numerical"
    "power_spectrum_type": "cosmopower",
    "cosmo_amplitude_parameter": "sigma_8", #"sigma_8" or "A_s"
    "scalrel_type_deriv": "analytical", #"analytical" or "numerical"

    #Redshift errors parameters

    "z_errors": True,
    "n_z_error_integral": 100,
    "z_error_sigma_integral_range": 4.,
    "z_error_min": 1e-5, #minimum z std for which an integral over redshift in the cluster data term is performed (if "z_errors" = True)

    #Only if binned likelihood is computed

    "binned_lik_type":"z_and_obs_select", #can be "obs_select", "z", or "z_and_obs_select"
    "bins_edges_z": np.linspace(0.01,1.01,11),
    "bins_edges_obs_select": np.exp(np.linspace(np.log(6.),np.log(60),6)),

    #Priors

    "priors": False,
    "theta_mc_prior": False,

    }

scaling_relation_params_default = {

"alpha":1.79,
"beta":0.66,
"log10_Y_star":-0.19,#0.646,#10.**(-0.186),
"sigma_lnq":0.173,
"bias_sz":0.62, #a.k.a. 1-b
"sigma_lnmlens":0.173,
"sigma_mlens":0.2,
"dof":0.,
"bias_cmblens":0.92,
"sigma_lnp":0.22,
"corr_lnq_lnp":0.77,
"a_lens":1.,


# spt style lkl:
"A_sz": 5.1,
"B_sz": 1.75,
"C_sz": 0.5,

"A_x": 6.5,
"B_x": 0.69,
"C_x": -0.25,

"sigma_lnYx":0.255,
# "SZmPivot" : 3e14
'corr_xi_Yx': 0.1,


}

cosmo_params_default = {

"Om0":0.315,
"Ob0":0.04897,
"h":0.674,
"A_s":1.9687e-9, #if amplitude_parameter == "sigma_8", this is overriden by the value given to "sigma_8" in this dictionary
"n_s":0.96,
"m_nu":0.06, #m_nu is sum of the three neutrino masses
"sigma_8":0.811, #if amplitude_paramter == "A_s", this is overriden; the amplitude is taken by the value given to "A_s" in this dictionary
"tau_reio": 0.0544
}

class priors:

    def __init__(self,prior_params=None):

        self.prior_params = prior_params

        #Flat priors (i.e., parameter limits if other priors are also chosen).

        self.flat_priors = {
        "Om0":[0.001,0.8],
        "Ob0":[0.001,0.8],
        "h":[0.01,5.],
        "A_s":[1e-11,1e-7],
        "n_s":[0.1,2.],
        "m_nu":[0.,2.],
        "sigma_8":[0.01,2.],

        "alpha":[0.1,10.],
        "beta":[0.01,10.],
        "log10_Y_star":[-2,2.],
        "sigma_lnq":[0.001,2.],
        "bias_sz":[0.01,2.],
        "sigma_lnmlens":[0.001,2.],
        "sigma_mlens":[0.001,2.],
        "bias_cmblens":[0.01,2.],
        "sigma_lnp":[0.01,2.],
        "corr_lnq_lnp":[0.,0.999]
        }

        # #Gaussian prior, list is mean and std
        # ### we want to move all of these in the yaml file.
        # self.gaussian_priors = {
        # "bias_cmblens":[0.93,0.05], #prior width from ZC19
        # "sigma_lnp":[0.22,0.05],
        # "corr_lnq_lnp":[0.77,0.1],
        # "alpha":[1.79,0.08], #prior from Planck15 cnc analysis and ZC19
        # "log10_Y_star":[-0.19,0.02],
        # "n_s":[0.96,0.0042], #std from Planck18
        # "sigma_lnq":[0.173,0.023], #prior from Planck15 cnc analysis
        # "bias_sz":[0.62,0.08]
        #}

    def eval_priors(self,cosmo_params,scal_rel_params):

        params = dict(cosmo_params)
        params.update(scal_rel_params)

        log_prior = 0.

        # for key in params.keys():
        #
        #     param_value = params[key]
        #
        #     #Unnormalised flat priors (effectively, boundaries imposed to parameters)
        #
        #     if key in self.flat_priors.keys():
        #
        #         if param_value < self.flat_priors[key][0] or param_value > self.flat_priors[key][1]:
        #
        #             print(key,param_value)
        #
        #             log_prior = -np.inf
        #
        #     #Gaussian priors
        #
        #     if key in self.gaussian_priors:
        #
        #         log_prior = log_prior + np.log(gaussian_1d(self.gaussian_priors[key][0]-param_value,self.gaussian_priors[key][1]))

        #Custom priors:

        #Theta MC prior
        # print('log_prior',log_prior)

        if self.prior_params["theta_mc_prior"] == True:

            cosmology = self.prior_params["cosmology"]

            theta_mc = cosmology.get_theta_mc()

            theta_mc_planck = 1.04093/100.
            theta_mc_planck_sigma = 0.00030/100.
            theta_mc_sim = 0.0104199189394683

            log_prior = log_prior + np.log(gaussian_1d(theta_mc-theta_mc_sim,theta_mc_planck_sigma))

        #Baryon density Planck prior
        # print('log_prior2',log_prior)
        # Ob0h2_Planck = 0.02237 #Planck18
        # Ob0h2_sim = 0.022245895
        # Ob0h2_Planck_sigma = 0.00015
        #
        # log_prior = log_prior + np.log(gaussian_1d(cosmo_params["Ob0"]*cosmo_params["h"]**2-Ob0h2_sim,Ob0h2_Planck_sigma))

        return log_prior

def gaussian_1d(x,sigma):

    return np.exp(-x**2/(2.*sigma**2))/(np.sqrt(2.*np.pi)*sigma)
