from cobaya.theories.classy import classy
from copy import deepcopy
from typing import NamedTuple, Sequence, Union, Optional
from cobaya.tools import load_module
import logging
import os
import numpy as np
from .utils import *

class cnc(classy):

    survey_sr : Optional[str] =  "none"
    survey_cat: Optional[str] =  "none"
    number_cores_hmf : Optional[str] = 8
    number_cores_abundance : Optional[str] = 8
    number_cores_data : Optional[str] = 8
    number_cores_stacked : Optional[str] = 8
    parallelise_type : Optional[str] = "patch" #"patch" or "redshift"

    #Precision parameters

    n_points : Optional[str] = 2048*2 #number of points in which the mass function at each redshift (and all the convolutions) is evaluated
    n_obs_select : Optional[str] =  2048*2 #Deprecated
    n_z : Optional[str] =  50
    downsample_hmf_bc: Optional[str] = 1

    n_points_data_lik : Optional[str] =  128 #number of points for the computation of the cluster data part of the likelihood
    sigma_mass_prior : Optional[str] =  5.



    #Observables and catalogue

    likelihood_type : Optional[str] =  "unbinned" #"unbinned", "binned", or "extreme_value"
    obs_select : Optional[str] =  "q_mmf3_mean" #"q_mmf3_mean",
    observables : Optional[str] =  [["q_mmf3_mean","p_zc19"]]
#    "observables": [["q_mmf3_mean"]],
    observables_mass_estimation : Optional[str] =  ["q_mmf3_mean"]
    cluster_catalogue :Optional[str] =  "zc19_simulated_12"
    #"cluster_catalogue":"zc19_simulated_12",#"Planck_MMF3_cosmo",
    #"cluster_catalogue":"q_mlens_simulated",
    data_lik_from_abundance :Optional[str] =  True #if True, and if the only observable is the selection observable,
    compute_abundance_matrix: Optional[str] = False
    apply_obs_cutoff: Optional[str] = False
    catalogue_params = {"downsample":True}
    data_lik_type: Optional[str] = "backward_convolutional"
    abundance_integral_type: Optional[str] = "fft" #fft or direct
    delta_m_with_ref: Optional[str] = False
    Hubble_parameter: Optional[str] = "h"
    sigma_scatter_min: Optional[str] = 1e-5

    cosmo_param_density: Optional[str] = "critical"
    #Range of abundance observables

    obs_select_min : Optional[str] =  6.
    obs_select_max : Optional[str] =  100.
    z_min : Optional[str] =  0.01
    z_max : Optional[str] =  1.01

    #hmf parameters

    cosmology_tool:  Optional[str] = "astropy" # current options are astropy (and cosmopower), classy, and classy_sz (fast-mode)
    M_min : Optional[str] =  1e13
    M_max : Optional[str] =  1e16
    hmf_calc : Optional[str] =  "cnc" #"cnc", "hmf", or "MiraTitan"
    hmf_type : Optional[str] =  "Tinker08"
    mass_definition : Optional[str] =  "500c"
    hmf_type_deriv : Optional[str] =  "numerical" #"analytical" or "numerical"
    power_spectrum_type : Optional[str] =  "cosmopower"
    cosmo_amplitude_parameter : Optional[str] =  "sigma_8" #"sigma_8" or "A_s"
    scalrel_type_deriv : Optional[str] =  "analytical" #"analytical" or "numerical"
    interp_tinker : Optional[str] =  "linear" #"analytical" or "numerical"
    padding_fraction : Optional[str] =  0. 

    #Stacked parameters

    stacked_likelihood: Optional[str] = False,
    stacked_data: Optional[str] = ["p_zc19_stacked"]
    compute_stacked_cov: Optional[str] = True

    likelihood_cal_alt: Optional[str] = False
    observables_cal_alt: Optional[str] = ["p_zc19"]

    #Redshift errors parameters

    z_errors: Optional[str] =  False
    n_z_error_integral: Optional[str] =  5
    z_error_sigma_integral_range: Optional[str] =  3
    z_error_min: Optional[str] =  1e-2 #minimum z std for which an integral over redshift in the cluster data term is performed (if "z_errors" = True)


    #Only if binned likelihood is computed

    binned_lik_type:Optional[str] =  "z_and_obs_select" #can be "obs_select", "z", or "z_and_obs_select"
    bins_edges_z: Optional[str] =  np.linspace(0.01,1.8,20)
    bins_edges_obs_select: Optional[str] =  np.exp(np.linspace(np.log(5.),np.log(60),12))

    #Priors

    priors: Optional[str] =  False
    theta_mc_prior: Optional[str] =  False

    # mass pivot for spt-style lkl:
    SZmPivot: Optional[str] = 3e14

    # scaling relation parameter:
    dof: Optional[str] = 0
    q_cutoff: Optional[str] = 0

    # class_sz parameters
    cosmo_model: Optional[str] = "lcdm"
   # class_sz_ndim_redshifts: Optional[str] = 500
    class_sz_ndim_masses: Optional[str] = 100
    class_sz_concentration_parameter: Optional[str] = "B13"
    class_sz_hmf: Optional[str] = "T08M500c"
    class_sz_output: Optional[str] = "m500c_to_m200c"


    # verbose parameter
    cosmocnc_verbose: Optional[str] = "none"


    def initialize(self):

        """Importing cnc from the correct path, if given, and if not, globally."""
        # this line is just for wrapping into cobaya.
        # cnc is fully independent from classy.
        self.classy_module = self.is_installed()

        from cosmocnc import cluster_number_counts

        self.cnc = cluster_number_counts()
        set_verbosity(self.cnc.cnc_params["cosmocnc_verbose"])
        self.logger = logging.getLogger(__name__)
        self.cnc.cnc_params["cosmology_tool"] = self.cosmology_tool
        self.cnc.cnc_params["number_cores_abundance"] = self.number_cores_abundance
        self.cnc.cnc_params["number_cores_hmf"] = self.number_cores_hmf
        self.cnc.cnc_params["number_cores_data"] = self.number_cores_data
        self.cnc.cnc_params["number_cores_stacked"] = self.number_cores_stacked
        self.cnc.cnc_params["parallelise_type"] = self.parallelise_type # "redshift", #"patch" or "redshift"
        self.cnc.cnc_params["survey_sr"] = self.survey_sr
        self.cnc.cnc_params["survey_cat"] = self.survey_cat

        #Precision parameters

        self.cnc.cnc_params["n_points"] = int(self.n_points) # 2**13, #number of points in which the mass function at each redshift (and all the convolutions) is evaluated
        self.cnc.cnc_params["n_obs_select"] = int(self.n_obs_select) # 2**13, #Deprecated
        self.cnc.cnc_params["n_z"] = int(self.n_z) # 100,
        self.cnc.cnc_params["downsample_hmf_bc"] = int(self.downsample_hmf_bc)

        self.cnc.cnc_params["n_points_data_lik"] = int(self.n_points_data_lik) # 128, #number of points for the computation of the cluster data part of the likelihood
        self.cnc.cnc_params["sigma_mass_prior"] = self.sigma_mass_prior # 5.,

        self.cnc.cnc_params["cosmo_model"] = self.cosmo_model
        self.cnc.cnc_params["cosmo_param_density"] = self.cosmo_param_density
        self.cnc.cnc_params["interp_tinker"] = self.interp_tinker

        #Observables and catalogue

        self.cnc.cnc_params["likelihood_type"] = self.likelihood_type # "unbinned", #"unbinned", "binned", or "extreme_value"
        self.cnc.cnc_params["obs_select"] = self.obs_select # "q_mmf3_mean", #"q_mmf3_mean",
        self.cnc.cnc_params["observables"] = self.observables # [["q_mmf3_mean","p_zc19"]],
    #    "observables": [["q_mmf3_mean"]],
        self.cnc.cnc_params["cluster_catalogue"] = self.cluster_catalogue #"zc19_simulated_12",
        #"cluster_catalogue":"zc19_simulated_12",#"Planck_MMF3_cosmo",
        #"cluster_catalogue":"q_mlens_simulated",
        self.cnc.cnc_params["data_lik_from_abundance"] = self.data_lik_from_abundance #True, #if True, and if the only observable is the selection observable,
        self.cnc.cnc_params["apply_obs_cutoff"] = self.apply_obs_cutoff
        self.cnc.cnc_params["catalogue_params"] = self.catalogue_params # 6.,
        self.cnc.cnc_params["data_lik_type"] = self.data_lik_type
        self.cnc.cnc_params["abundance_integral_type"] = self.abundance_integral_type
        self.cnc.cnc_params["delta_m_with_ref"] = self.delta_m_with_ref
        self.cnc.cnc_params["padding_fraction"] = self.padding_fraction

        #Range of abundance observables

        self.cnc.cnc_params["obs_select_min"] = self.obs_select_min # 6.,
        self.cnc.cnc_params["obs_select_max"] = self.obs_select_max # 100.,

        self.cnc.cnc_params["z_min"] = self.z_min # 0.01,
        self.cnc.cnc_params["z_max"] = self.z_max # 1.01,

        #hmf parameters

        self.cnc.cnc_params["M_min"] = float(self.M_min) # 1e13,
        self.cnc.cnc_params["M_max"] = float(self.M_max) # 1e16,
        self.cnc.cnc_params["hmf_calc"] = self.hmf_calc # "cnc", #"cnc", "hmf", or "MiraTitan"
        self.cnc.cnc_params["hmf_type"] = self.hmf_type # "Tinker08",
        self.cnc.cnc_params["mass_definition"] = self.mass_definition # "500c",
        self.cnc.cnc_params["hmf_type_deriv"] = self.hmf_type_deriv # "numerical", #"analytical" or "numerical"
        self.cnc.cnc_params["power_spectrum_type"] = self.power_spectrum_type # "cosmopower",
        self.cnc.cnc_params["cosmo_amplitude_parameter"] = self.cosmo_amplitude_parameter # "sigma_8", #"sigma_8" or "A_s"
        self.cnc.cnc_params["scalrel_type_deriv"] = self.scalrel_type_deriv # "analytical", #"analytical" or "numerical"
        self.cnc.cnc_params["sigma_scatter_min"] = self.sigma_scatter_min

        #Redshift errors parameters

        self.cnc.cnc_params["z_errors"] = self.z_errors # False,
        self.cnc.cnc_params["n_z_error_integral"] = self.n_z_error_integral # 5,
        self.cnc.cnc_params["z_error_sigma_integral_range"] = self.z_error_sigma_integral_range # 3,
        self.cnc.cnc_params["z_error_min"] = self.z_error_min # 1e-2, #minimum z std for which an integral over redshift in the cluster data term is performed (if "z_errors" = True)

        #Only if binned likelihood is computed

        self.cnc.cnc_params["binned_lik_type"] = self.binned_lik_type #"z_and_obs_select", #can be "obs_select", "z", or "z_and_obs_select"
        self.cnc.cnc_params["bins_edges_z"] = self.bins_edges_z # np.linspace(0.01,1.01,11),
        self.cnc.cnc_params["bins_edges_obs_select"] = self.bins_edges_obs_select # np.exp(np.linspace(np.log(6.),np.log(60),6)),

        self.cnc.cnc_params["stacked_likelihood"] = self.stacked_likelihood
        self.cnc.cnc_params["stacked_data"] = self.stacked_data
        self.cnc.cnc_params["compute_stacked_cov"] = self.compute_stacked_cov
        self.cnc.cnc_params["Hubble_parameter"] = self.Hubble_parameter

        self.cnc.cnc_params["likelihood_cal_alt"] = self.likelihood_cal_alt
        self.cnc.cnc_params["observables_cal_alt"] = self.observables_cal_alt

        #Priors

        # self.cnc.cnc_params["priors"] = self.priors # True,
        # self.cnc.cnc_params["theta_mc_prior"] = self.theta_mc_prior # True,

        # scaling relation param:
        # dealing with degrees of freedom and truncations
        self.cnc.cnc_params["dof"] = self.dof
        self.cnc.cnc_params["q_cutoff"] = self.q_cutoff


        # class_sz parameters
       # self.cnc.cnc_params["class_sz_ndim_redshifts"] = self.class_sz_ndim_redshifts
        self.cnc.cnc_params["class_sz_ndim_masses"] = self.class_sz_ndim_masses
        self.cnc.cnc_params["class_sz_concentration_parameter"] = self.class_sz_concentration_parameter
        self.cnc.cnc_params["class_sz_hmf"] = self.class_sz_hmf
    #    self.cnc.cnc_params["class_sz_cosmo_model"] = self.class_sz_cosmo_model

        self.cnc.cnc_params["class_sz_output"] = self.class_sz_output

        # verbose paramater
        self.cnc.cnc_params["cosmocnc_verbose"] = self.cosmocnc_verbose

        super(classy,self).initialize()
        self.extra_args["output"] = self.extra_args.get("output","")
        # print(self.extra_args)
        # exit(0)

        self.cnc.initialise()

        self.derived_extra = []
        self.log.info("Initialized")

    def must_provide(self, **requirements):

        if "sz_unbinned_cluster_counts" in requirements:

            # make sure cobaya still runs as it does for standard classy

            requirements.pop("sz_unbinned_cluster_counts")

            # specify the method to collect the new observable

            self.collectors["sz_unbinned_cluster_counts"] = Collector(
                    method="get_log_lik", # name of the method in classy.pyx
                    args_names=[],
                    args=[])

    # get the required new observable

    def get_sz_unbinned_cluster_counts(self):

        # thats the function passed to soliket

        # print('passing to cobaya lkl')
        cls = deepcopy(self._current_state["sz_unbinned_cluster_counts"])
        # print('cls',cls)

        return cls

    def calculate(self, state, want_derived=True, **params_values_dict):

        params_values = params_values_dict.copy()

        cosmo_params = self.cnc.cosmo_params
        assign_parameter_value(cosmo_params,params_values,"Om0")
        assign_parameter_value(cosmo_params,params_values,"Ob0")
        assign_parameter_value(cosmo_params,params_values,"h")
        assign_parameter_value(cosmo_params,params_values,"sigma_8")
        assign_parameter_value(cosmo_params,params_values,"A_s")
        assign_parameter_value(cosmo_params,params_values,"n_s")
        assign_parameter_value(cosmo_params,params_values,"m_nu")
        assign_parameter_value(cosmo_params,params_values,"Onu0")
        assign_parameter_value(cosmo_params,params_values,"w0")
        assign_parameter_value(cosmo_params,params_values,"tau_reio")
        assign_parameter_value(cosmo_params,params_values,"Ob0h2")
        assign_parameter_value(cosmo_params,params_values,"Oc0h2")
        assign_parameter_value(cosmo_params,params_values,"Onu0h2")

        scal_rel_params = self.cnc.scal_rel_params

        assign_parameter_value(scal_rel_params,params_values,"alpha")
        assign_parameter_value(scal_rel_params,params_values,"log10_Y_star")
        assign_parameter_value(scal_rel_params,params_values,"bias_sz")
        assign_parameter_value(scal_rel_params,params_values,"bias_cmblens")
        assign_parameter_value(scal_rel_params,params_values,"sigma_lnq")
        assign_parameter_value(scal_rel_params,params_values,"sigma_lnp")
        assign_parameter_value(scal_rel_params,params_values,"corr_lnq_lnp")
        assign_parameter_value(scal_rel_params,params_values,"a_lens")
        assign_parameter_value(scal_rel_params,params_values,"alpha_szifi")
        assign_parameter_value(scal_rel_params,params_values,"A_szifi")
        assign_parameter_value(scal_rel_params,params_values,"sigma_lnq_szifi")

        # SPT-style parameters:

        assign_parameter_value(scal_rel_params,params_values,"A_sz")
        assign_parameter_value(scal_rel_params,params_values,"B_sz")
        assign_parameter_value(scal_rel_params,params_values,"C_sz")

        assign_parameter_value(scal_rel_params,params_values,"A_x")
        assign_parameter_value(scal_rel_params,params_values,"B_x")
        assign_parameter_value(scal_rel_params,params_values,"C_x")

        assign_parameter_value(scal_rel_params,params_values,"dlnMg_dlnr")

        assign_parameter_value(scal_rel_params,params_values,"sigma_lnYx")

        assign_parameter_value(scal_rel_params,params_values,"corr_xi_Yx")
        assign_parameter_value(scal_rel_params,params_values,"corr_xi_WL")
        assign_parameter_value(scal_rel_params,params_values,"corr_Yx_WL")

        assign_parameter_value(scal_rel_params,params_values,"WLbias")
        assign_parameter_value(scal_rel_params,params_values,"WLscatter")

        assign_parameter_value(scal_rel_params,params_values,"HSTbias")
        assign_parameter_value(scal_rel_params,params_values,"HSTscatterLSS")

        assign_parameter_value(scal_rel_params,params_values,'MegacamBias')
        assign_parameter_value(scal_rel_params,params_values,'MegacamScatterLSS')
        assign_parameter_value(scal_rel_params,params_values,'SZmPivot')
        # updating scaling relations params that are not varied in mcmc, but passed in input
        scal_rel_params['dof'] = self.cnc.cnc_params["dof"]
        scal_rel_params['q_cutoff'] = self.cnc.cnc_params["q_cutoff"]

        self.cnc.update_params(cosmo_params,scal_rel_params)

        # Gather products
        # print(self.collectors.items())
        # d, d_extra = self._cnc_get_derived_all(derived_requested=want_derived)
        # print('d',d)
        derived = {}
        for p in self.output_params:
            if p == 'theta_mc':
                cosmology = self.cnc.cosmology
                derived[p] = cosmology.get_theta_mc()
        # print('derived:',derived)
        # print('self.output_params', self.output_params)
        if want_derived:
            state["derived"] = {p: derived.get(p) for p in self.output_params}

        derived = {}

        for p in self.output_params:

            if p == 'theta_mc':

                cosmology = self.cnc.cosmology
                derived[p] = cosmology.get_theta_mc()

            if p == 'Onu0':

                derived[p] = self.cnc.cosmo_params["Onu0"]

            if p == "sigma_8":

                derived[p] = self.cnc.cosmo_params["sigma_8"]

            if p == "A_s":

                derived[p] = self.cnc.cosmo_params["A_s"]

            if p == "Om0":

                derived[p] = self.cnc.cosmo_params["Om0"]

            if p == "Ob0":

                derived[p] = self.cnc.cosmo_params["Ob0"]

            if p == "Om0h2":

                derived[p] = self.cnc.cosmo_params["Om0h2"]

            if p == "Ob0h2":

                derived[p] = self.cnc.cosmo_params["Ob0h2"]

        if want_derived:

            state["derived"] = {p: derived.get(p) for p in self.output_params}

        self.logger.info(f"Derived parameters: {derived}")

        for product, collector in self.collectors.items():

            method = getattr(self.cnc, collector.method)
            state[product] = method()

        # exit(0)

    def close(self):

        return 1
    #
    @classmethod

    def is_installed(cls, **kwargs):

        return load_module('cosmocnc')

def assign_parameter_value(lik_dict,cobaya_dict,parameter):
    # print("cobaya_dict:",cobaya_dict)

    if parameter in cobaya_dict:

        lik_dict[parameter] = cobaya_dict[parameter]
    if parameter == 'h' and 'H0' in cobaya_dict.keys():
        # print('cobaya dict:',cobaya_dict.keys())
        lik_dict[parameter] = cobaya_dict['H0']/100.


    if parameter == 'h' and 'H0' in cobaya_dict.keys():
        # print('cobaya dict:',cobaya_dict.keys())
        lik_dict[parameter] = cobaya_dict['H0']/100.

# this just need to be there as it's used to fill-in self.collectors in must_provide:

class Collector(NamedTuple):

    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence] = None
    post: Optional[callable] = None
