from cobaya.likelihood import Likelihood
from typing import Optional, Sequence
import numpy as np
# from pkg_resources import resource_filename
# from astropy.io import fits
# from cobaya.theory import Theory

class theta_mc_prior(Likelihood):
    # variables from yaml
    theta_mc_mean: float
    theta_mc_std: float

    def initialize(self):
        self.minus_half_invvar = - 0.5 / self.theta_mc_std ** 2

    def get_requirements(self):
        return {'theta_mc': None}

    def logp(self, **params_values):
        theta_mc_theory = self.provider.get_param("theta_mc")
        return self.minus_half_invvar * (theta_mc_theory - self.theta_mc_mean) ** 2

class cnc_likelihood(Likelihood):
#     number_cores : Optional[str] = 8
#     number_cores_hmf : Optional[str] = 8
#     parallelise_type : Optional[str] = "redshift" #"patch" or "redshift"
#
#     #Precision parameters
#
#     n_points : Optional[str] = 2**13 #number of points in which the mass function at each redshift (and all the convolutions) is evaluated
#     n_obs_select : Optional[str] =  2**13
#     n_z : Optional[str] =  100
#
#     n_points_data_lik : Optional[str] =  128 #number of points for the computation of the cluster data part of the likelihood
#     sigma_mass_prior : Optional[str] =  5.
#
#     #Observables and catalogue
#
#     likelihood_type : Optional[str] =  "unbinned" #"unbinned", "binned", or "extreme_value"
#     obs_select : Optional[str] =  "q_mmf3_mean" #"q_mmf3_mean",
#     observables : Optional[str] =  [["q_mmf3_mean","p_zc19"]]
# #    "observables": [["q_mmf3_mean"]],
#     observables_mass_estimation : Optional[str] =  ["q_mmf3_mean"]
#     cluster_catalogue :Optional[str] =  "zc19_simulated_12"
#     #"cluster_catalogue":"zc19_simulated_12",#"Planck_MMF3_cosmo",
#     #"cluster_catalogue":"q_mlens_simulated",
#     data_lik_from_abundance :Optional[str] =  True #if True, and if the only observable is the selection observable,
#
#     #Range of abundance observables
#
#     obs_select_min : Optional[str] =  6.
#     obs_select_max : Optional[str] =  100.
#     z_min : Optional[str] =  0.01
#     z_max : Optional[str] =  1.01
#
#     #hmf parameters
#
#     M_min : Optional[str] =  1e13
#     M_max : Optional[str] =  1e16
#     hmf_calc : Optional[str] =  "cnc" #"cnc", "hmf", or "MiraTitan"
#     hmf_type : Optional[str] =  "Tinker08"
#     mass_definition : Optional[str] =  "500c"
#     hmf_type_deriv : Optional[str] =  "numerical" #"analytical" or "numerical"
#     power_spectrum_type : Optional[str] =  "cosmopower"
#     cosmo_amplitude_parameter : Optional[str] =  "sigma_8" #"sigma_8" or "A_s"
#     scalrel_type_deriv : Optional[str] =  "analytical" #"analytical" or "numerical"
#
#     #Redshift errors parameters
#
#     z_errors: Optional[str] =  False
#     n_z_error_integral: Optional[str] =  5
#     z_error_sigma_integral_range: Optional[str] =  3
#     z_error_min: Optional[str] =  1e-2 #minimum z std for which an integral over redshift in the cluster data term is performed (if "z_errors" = True)
#
#     #Only if binned likelihood is computed
#
#     binned_lik_type:Optional[str] =  "z_and_obs_select" #can be "obs_select", "z", or "z_and_obs_select"
#     bins_edges_z: Optional[str] =  np.linspace(0.01,1.01,11)
#     bins_edges_obs_select: Optional[str] =  np.exp(np.linspace(np.log(6.),np.log(60),6))
#
#     #Priors
#
#     priors: Optional[str] =  True
#     theta_mc_prior: Optional[str] =  True

    def initialize(self):
        super().initialize()
    def get_requirements(self):
        return {"sz_unbinned_cluster_counts": {}}

    def _get_theory(self, **params_values):
        theory = self.theory.get_sz_unbinned_cluster_counts()
        return theory

    def logp(self, **params_values):
        _derived = params_values.pop("_derived", None)
        theory = self._get_theory(**params_values)
        loglkl = theory
        return loglkl
