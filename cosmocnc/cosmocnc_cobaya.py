from typing import Optional
import numpy as np
from cobaya.likelihoods.base_classes import InstallableLikelihood

class CNCLike(InstallableLikelihood):

    #Attributes

    #Catalogue and survey files

    survey_sr: Optional[str] = "path" #File where the survey scaling relations are defined
    survey_cat: Optional[str] = "path" #File where the survey catalogue(s) are defined

    #Number of cores

    number_cores_hmf: Optional[int] =  1
    number_cores_abundance: Optional[int] =  1
    number_cores_data: Optional[int] =  8
    number_cores_stacked: Optional[int] = 8

    parallelise_type: Optional[str] = "patch" #"patch" or "redshift"

    #Precision parameters

    n_points: Optional[int] = 4096 ##number of points in which the mass function at each redshift (and all the convolutions) is evaluated
    n_z: Optional[int] = 50
    n_points_data_lik: Optional[int] = 128 #number of points for the computation of the cluster data part of the likelihood
    sigma_mass_prior: Optional[float] = 5.
    downsample_hmf_bc: Optional[int] = 1
    padding_fraction: Optional[float] = 0.

    #Observables and catalogue

    load_catalogue: Optional[bool] = True
    precompute_cnc_quantities_catalogue: Optional[bool] = True
    likelihood_type: Optional[str] = "unbinned" #"unbinned", "binned", or "extreme_value"
    obs_select:Optional[str] = "q_so_sim" #"q_mmf3_mean",
    observables: Optional[str] = [["q_so_sim"],["p_so_sim"]]
    cluster_catalogue: Optional[str] = "SO_sim_0"#"Planck_MMF3_cosmo",
    data_lik_from_abundance: Optional[bool] = True #if True, and if the only observable is the selection observable,
    data_lik_type: Optional[str] = "backward_convolutional" #"backward_convolutional" or "direct_integral". Note that "direct_integral" only works with one correlation set
    abundance_integral_type: Optional[str] = "fft" #"fft" or "direct"
    compute_abundance_matrix: Optional[bool] = False #only true if the abundance matrix is needed
    catalogue_params: Optional[dict] = {"downsample":True}
    apply_obs_cutoff: Optional[bool] = False
    compute_masses: Optional[bool] = False
    delta_m_with_ref: Optional[bool] = False

    #Range of abundance observables

    obs_select_min: Optional[float] = 6.
    obs_select_max: Optional[float] = 200.
    z_min: Optional[float] = 0.01
    z_max: Optional[str] = 1.01

    #cosmology and hmf parameters

    cosmology_tool: Optional[str] = "cobaya" #"astropy", "classy_sz", or "cobaya"
    M_min: Optional[float] = 5e13
    M_max: Optional[float] = 5e15
    M_min_extended: Optional[float] = None
    hmf_calc:  Optional[str] = "cnc" #"cnc", "hmf", "MiraTitan", or "classy_sz"
    hmf_type: Optional[str] = "Tinker08"
    mass_definition: Optional[str] = "500c"
    hmf_type_deriv: Optional[str] = "numerical" #"analytical" or "numerical"
    power_spectrum_type: Optional[str] = "cobaya" # "cosmopower" or "cobaya"
    cosmo_amplitude_parameter: Optional[str] = "sigma_8" #"sigma_8" or "A_s" , not needed if cosmology_tool = "cobaya"
    cosmo_param_density: Optional[str] = "critical" #"physical" or "critical"
    scalrel_type_deriv: Optional[str] = "analytical" #"analytical" or "numerical"
    sigma_scatter_min: Optional[float] = 1e-5
    interp_tinker: Optional[str] = "linear" #"linear" or "log", only if "hmf_calc"=="cnc"

    #Not needed if cosmology_tool = "cobaya"

    cosmo_model: Optional[str] = "lcdm" 
    class_sz_ndim_masses : Optional[int] = 100  # when using emulators this is automatically fixed.
    class_sz_concentration_parameter : Optional[str] = "B13"
    class_sz_output: Optional[str] = 'mPk,m500c_to_m200c,m200c_to_m500c'
    class_sz_hmf: Optional[str] = "T08M500c" # M500 or T08M500c for Tinker et al 208 HMF defined at m500 critical.
    class_sz_use_m500c_in_ym_relation: Optional[int] = 1
    class_sz_use_m200c_in_ym_relation: Optional[int] = 0

    #Redshift errors parameters

    z_errors: Optional[bool] = False
    n_z_error_integral: Optional[int] = 100
    z_error_sigma_integral_range: Optional[float] = 4.
    z_error_min: Optional[float] = 1e-5 #minimum z std for which an integral over redshift in the cluster data term is performed (if "z_errors" = True)
    z_bounds: Optional[bool] = False #redshift bounds if there's no redshift measurement by the redshift is bounded (as in, e.g., SPT)

    convolve_nz: Optional[bool] = False
    sigma_nz: Optional[float] = 0.

    #False detections

    non_validated_clusters: Optional[bool] = False, #True if there are clusters which aren't validated. If so, a distribution for their selection obsevable pdf must be provided

    #Binned likelihood params

    binned_lik_type: Optional[str] = "z_and_obs_select" #can be "obs_select", "z", or "z_and_obs_select"
    bins_edges_z: Optional[str] = np.linspace(0.01,1.01,11)
    bins_edges_obs_select: Optional[str] = np.exp(np.linspace(np.log(6.),np.log(60),6))

    #Stacked likelihood params

    stacked_likelihood: Optional[bool] = False
    stacked_data: Optional[str] = ["p_zc19_stacked"] #list of stacked data
    compute_stacked_cov: Optional[bool] = True

    # Verbose:
    
    cosmocnc_verbose: Optional[str] = "none" # none, minimal or extensive

    def initialize(self):

        self._cnc_initialized = False

        from cosmocnc import cluster_number_counts

        self.cnc = cluster_number_counts()
        
        for attr_name in self.__annotations__:

            self.cnc.cnc_params[attr_name] = getattr(self,attr_name)

        self.log.info("CNCLike: First stage initialization (static config) complete.")

        self.requirements = self.get_requirements()


    def get_requirements(self):

        zs = np.linspace(getattr(self,"z_min"),getattr(self,"z_max"),getattr(self,"n_z"))
        zs = np.concatenate([[0.],zs,[1100.]])
        kmax = 50.

        requirements = {
        "Pk_interpolator":{
            "z":zs,
            "k_max":kmax,
            "nonlinear":[False],
        },
        "Hubble":{"z":zs},
        "angular_diameter_distance":{"z":zs},
        "angular_diameter_distance_2":{"z_pairs":zs},
        "comoving_radial_distance":{"z":zs},
        "sigma8_z":{"z":zs},
        "Omega_b":{"z":zs},
        "Omega_cdm":{"z":zs},
        "Omega_nu_massive":{"z":zs},
        "Pk_grid":{
            "z":zs,
            "k_max":kmax,
            "nonlinear":[False],
        },
        }

        return requirements

    def logp(self,**params_values):

        #cosmocnc has two dictionaries of parameters; here just setting them to all the parameters in the yaml file.

        self.cnc.cosmo_params = {
            name: value
            for name, value in params_values.items()
        }
        
        self.cnc.scal_rel_params = {
            name: value
            for name, value in params_values.items()
        }

        if self._cnc_initialized is False:
            # Safe to initialize now — all param values are set

            self.cnc.cnc_params["cobaya_provider"] = self.provider
                    
            self.cnc.cosmo_params["H0"] = self.provider.get_param("H0")
            self.cnc.cosmo_params["ombh2"] = self.provider.get_param("ombh2")
            self.cnc.cosmo_params["omch2"] = self.provider.get_param("omch2")
            self.cnc.cosmo_params["logA"] = self.provider.get_param("logA")
            self.cnc.cosmo_params["ns"] = self.provider.get_param("ns")
            self.cnc.cosmo_params["tau"] = self.provider.get_param("tau")

            self.cnc.initialise()
            self._cnc_initialized = True
            self.log.info("CNCLike: cosmocnc fully initialized on first call to logp.")

        log_like = self.cnc.get_log_lik()

        #print("log lik",log_like)

        return log_like

    @classmethod
    def get_allow_agnostic(cls):
        """
        Allow this component to accept any unassigned input parameters.
        Use with caution!
        """
        return True