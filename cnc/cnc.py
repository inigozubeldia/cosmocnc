import numpy as np
import pylab as pl
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.stats as stats
from numba import jit
from .cosmo import *
from .hmf import *
from .sr import *
from .cat import *
from .params import *
from .utils import *
import time

class cluster_number_counts:

    def __init__(self,cnc_params=cnc_params_default):

        self.cnc_params = cnc_params
        self.cosmo_params = cosmo_params_default
        self.scal_rel_params = scaling_relation_params_default

        self.abundance_matrix = None
        self.n_obs_matrix = None
        self.hmf_matrix = None
        self.n_tot = None
        self.n_binned = None
        self.abundance_tensor = None
        self.n_obs_false = 0.

        self.hmf_extra_params = {}

    #Loads data (catalogue and scaling relation data)

    def initialise(self):

        self.cosmology = cosmology_model(cosmo_params=self.cosmo_params,
                                         cosmology_tool = self.cnc_params["cosmology_tool"],
                                         power_spectrum_type=self.cnc_params["power_spectrum_type"],
                                         amplitude_parameter=self.cnc_params["cosmo_amplitude_parameter"],
                                         )

        if self.cnc_params["load_catalogue"] == True:

            self.catalogue = cluster_catalogue(catalogue_name=self.cnc_params["cluster_catalogue"],
                                               precompute_cnc_quantities=True,
                                               bins_obs_select_edges=self.cnc_params["bins_edges_obs_select"],
                                               bins_z_edges=self.cnc_params["bins_edges_z"],
                                               observables=self.cnc_params["observables"],
                                               obs_select=self.cnc_params["obs_select"],
                                               cnc_params = self.cnc_params,
                                               scal_rel_params=self.scal_rel_params)

        elif self.cnc_params["load_catalogue"] == False:

            self.catalogue = None

        self.scaling_relations = {}

        for observable_set in self.cnc_params["observables"]:

            for observable in observable_set:

                self.scaling_relations[observable] = scaling_relations(observable=observable,cnc_params=self.cnc_params,catalogue=self.catalogue)
                self.scaling_relations[observable].initialise_scaling_relation()

        if self.cnc_params["likelihood_cal_alt"] == True:

            for observable in self.cnc_params["observables_cal_alt"]:

                self.scaling_relations[observable] = scaling_relations(observable=observable,cnc_params=self.cnc_params,catalogue=self.catalogue)
                self.scaling_relations[observable].initialise_scaling_relation()

        if self.cnc_params["stacked_likelihood"] == True:

            self.stacked_data_labels = self.cnc_params["stacked_data"]

            for key in self.stacked_data_labels:

                observable = self.catalogue.stacked_data[key]["observable"]

                if observable not in self.cnc_params["observables"]:

                    self.scaling_relations[observable] = scaling_relations(observable=observable,cnc_params=self.cnc_params,catalogue=self.catalogue)
                    self.scaling_relations[observable].initialise_scaling_relation()

        self.scatter = scatter(params=self.scal_rel_params,catalogue=self.catalogue)
        # self.priors = priors(prior_params={"cosmology":self.cosmology,"theta_mc_prior":self.cnc_params["theta_mc_prior"]})

        if self.cnc_params["hmf_calc"] == "MiraTitan":

            import MiraTitanHMFemulator

            self.MT_emulator = MiraTitanHMFemulator.Emulator()
            self.hmf_extra_params["emulator"] = self.MT_emulator

        # for key in self.cnc_params:
        #
        #     print(key,self.cnc_params[key])
        #
        # for key in self.scal_rel_params:
        #
        #     print(key,self.scal_rel_params[key])
        #
        # for key in self.cosmo_params:
        #
        #     print(key,self.cosmo_params[key])

    #Updates parameter values (cosmological and scaling relation)

    def reinitialise(self):

        self.abundance_matrix = None
        self.n_obs_matrix = None
        self.hmf_matrix = None
        self.n_tot = None
        self.abundance_tensor = None

    def update_params(self,cosmo_params,scal_rel_params):

        self.cosmo_params = cosmo_params
        self.cosmology.update_cosmology(cosmo_params,cosmology_tool=self.cnc_params["cosmology_tool"])
        self.scal_rel_params = {}

        for key in scal_rel_params.keys():

            self.scal_rel_params[key] = scal_rel_params[key]

        self.scatter = scatter(params=self.scal_rel_params,catalogue=self.catalogue)

        self.abundance_matrix = None
        self.n_obs_matrix = None
        self.hmf_matrix = None
        self.n_tot = None
        self.abundance_tensor = None
        self.n_obs_matrix_fd = None

    #Computes the hmf as a function of redshift

    def get_hmf(self):

        self.const = constants()

        #Define redshift and observable ranges

        self.redshift_vec = np.linspace(self.cnc_params["z_min"],self.cnc_params["z_max"],self.cnc_params["n_z"])
        self.obs_select_vec = np.linspace(self.cnc_params["obs_select_min"],self.cnc_params["obs_select_max"],self.cnc_params["n_obs_select"])

        #Evaluate some useful quantities (to be potentially passed to scaling relations)

        self.D_A = self.cosmology.background_cosmology.angular_diameter_distance(self.redshift_vec).value
        self.E_z = self.cosmology.background_cosmology.H(self.redshift_vec).value/(self.cosmology.cosmo_params["h"]*100.)
        self.D_l_CMB = self.cosmology.background_cosmology.angular_diameter_distance_z1z2(self.redshift_vec,self.cosmology.z_CMB).value
        self.rho_c = self.cosmology.background_cosmology.critical_density(self.redshift_vec).value*1000.*self.const.mpc**3/self.const.solar
        self.E_z0p6 = self.cosmology.background_cosmology.H(0.6).value/(self.cosmology.cosmo_params["h"]*100.)

        #Evaluate the halo mass function

        self.halo_mass_function = halo_mass_function(cosmology=self.cosmology,hmf_type=self.cnc_params["hmf_type"],
        mass_definition=self.cnc_params["mass_definition"],M_min=self.cnc_params["M_min"],
        M_max=self.cnc_params["M_max"],n_points=self.cnc_params["n_points"],type_deriv=self.cnc_params["hmf_type_deriv"],
        hmf_calc=self.cnc_params["hmf_calc"],extra_params=self.hmf_extra_params)

        n_cores = self.cnc_params["number_cores_hmf"]
        indices_split = np.array_split(np.arange(self.cnc_params["n_z"]),n_cores)

        t0 = time.time()

        if self.cnc_params["hmf_calc"] == "cnc" or self.cnc_params["hmf_calc"] == "hmf":

            self.hmf_matrix = np.zeros((self.cnc_params["n_z"],self.cnc_params["n_points"]))

            def f_mp(rank,out_q):

                return_dict = {}

                for i in range(0,len(indices_split[rank])):

                    ln_M,hmf_eval = self.halo_mass_function.eval_hmf(self.redshift_vec[indices_split[rank][i]],log=True,volume_element=True)

                    return_dict[str(indices_split[rank][i])] = hmf_eval
                    return_dict["ln_M"] = ln_M

                if n_cores > 1:

                    out_q.put(return_dict)

                else:

                    return return_dict

            return_dict = launch_multiprocessing(f_mp,n_cores)

            self.ln_M = return_dict["ln_M"]

            for i in range(0,len(self.redshift_vec)):

                self.hmf_matrix[i] = return_dict[str(i)]

        elif self.cnc_params["hmf_calc"] == "MiraTitan":

            self.ln_M,self.hmf_matrix = self.halo_mass_function.eval_hmf(self.redshift_vec,log=True,volume_element=True)

        t1 = time.time()

        self.t_hmf = t1-t0

        self.time_back = 0.
        self.time_hmf2 = 0.
        self.time_select = 0.
        self.time_mass_range = 0.
        self.t_00 = 0.
        self.t_11 = 0.
        self.t_22 = 0.
        self.t_33 = 0.
        self.t_44 = 0.
        self.t_55 = 0.
        self.t_66 = 0.
        self.t_77 = 0.
        self.t_88 = 0.
        self.t_99 = 0.

    #Computes the cluster abundance across selection observable and redshift

    def get_cluster_abundance(self):

        if self.hmf_matrix is None:

            self.get_hmf()

        self.scal_rel_selection = self.scaling_relations[self.cnc_params["obs_select"]]

        skyfracs = self.scal_rel_selection.skyfracs
        self.n_patches = len(skyfracs)

        n_cores = self.cnc_params["number_cores_abundance"]

        if self.cnc_params["parallelise_type"] == "patch":

            indices_split_patches = np.array_split(np.arange(self.n_patches),n_cores)
            indices_split_redshift = [np.arange(len(self.redshift_vec)) for i in range(0,n_cores)]

        elif self.cnc_params["parallelise_type"] == "redshift":

            indices_split_patches = [np.arange(self.n_patches) for i in range(0,n_cores)]
            indices_split_redshift = np.array_split(np.arange(len(self.redshift_vec)),n_cores)

        self.abundance_tensor = np.zeros((self.n_patches,self.cnc_params["n_z"],self.cnc_params["n_obs_select"]))
        self.n_obs_matrix = np.zeros((self.n_patches,self.cnc_params["n_obs_select"]))
        self.n_tot_vec = np.zeros(self.n_patches)
    #    self.scal_rel_selection.preprecompute_scaling_relation(params=self.scal_rel_params,other_params={"lnM":self.ln_M})

        def f_mp(rank,out_q):

            return_dict = {}

            for i in range(0,len(indices_split_patches[rank])):

                abundance_matrix = np.zeros((self.cnc_params["n_z"],self.cnc_params["n_obs_select"]))
                patch_index = int(indices_split_patches[rank][i])

                for j in range(0,len(indices_split_redshift[rank])):

                    redshift_index = indices_split_redshift[rank][j]

                    if skyfracs[patch_index] < 1e-8:

                        abundance = np.zeros(len(self.ln_M))

                    else:

                        other_params = {"D_A": self.D_A[redshift_index],
                                        "E_z": self.E_z[redshift_index],
                                        "H0": self.cosmology.background_cosmology.H0.value,
                                        "E_z0p6" : self.E_z0p6}

                        dn_dx0 = self.hmf_matrix[redshift_index,:]
                        x0 = self.ln_M

                        t0 = time.time()

                        self.scal_rel_selection.precompute_scaling_relation(params=self.scal_rel_params,
                                                other_params=other_params,
                                                patch_index=patch_index)

                        for k in range(0,self.scal_rel_selection.get_n_layers()):

                            x1 = self.scal_rel_selection.eval_scaling_relation(x0,
                                                         layer=k,
                                                        other_params=other_params,
                                                         patch_index=patch_index)

                            dx1_dx0 = self.scal_rel_selection.eval_derivative_scaling_relation(x0,
                                                              layer=k,patch_index=patch_index,
                                                              scalrel_type_deriv=self.cnc_params["scalrel_type_deriv"])

                            dn_dx1 = dn_dx0/dx1_dx0

                            x1_interp = np.linspace(np.min(x1),np.max(x1),self.cnc_params["n_points"])

                            #if k < 4:

                            dn_dx1 = np.interp(x1_interp,x1,dn_dx1)

                            #elif k == 4:

                            #    dn_dx1 = np.exp(np.interp(np.log(x1_interp),np.log(x1),np.log(dn_dx1)))

                            sigma_scatter = np.sqrt(self.scatter.get_cov(observable1=self.cnc_params["obs_select"],
                                                                         observable2=self.cnc_params["obs_select"],
                                                                         layer=k,patch1=patch_index,patch2=patch_index))

                            if self.cnc_params["apply_obs_cutoff"] == True:

                                cutoff = self.scal_rel_selection.get_cutoff(layer=k)

                                indices = np.where(x1_interp < cutoff)
                                dn_dx1[indices] = 0.

                            #pl.figure()
                            #pl.plot(x1_interp,dn_dx1)
                            #area1 = integrate.simps(dn_dx1,x1_interp)

                            dn_dx1 = convolve_1d(x1_interp,dn_dx1,sigma_scatter,type=self.cnc_params["abundance_integral_type"])

                            #pl.plot(x1_interp,dn_dx1/dn_dx10)
                            #pl.savefig("/home/iz221/cnc/figures/test_convolved/convolved_z_" + str(redshift_index) + "_" + str(k) + ".pdf")
                            #pl.show()

                            #area2 = integrate.simps(dn_dx1,x1_interp)

                            #print(redshift_index,k,area1,area2)


                            x0 = x1_interp
                            dn_dx0 = dn_dx1

                        dn_dx1_interp = np.interp(self.obs_select_vec,x0,dn_dx0)

                        abundance = dn_dx1_interp*4.*np.pi*skyfracs[patch_index] #number counts per patch

                    return_dict[str(patch_index) + "_" + str(redshift_index)] = abundance  #number counts per patch


            if n_cores > 1:

                out_q.put(return_dict)

            else:

                return return_dict

        return_dict = launch_multiprocessing(f_mp,n_cores)

        for i in range(0,self.n_patches):

            for j in range(0,len(self.redshift_vec)):

                self.abundance_tensor[i,j,:] = return_dict[str(i) + "_" + str(j)]

            self.n_obs_matrix[i,:] = integrate.simps(self.abundance_tensor[i,:,:],self.redshift_vec,axis=0)
            self.n_tot_vec[i] = integrate.simps(self.n_obs_matrix[i,:],self.obs_select_vec,axis=0)

        if self.cnc_params["compute_abundance_matrix"] == True:

            self.get_abundance_matrix()

    #Computes the data part of the unbinned likelihood

    def get_log_lik_data(self):

        indices_no_z = self.catalogue.indices_no_z #indices of clusters with no redshift
        indices_obs_select = self.catalogue.indices_obs_select #indices of clusters with redshift and only the selection observable
        indices_other_obs = self.catalogue.indices_other_obs #indices of clusters with redshift, the selection observable, and other observables

        #Computes log lik of data for clusters with no redshift measurement

        log_lik_data = 0.

        for i in range(0,len(indices_no_z)):

            cluster_index = int(indices_no_z[i])
            cluster_patch = int(self.catalogue.catalogue_patch[self.cnc_params["obs_select"]][cluster_index])

            #If all clusters are confirmed objects

            if self.cnc_params["z_bounds"] == False:

                n_obs = self.n_obs_matrix[cluster_patch]
                obs_select = self.catalogue.catalogue[self.cnc_params["obs_select"]][cluster_index]

                if self.cnc_params["non_validated_clusters"] == True:

                    if self.catalogue.catalogue["validated"][cluster_index] < 0.5: #i.e., if "validated" == 0

                            n_obs = n_obs*(1.-self.cnc_params["f_true_validated"]) + self.n_obs_matrix_fd[cluster_patch]

                log_lik_data = log_lik_data + np.log(np.interp(obs_select,self.obs_select_vec,n_obs))

            elif self.cnc_params["z_bounds"] == True:

                if self.catalogue.catalogue["z_bounds"][cluster_index] == False: #repeated, can probably done in a nicer way

                    n_obs = self.n_obs_matrix[cluster_patch]
                    obs_select = self.catalogue.catalogue[self.cnc_params["obs_select"]][cluster_index]

                    if self.cnc_params["non_validated_clusters"] == True:

                        if self.catalogue.catalogue["validated"][cluster_index] < 0.5: #i.e., if "validated" == 0

                            n_obs = n_obs*(1.-self.cnc_params["f_true_validated"]) + self.n_obs_matrix_fd[cluster_patch]

                    log_lik_data = log_lik_data + np.log(np.interp(obs_select,self.obs_select_vec,n_obs))

                elif self.catalogue.catalogue["z_bounds"][cluster_index] == True:

                    lower_z = self.catalogue.catalogue["low_z"][cluster_index]
                    upper_z = self.catalogue.catalogue["up_z"][cluster_index]

                    abundance_matrix = self.abundance_tensor[patch_index,:,:]

                    abundance_interp = interpolate.interp1d(self.redshift_vec,abundance_matrix,axis=0,kind="linear")
                    z_bounded = np.linspace(lower_z,upper_z,self.cnc_params["n_z"])
                    abundance_z_bounds = abundance_interp(z_bounded)
                    n_obs = integrate.simps(abundance_z_bounds,z_bounded,axis=0)

                    if self.cnc_params["non_validated_clusters"] == True:

                        if self.catalogue.catalogue["validated"][cluster_index] < 0.5: #i.e., if "validated" == 0

                            n_obs = n_obs + self.n_obs_matrix_fd[cluster_patch]

                    log_lik_data = log_lik_data + np.log(np.interp(obs_select,self.obs_select_vec,n_obs))


        #Computes log lik of data for clusters with z if there is only the selection observable

        if self.cnc_params["data_lik_from_abundance"] == True:

            indices_unique = self.catalogue.indices_unique
            indices_unique_dict = self.catalogue.indices_unique_dict

            for i in range(0,len(indices_unique)):

                patch_index = int(indices_unique[i])
                abundance_matrix = self.abundance_tensor[patch_index,:,:]

                abundance_interp = interpolate.RegularGridInterpolator((self.redshift_vec,self.obs_select_vec),abundance_matrix)


                indices = indices_unique_dict[str(int(patch_index))]
                z_select = self.catalogue.catalogue["z"][indices_obs_select][indices]
                obs_select = self.catalogue.catalogue[self.cnc_params["obs_select"]][indices_obs_select][indices]

                #If redshift errors are negligible

                z_std_select = self.catalogue.catalogue["z_std"][indices_obs_select][indices]

                if self.cnc_params["z_errors"] == False or all(z_std_select < self.cnc_params["z_error_min"]):

                    lik_clusters = abundance_interp(np.transpose(np.array([z_select,obs_select])))
                    log_lik_clusters = np.sum(np.log(lik_clusters))

                #If redshift errors are not negligible

                if self.cnc_params["z_errors"] == True:

                    if all(z_std_select < self.cnc_params["z_error_min"]):

                        lik_clusters = abundance_interp(np.transpose(np.array([z_select,obs_select])))
                        log_lik_clusters = np.sum(np.log(lik_clusters))

                    else:

                        indices_z_err = np.where(z_std_select >= self.cnc_params["z_error_min"])[0]
                        indices_no_z_err = np.where(z_std_select < self.cnc_params["z_error_min"])[0]

                        lik_clusters_no_z_err = abundance_interp(np.transpose(np.array([z_select[indices_no_z_err],obs_select[indices_no_z_err]])))

                        log_lik_clusters = np.sum(np.log(lik_clusters_no_z_err))

                        for l in range(0,len(indices_z_err)):

                            redshift = z_select[indices_z_err][l]
                            obs_select_cluster = obs_select[indices_z_err][l]
                            z_std = z_std_select[indices_z_err][l]

                            z_error_min = np.max([redshift-self.cnc_params["z_error_sigma_integral_range"]*z_std,self.redshift_vec[0]])
                            z_error_max = np.min([redshift+self.cnc_params["z_error_sigma_integral_range"]*z_std,self.redshift_vec[-1]])
                            z_eval_vec = np.linspace(z_error_min,z_error_max,self.cnc_params["n_z_error_integral"])
                            z_error_likelihood = gaussian_1d(z_eval_vec-redshift,z_std)

                            eval_points = np.zeros((len(z_eval_vec),2))
                            eval_points[:,0] = z_eval_vec
                            eval_points[:,1] = obs_select_cluster

                            abundance_z = abundance_interp(eval_points)

                            lik_cluster = integrate.simps(abundance_z*z_error_likelihood,z_eval_vec)

                            log_lik_clusters = log_lik_clusters + np.log(lik_cluster)

                log_lik_data = log_lik_data + log_lik_clusters

        #Computes log lik of data if there are more observables than the selection observable

        elif self.cnc_params["data_lik_from_abundance"] == False:

            indices_other_obs = np.concatenate((indices_other_obs,indices_obs_select))
            self.indices_bc = indices_other_obs

        if len(indices_other_obs) > 0:

            t0 = time.time()

            n_cores = self.cnc_params["number_cores_data"]
            indices_split = np.array_split(indices_other_obs,n_cores)

            hmf_interp = interpolate.interp1d(self.redshift_vec,self.hmf_matrix,axis=0,kind="linear")
            lnM0 = self.ln_M

            t1 = time.time()

        #    print("Ini",t1-t0)

            def f_mp(rank,out_q):

                return_dict = {}

                log_lik_data_rank = 0.

                for cluster_index in indices_split[rank]:

                    cluster_index = int(cluster_index)
                    redshift = self.catalogue.catalogue["z"][cluster_index]

                    if self.cnc_params["z_errors"] == True and self.catalogue.catalogue["z_std"][cluster_index] > self.cnc_params["z_error_min"]:

                        z_std = self.catalogue.catalogue["z_std"][cluster_index]
                        z_error_min = np.max([redshift-self.cnc_params["z_error_sigma_integral_range"]*z_std,self.redshift_vec[0]])
                        z_error_max = np.min([redshift+self.cnc_params["z_error_sigma_integral_range"]*z_std,self.redshift_vec[-1]])
                        z_eval_vec = np.linspace(z_error_min,z_error_max,self.cnc_params["n_z_error_integral"])
                        z_error_likelihood = gaussian_1d(z_eval_vec-redshift,z_std)

                    else:

                        z_eval_vec = np.array([redshift])

                    lik_cluster_vec = np.zeros(len(z_eval_vec))

                    for redshift_error_id in range(0,len(z_eval_vec)):

                        redshift_eval = z_eval_vec[redshift_error_id]

                        D_A = np.interp(redshift_eval,self.redshift_vec,self.D_A)
                        E_z = np.interp(redshift_eval,self.redshift_vec,self.E_z)
                        D_l_CMB = np.interp(redshift_eval,self.redshift_vec,self.D_l_CMB)
                        rho_c = np.interp(redshift_eval,self.redshift_vec,self.rho_c)

                        other_params = {"D_A": D_A,
                        "E_z": E_z,
                        "H0": self.cosmology.background_cosmology.H0.value,
                        "D_l_CMB":D_l_CMB,
                        "rho_c":rho_c,
                        "D_CMB":self.cosmology.D_CMB,
                        "E_z0p6" : self.E_z0p6,
                        "zc":redshift_eval,
                        "cosmology":self.cosmology}

                        t0 = time.time()

                        halo_mass_function_z = hmf_interp(redshift_eval)

                        t1 = time.time()

                        self.time_hmf2 = self.time_hmf2+t1-t0

                        #Select observables that are available for cluster cluster_index

                        observables_select = self.catalogue.observable_dict[cluster_index]
                        observable_patches = {}

                        for observable_set in observables_select:

                            for observable in observable_set:

                                observable_patches[observable] = self.catalogue.catalogue_patch[observable][cluster_index]

                                layers = np.arange(self.scaling_relations[observable].get_n_layers())

                                self.scaling_relations[observable].precompute_scaling_relation(params=self.scal_rel_params,
                                other_params=other_params,patch_index=observable_patches[observable])

                        t2 = time.time()

                        self.time_select = self.time_select + t2-t1

                        #Estimation of ln M range

                        obs_mass_def = self.cnc_params["obs_select"]

                        lnM = np.linspace(np.log(self.cnc_params["M_min"]/1e14),np.log(self.cnc_params["M_max"]/1e14),self.cnc_params["n_points_data_lik"])

                        x0 = lnM

                        for i in layers:

                            x1 = self.scaling_relations[obs_mass_def].eval_scaling_relation(x0,
                            layer=i,
                            patch_index=int(observable_patches[obs_mass_def]),
                            other_params=other_params)
                            x0 = x1

                        x_obs_massdef = self.catalogue.catalogue[obs_mass_def][cluster_index]
                        i_min = np.argmin(np.abs(x_obs_massdef-x1))
                        lnM_centre = lnM[i_min]
                        x0_centre = lnM_centre

                        x_centre_list = []
                        derivative_list = []

                        for i in layers:

                            if self.cnc_params["scalrel_type_deriv"] == "analytical":

                                x1_centre = self.scaling_relations[obs_mass_def].eval_scaling_relation(x0_centre,layer=i,patch_index=int(observable_patches[obs_mass_def]),other_params=other_params)
                                der = self.scaling_relations[obs_mass_def].eval_derivative_scaling_relation(x0_centre,layer=i,patch_index=int(observable_patches[obs_mass_def]),
                                scalrel_type_deriv=self.cnc_params["scalrel_type_deriv"])

                            elif self.cnc_params["scalrel_type_deriv"] == "numerical":

                                x11 = self.scaling_relations[obs_mass_def].eval_scaling_relation(lnM,layer=i,patch_index=int(observable_patches[obs_mass_def]),other_params=other_params)
                                der = self.scaling_relations[obs_mass_def].eval_derivative_scaling_relation(lnM,layer=i,patch_index=int(observable_patches[obs_mass_def]),
                                scalrel_type_deriv=self.cnc_params["scalrel_type_deriv"])
                                x1_centre = x11[i_min]
                                der = np.interp(x1_centre,x11,der)

                            x_centre_list.append(x1_centre)
                            derivative_list.append(der)
                            x0_centre = x1_centre

                        DlnM = 0.

                        for i in np.flip(layers):

                            sigma = np.sqrt(self.scatter.get_cov(observable1=obs_mass_def,observable2=obs_mass_def,
                            layer=i,patch1=observable_patches[obs_mass_def],patch2=observable_patches[obs_mass_def]))
                            DlnM = np.sqrt(DlnM**2+(1./derivative_list[i]*sigma)**2)

                        sigma_factor = self.cnc_params["sigma_mass_prior"]

                        lnM = np.linspace(lnM_centre-sigma_factor*DlnM,lnM_centre+sigma_factor*DlnM,self.cnc_params["n_points_data_lik"])

                        t3 = time.time()

                        self.time_mass_range = self.time_mass_range + t3-t2

                        #Backpropagate the scatter

                        if self.cnc_params["data_lik_type"] == "backward_convolutional":

                            cpdf_product = 1.

                            for observable_set in observables_select:


                                tt = time.time()

                                x_obs = []

                                for observable in observable_set:

                                    x_obs.append(self.catalogue.catalogue[observable][cluster_index])

                                x_obs = np.array(x_obs)

                                covariance = covariance_matrix(self.scatter,observable_set,
                                observable_patches,layer=layers)

                                n_obs = len(observable_set)

                                tt1 = time.time()

                                self.t_00 = self.t_00 + tt1-tt

                                if len(layers) == 1:

                                    lay = layers[0]

                                    x1 = np.zeros((n_obs,len(lnM)))

                                    for i in range(0,n_obs):

                                        x1[i,:] = self.scaling_relations[observable_set[i]].eval_scaling_relation(lnM,layer=lay,patch_index=int(observable_patches[observable_set[i]]),other_params=other_params)-x_obs[i]

                                    cpdf = stats.multivariate_normal.pdf(np.transpose(x1),cov=covariance.cov[lay])
                                    cpdf = np.interp(lnM0,lnM,cpdf,left=0,right=0)

                                elif len(layers) > 1:

                                    x_list_linear = []
                                    x_list = []
                                    x = lnM

                                    x = np.zeros((n_obs,len(lnM)))

                                    for i in range(0,n_obs):

                                        x[i,:] = lnM


                                    tt2 = time.time()
                                    self.t_11 = self.t_11 + tt2 - tt1

                                    for i in layers[0:-1]:

                                        x1 = np.zeros((n_obs,len(lnM)))
                                        x1_linear = np.zeros((n_obs,len(lnM)))

                                        for j in range(0,n_obs):

                                            x1[j,:] = self.scaling_relations[observable_set[j]].eval_scaling_relation(x[j,:],layer=i,patch_index=int(observable_patches[observable_set[j]]),other_params=other_params)
                                            x1_linear[j,:] = np.linspace(x1[j,0],x1[j,-1],self.cnc_params["n_points_data_lik"])

                                        x_list.append(x1)
                                        x_list_linear.append(x1_linear)
                                        x = x1

                                    tt3 = time.time()
                                    self.t_22 = self.t_22  + tt3 - tt2

                                    for i in np.flip(layers[0:-1]):

                                        lay = i
                                        x_p = x_list_linear[lay]

                                        if lay == layers[-2]:

                                            x1 = np.zeros((n_obs,len(lnM)))

                                            for j in range(0,n_obs):

                                                x_obs_j = x_obs[j]

                                                if isinstance(x_obs_j,(float,int,np.float32)) == True:

                                                    x1[j,:] = self.scaling_relations[observable_set[j]].eval_scaling_relation(x_p[j,:],layer=lay+1,patch_index=int(observable_patches[observable_set[j]]),other_params=other_params)-x_obs_j

                                                else:

                                                    xx = self.scaling_relations[observable_set[j]].eval_scaling_relation(x_p[j,:],layer=lay+1,patch_index=int(observable_patches[observable_set[j]]),other_params=other_params)
                                                    std = self.catalogue.catalogue[observable_set[j] + "_std"][cluster_index]

                                                    rInclude = self.scaling_relations[observable_set[j]].rInclude

                                                    dxwl = xx - x_obs_j[rInclude,None]
                                                    dxwl_std = std[rInclude,None]

                                                    x1[j,:] = np.sqrt(np.sum((dxwl/dxwl_std)**2,axis=0))

                                            tt3 = time.time()

                                            x_mesh = get_mesh(x1)
                                            cpdf = eval_gaussian_nd(x_mesh,cov=covariance.cov[lay+1])

                                        tt4 = time.time()
                                        self.t_33 = self.t_33  + tt4 - tt3

                                        x_p_m = np.zeros((n_obs,len(lnM)))

                                        for j in range(0,n_obs):

                                            x_p_m[j,:] = x_p[j,:] - np.mean(x_p[j,:]) + (x_p[j,1]-x_p[j,0])*0.5

                                        x_p_mesh = get_mesh(x_p_m)

                                        tt4b = time.time()

                                        self.t_44 = self.t_44 + tt4b - tt4

                                        kernel = eval_gaussian_nd(x_p_mesh,cov=covariance.cov[lay])

                                        tt5 = time.time()
                                        self.t_55 = self.t_55  + tt5 - tt4b

                                    #    cpdf = apodise(cpdf)
                                        cpdf = convolve_nd(cpdf,kernel)

                                        tt6 = time.time()
                                        self.t_66 = self.t_66 + tt6 - tt5

                                        cpdf[np.where(cpdf < 0.)] = 0.

                                        if n_obs > 1:

                                            x_mesh_interp = get_mesh(x_list[lay])
                                            shape = x_mesh.shape
                                            x_mesh_interp = np.transpose(x_mesh_interp.reshape(*x_mesh.shape[:-2],-1))

                                            cpdf = interpolate.RegularGridInterpolator(x_p,cpdf,method="linear",fill_value=0.,bounds_error=False)(x_mesh_interp)
                                            #cpdf = interpolate_nd(x_mesh_interp,x_p,cpdf,method="linear")

                                            cpdf = np.transpose(cpdf.reshape(shape[1:]))

                                        else:

                                            cpdf = np.interp(x_list[lay][0,:],x_p[0,:],cpdf)

                                        tt7 = time.time()

                                        self.t_77 = self.t_77 + tt7 - tt6

                                        if lay == layers[0]:

                                            if n_obs > 1:

                                                cpdf = extract_diagonal(cpdf)

                                            cpdf = np.interp(lnM0,lnM,cpdf,left=0,right=0)

                                        else:

                                            cpdf = np.interp(x_list_linear[lay-1],x_list[lay-1],cpdf)

                                        tt8 = time.time()
                                        self.t_88 = self.t_88 + tt8 - tt7

                                cpdf_product = cpdf_product*cpdf

                            patch_select = int(observable_patches[self.cnc_params["obs_select"]])

                            cpdf_product_with_hmf = cpdf_product*halo_mass_function_z*4.*np.pi*self.scal_rel_selection.skyfracs[patch_select]

                            return_dict["cpdf_" + str(cluster_index) + "_" + str(redshift_error_id)] = cpdf_product_with_hmf
                            return_dict["lnm_vec_" + str(cluster_index) + "_" + str(redshift_error_id)] = lnM0

                            tt9 = time.time()

                            lik_cluster_vec[redshift_error_id] = integrate.simps(cpdf_product_with_hmf,lnM0)

                            self.t_99 = self.t_99 + time.time() - tt9

                        elif self.cnc_params["data_lik_type"] == "direct_integral":

                            observable_set = observables_select[0]

                            x_obs = []
                            observable_patches = {}

                            for observable in observable_set:

                                x_obs.append(self.catalogue.catalogue[observable][cluster_index])
                                observable_patches[observable] = self.catalogue.catalogue_patch[observable][cluster_index]

                            patch_select = int(observable_patches[self.cnc_params["obs_select"]])

                            covariance = covariance_matrix(self.scatter,observable_set,
                            observable_patches,layer=layers)

                            x_obs = np.array(x_obs)
                            n_obs = len(x_obs)

                            hmf_cluster = np.interp(lnM,lnM0,halo_mass_function_z)*4.*np.pi*self.scal_rel_selection.skyfracs[patch_select]
                            integrand_m = np.zeros(len(hmf_cluster))

                            if len(layers) > 1:

                                x_list_linear = []
                                x_list = []
                                x_list_fromlinear = []
                                x = np.zeros((n_obs,len(lnM)))

                                for i in range(0,n_obs):

                                    x[i,:] = lnM

                                x1_fromlinear = x

                                for i in layers:

                                    x1 = np.zeros((n_obs,len(lnM)))
                                    x1_linear = np.zeros((n_obs,len(lnM)))
                                    x1_fromlinear = np.zeros((n_obs,len(lnM)))

                                    for j in range(0,n_obs):

                                        x1[j,:] = self.scaling_relations[observable_set[j]].eval_scaling_relation(x[j,:],layer=i,patch_index=int(observable_patches[observable_set[j]]),other_params=other_params)
                                        x1_linear[j,:] = np.linspace(x1[j,0],x1[j,-1],self.cnc_params["n_points_data_lik"])

                                        if i == 1:

                                            x1_fromlinear[j,:] = self.scaling_relations[observable_set[j]].eval_scaling_relation(x_list_linear[0][j,:],layer=i,patch_index=int(observable_patches[observable_set[j]]),other_params=other_params)


                                    x_list.append(x1)
                                    x_list_linear.append(x1_linear)

                                    if i == 1:

                                        x_list_fromlinear.append(x1_fromlinear)

                                    x = x1

                            for layer in layers[0:-1]:

                                x_variables = np.zeros((n_obs,len(lnM)))
                                x_variables_obs = np.zeros((n_obs,len(lnM)))

                                for j in range(0,n_obs):

                                    x_variables[j] = x_list_linear[layer][j]
                                    x_variables_obs[j] = x_list_fromlinear[0][j]-x_obs[j]

                            x_mesh_obs = get_mesh(x_variables_obs)
                            pdf_obs = stats.multivariate_normal.pdf(np.transpose(x_mesh_obs),cov=covariance.cov[1])
                            x_mesh = get_mesh(x_variables)

                            for m in range(0,len(lnM)):

                                x_true = x_list[0][:,m]
                                pdf_int = stats.multivariate_normal.pdf(np.transpose(x_mesh),cov=covariance.cov[0],mean=x_true)
                                integrand = pdf_int*pdf_obs*hmf_cluster[m]

                                for o in range(0,n_obs):

                                    integrand = integrate.simps(integrand,x_variables[o],axis=0)

                                integrand_m[m] = integrand

                            integral = integrate.simps(integrand_m,lnM)
                            lik_cluster_vec[redshift_error_id] = integral


                        t4 = time.time()

                        self.time_back = self.time_back + t4-t3

                    return_dict["z_vec_" + str(cluster_index)] = z_eval_vec

                    if self.cnc_params["z_errors"] == True and self.catalogue.catalogue["z_std"][cluster_index] > self.cnc_params["z_error_min"]:

                        return_dict["z_err_lik_" + str(cluster_index)] = z_error_likelihood

                        lik_cluster = integrate.simps(lik_cluster_vec*z_error_likelihood,z_eval_vec)

                    else:

                        lik_cluster = lik_cluster_vec[0]

                        if self.cnc_params["get_masses"] == True: #only with FFT, not with direct integral

                            cpdf = return_dict["cpdf_" + str(cluster_index) + "_" + str(redshift_error_id)]
                            lnM_vec = return_dict["lnm_vec_" + str(cluster_index) + "_" + str(redshift_error_id)]
                            norm = integrate.simps(cpdf,lnM_vec)
                            lnM_mean = integrate.simps(lnM_vec*cpdf,lnM_vec)/norm
                            lnM_std = np.sqrt(integrate.simps(lnM_vec**2*cpdf,lnM_vec)/norm-lnM_mean**2)

                            return_dict["lnM_mean_" + str(cluster_index)] = lnM_mean
                            return_dict["lnM_std_" + str(cluster_index)] = lnM_std

                    log_lik_data_rank = log_lik_data_rank + np.log(lik_cluster)

                return_dict["lik_" + str(rank)] = log_lik_data_rank

                if n_cores > 1:

                    out_q.put(return_dict)

                else:

                    return return_dict

            return_dict = launch_multiprocessing(f_mp,n_cores)

            for key in range(0,n_cores):

                log_lik_data = log_lik_data + return_dict["lik_" + str(key)]

            self.cpdf_dict = return_dict

            # print("")
            # print("")
            # print("")
            # print("Time hmf2",self.time_hmf2)
            # print("Time select",self.time_select)
            # print("Time mass range",self.time_mass_range)
            # print("Time back",self.time_back)
            # print("T0",self.t_00)
            # print("T1",self.t_11)
            # print("T2",self.t_22)
            # print("T3",self.t_33)
            # print("T4",self.t_44)
            # print("T5",self.t_55)
            # print("T6",self.t_66)
            # print("T7",self.t_77)
            # print("T8",self.t_88)
            # print("T9",self.t_99)
            # print("T sum",self.t_00+self.t_11+self.t_22+self.t_33+self.t_44+self.t_55+self.t_66+self.t_77+self.t_88+self.t_99)

        return log_lik_data

    #Computes the stacked likelihood. Must be called after the unbinned likelihood has been computed.

    def get_log_lik_stacked(self):

        log_lik = 0.

        n_cores = self.cnc_params["number_cores_stacked"]
        cluster_indices = []
        cluster_stack_data = []

        for stacked_data_label in self.stacked_data_labels:

            stacked_cluster_indices = self.catalogue.stacked_data[stacked_data_label]["cluster_index"]

            n_clusters = len(stacked_cluster_indices)

            for i in range(0,n_clusters):

                cluster_indices.append(int(stacked_cluster_indices[i]))
                cluster_stack_data.append(stacked_data_label)

        n_cores = self.cnc_params["number_cores_stacked"]
        indices_split = np.array_split(np.arange(len(cluster_indices)),n_cores)

        def f_mp(rank,out_q):

            return_dict = {}

            for i in range(0,len(indices_split[rank])):

                cluster_index = int(cluster_indices[indices_split[rank][i]])
                stacked_data_label = cluster_stack_data[indices_split[rank][i]]

                stacked_observable = self.catalogue.stacked_data[stacked_data_label]["observable"]

                cluster_patch = int(self.catalogue.catalogue_patch[stacked_observable][cluster_index])
                n_layer = self.scaling_relations[stacked_observable].get_n_layers()

                z_vec = self.cpdf_dict["z_vec_" + str(cluster_index)]

                if len(z_vec) == 1:

                    redshift_error_id = 0
                    redshift_eval = z_vec[0]

                    cpdf = self.cpdf_dict["cpdf_" + str(cluster_index) + "_" + str(redshift_error_id)]
                    lnM_vec = self.cpdf_dict["lnm_vec_" + str(cluster_index) + "_" + str(redshift_error_id)]
                    cpdf = cpdf/integrate.simps(cpdf,lnM_vec)

                    D_A = np.interp(redshift_eval,self.redshift_vec,self.D_A)
                    E_z = np.interp(redshift_eval,self.redshift_vec,self.E_z)
                    D_l_CMB = np.interp(redshift_eval,self.redshift_vec,self.D_l_CMB)
                    rho_c = np.interp(redshift_eval,self.redshift_vec,self.rho_c)

                    other_params = {"D_A": D_A,
                    "E_z": E_z,
                    "H0": self.cosmology.background_cosmology.H0.value,
                    "D_l_CMB":D_l_CMB,
                    "rho_c":rho_c,
                    "D_CMB":self.cosmology.D_CMB,
                    "E_z0p6" : self.E_z0p6,
                    "zc":redshift_eval,
                    "cosmology":self.cosmology}

                    self.scaling_relations[stacked_observable].precompute_scaling_relation(params=self.scal_rel_params,
                    other_params=other_params,patch_index=cluster_patch)

                    obs_mean_vec = self.scaling_relations[stacked_observable].get_mean(lnM_vec,
                    patch_index=cluster_patch,scatter=self.scatter,compute_var=self.cnc_params["compute_stacked_cov"])

                    if self.cnc_params["compute_stacked_cov"] == True:

                        obs_mean_vec,obs_var_vec = obs_mean_vec
                        obs_second_moment_vec = obs_var_vec + obs_mean_vec**2

                    obs_mean = integrate.simps(obs_mean_vec*cpdf,lnM_vec)

                    if self.cnc_params["compute_stacked_cov"] == True:

                        obs_second_moment = integrate.simps(obs_second_moment_vec*cpdf,lnM_vec)
                        obs_var = obs_second_moment - obs_mean**2

                return_dict[stacked_data_label + "_" + str(cluster_index)] = obs_mean

                if self.cnc_params["compute_stacked_cov"] == True:

                    return_dict[stacked_data_label + "_" + str(cluster_index) + "_var"] = obs_var

            if n_cores > 1:

                out_q.put(return_dict)

            else:

                return return_dict

        return_dict = launch_multiprocessing(f_mp,n_cores)

        log_lik = 0.

        self.stacked_model = {}
        self.stacked_variance = {}

        for stacked_data_label in self.stacked_data_labels:

            stacked_obs_vec = self.catalogue.stacked_data[stacked_data_label]["data_vec"] #mean data, not stacked
            stacked_inv_cov = self.catalogue.stacked_data[stacked_data_label]["inv_cov"]
            stacked_cluster_indices = self.catalogue.stacked_data[stacked_data_label]["cluster_index"]
            n_clusters = len(stacked_cluster_indices)

            stacked_model_vec = 0.
            stacked_var_vec = 0.

            for i in range(0,n_clusters):

                stacked_model_vec = stacked_model_vec + return_dict[stacked_data_label + "_" + str(int(stacked_cluster_indices[i]))]

                if self.cnc_params["compute_stacked_cov"] == True:

                    stacked_var_vec = stacked_var_vec + return_dict[stacked_data_label + "_" + str(int(stacked_cluster_indices[i])) + "_var"]

            stacked_model_vec = stacked_model_vec/float(len(stacked_cluster_indices))

            if self.cnc_params["compute_stacked_cov"] == True:

                stacked_var_vec = stacked_var_vec/float(len(stacked_cluster_indices))**2
                stacked_inv_cov = 1./stacked_var_vec

            res = np.array(stacked_obs_vec-stacked_model_vec)
            stacked_inv_cov = np.array(stacked_inv_cov)

            self.stacked_model[stacked_data_label] = stacked_model_vec
            self.stacked_variance[stacked_data_label] = stacked_var_vec

            log_lik = log_lik - 0.5*np.dot(res,np.dot(stacked_inv_cov,res))

        return log_lik

    #Retrieve cluster mean log masses, use only after the unbinned likelihood
    #has been calculated with the backward convolutional approach. Eddington bias included.

    def get_masses(self):

        self.indices_bc
        self.cluster_lnM = np.zeros(len(self.indices_bc))
        self.cluster_lnM_std = np.zeros(len(self.indices_bc))

        for cluster_index in self.indices_bc:

            cluster_index = int(cluster_index)
            self.cluster_lnM[cluster_index] = self.cpdf_dict["lnM_mean_" + str(cluster_index)]
            self.cluster_lnM_std[cluster_index] = self.cpdf_dict["lnM_std_" + str(cluster_index)]


    def get_log_lik_calibration_alt(self):

        n_cores = self.cnc_params["number_cores_stacked"]
        cluster_indices = []
        cluster_observable = []

        self.n_clusters_cal = {}
        self.cluster_indices_cal = {}

        for observable in self.cnc_params["observables_cal_alt"]:

            cal_cluster_indices = np.argwhere(~np.isnan(self.catalogue.catalogue[observable]))[:,0]
            self.cluster_indices_cal[observable] = cal_cluster_indices

            for i in range(0,len(cal_cluster_indices)):

                cluster_indices.append(int(cal_cluster_indices[i]))
                cluster_observable.append(observable)

        n_cores = self.cnc_params["number_cores_stacked"]
        indices_split = np.array_split(np.arange(len(cluster_indices)),n_cores)

        def f_mp(rank,out_q):

            return_dict = {}

            for i in range(0,len(indices_split[rank])):

                cluster_index = int(cluster_indices[indices_split[rank][i]])
                observable = cluster_observable[indices_split[rank][i]]
                x_obs = self.catalogue.catalogue[observable][cluster_index]

                cluster_patch = int(self.catalogue.catalogue_patch[observable][cluster_index])
                n_layer = self.scaling_relations[observable].get_n_layers()

                z_vec = self.cpdf_dict["z_vec_" + str(cluster_index)]

                if len(z_vec) == 1:

                    redshift_error_id = 0
                    redshift_eval = z_vec[0]

                    cpdf = self.cpdf_dict["cpdf_" + str(cluster_index) + "_" + str(redshift_error_id)]
                    lnM_vec = self.cpdf_dict["lnm_vec_" + str(cluster_index) + "_" + str(redshift_error_id)]
                    cpdf = cpdf/integrate.simps(cpdf,lnM_vec)

                    D_A = np.interp(redshift_eval,self.redshift_vec,self.D_A)
                    E_z = np.interp(redshift_eval,self.redshift_vec,self.E_z)
                    D_l_CMB = np.interp(redshift_eval,self.redshift_vec,self.D_l_CMB)
                    rho_c = np.interp(redshift_eval,self.redshift_vec,self.rho_c)

                    other_params = {"D_A": D_A,
                    "E_z": E_z,
                    "H0": self.cosmology.background_cosmology.H0.value,
                    "D_l_CMB":D_l_CMB,
                    "rho_c":rho_c,
                    "D_CMB":self.cosmology.D_CMB,
                    "E_z0p6" : self.E_z0p6,
                    "zc":redshift_eval,
                    "cosmology":self.cosmology}

                    self.scaling_relations[observable].precompute_scaling_relation(params=self.scal_rel_params,
                    other_params=other_params,patch_index=cluster_patch)

                    x0 = lnM_vec
                    dn_dx0 = cpdf

                    for layer in range(0,n_layer-1):

                        x1 = self.scaling_relations[observable].eval_scaling_relation(x0,layer=layer,patch_index=cluster_patch,other_params=other_params)

                        dx1_dx0 = self.scaling_relations[observable].eval_derivative_scaling_relation(x0,
                                                          layer=layer,patch_index=cluster_patch,
                                                          scalrel_type_deriv=self.cnc_params["scalrel_type_deriv"])

                        dn_dx1 = dn_dx0/dx1_dx0
                        x1_interp = np.linspace(np.min(x1),np.max(x1),self.cnc_params["n_points"])
                        dn_dx1 = np.interp(x1_interp,x1,dn_dx1)

                        sigma_scatter = np.sqrt(self.scatter.get_cov(observable1=observable,
                                                                     observable2=observable,
                                                                     layer=layer,patch1=cluster_patch,patch2=cluster_patch))

                        dn_dx1 = convolve_1d(x1_interp,dn_dx1,sigma_scatter)

                        x0 = x1_interp
                        dn_dx0 = dn_dx1

                    cpdf = dn_dx1

                    x1 = self.scaling_relations[observable].eval_scaling_relation(x0,layer=n_layer-1,patch_index=cluster_patch,other_params=other_params)

                    sigma_scatter = np.sqrt(self.scatter.get_cov(observable1=observable,
                                                                 observable2=observable,
                                                                 layer=n_layer-1,patch1=cluster_patch,patch2=cluster_patch))

                    pdf_last = gaussian_1d(x1-x_obs,sigma_scatter)

                    log_lik_cluster = np.log(integrate.simps(pdf_last*cpdf,x0))

                return_dict[observable + "_" + str(cluster_index)] = log_lik_cluster

            if n_cores > 1:

                out_q.put(return_dict)

            else:

                return return_dict

        return_dict = launch_multiprocessing(f_mp,n_cores)

        log_lik = 0.

        for observable in self.cnc_params["observables_cal_alt"]:

            cluster_indices = self.cluster_indices_cal[observable]

            for i in cluster_indices:

                log_lik = log_lik + return_dict[observable + "_" + str(int(i))]

        return log_lik

    #Computes the integrated cluster number counts as a function of redshift, selection
    #observable, and the total number counts

    def get_number_counts_false_detections(self):

        self.n_obs_matrix_fd = np.zeros(self.n_obs_matrix.shape)

        f_false_detection = self.scal_rel_params["f_false_detection"] #ratio of false detections to total detections

        for i in range(0,self.n_obs_matrix_fd.shape[0]):

            [obs_select_fd,pdf_fd] = self.scaling_relations[self.cnc_params["obs_select"]].pdf_false_detection #pdf must be normalised to 1
            self.n_obs_matrix_fd[i,:] = np.interp(self.obs_select_vec,obs_select_fd,pdf_fd)*self.n_tot_vec[i]*f_false_detection/(1.-f_false_detection)

        self.n_tot_vec_fd = self.n_tot_vec*f_false_detection/(1.-f_false_detection)

    def get_number_counts(self):

        if self.abundance_tensor is None:

            self.get_cluster_abundance()

        self.n_obs = np.sum(self.n_obs_matrix,axis=0)
        self.n_tot = np.sum(self.n_tot_vec)

        print("Total clusters",self.n_tot)

        if self.cnc_params["non_validated_clusters"] == True:

            self.get_number_counts_false_detections()

            self.n_obs_fd = np.sum(self.n_obs_matrix_fd,axis=0)
            self.n_tot_fd = np.sum(self.n_tot_vec_fd,axis=0)

    def get_log_lik_extreme_value(self,obs_max=None):

        if self.n_tot is None:

            self.get_number_counts()

        n_obs = self.n_obs

        if obs_max is None:

            obs_max = self.catalogue.obs_select_max

        if self.cnc_params["non_validated_clusters"] == True:

            n_obs = n_obs + self.n_obs_fd

        obs_select_vec_interp = np.linspace(obs_max,self.cnc_params["obs_select_max"],100)
        n_interp = np.interp(obs_select_vec_interp,self.obs_select_vec,n_obs)
        n_theory = integrate.simps(n_interp,obs_select_vec_interp)

        log_lik = - n_theory

        return log_lik

    #Evaluates some extreme value quantities. Execute after get_log_lik_extreme_value

    def eval_extreme_value_quantities(self):

        self.log_lik_ev_eval = np.zeros(len(self.obs_select_vec))

        for i in range(0,len(self.obs_select_vec)):

            self.log_lik_ev_eval[i] = self.get_log_lik_extreme_value(obs_max=self.obs_select_vec[i])

        self.lik_ev_eval = np.exp(self.log_lik_ev_eval)

        self.obs_select_max_pdf = np.gradient(self.lik_ev_eval,self.obs_select_vec)
        self.obs_select_max_mean = integrate.simps(self.obs_select_max_pdf*self.obs_select_vec,self.obs_select_vec)
        self.obs_select_max_std = np.sqrt(integrate.simps(self.obs_select_max_pdf*(self.obs_select_vec-self.obs_select_max_mean)**2,self.obs_select_vec))

    #Reurns log likelihood, with priors


    def get_log_lik(self):

        t0 = time.time()

        log_lik = 0.

        if self.cnc_params["priors"] == True:

            log_lik = log_lik + self.priors.eval_priors(self.cosmo_params,self.scal_rel_params)

        if self.cnc_params["likelihood_type"] == "unbinned":

            log_lik = log_lik + self.get_log_lik_unbinned()

        elif self.cnc_params["likelihood_type"] == "binned":

            log_lik = log_lik + self.get_log_lik_binned()

        elif self.cnc_params["likelihood_type"] == "extreme_value":

            log_lik = log_lik + self.get_log_lik_extreme_value()

        self.t_total = time.time()-t0

        print("Time",self.t_total)

        print("log_lik",log_lik)

        if np.isnan(log_lik) == True:

            log_lik = -np.inf

        self.log_lik = log_lik

        return log_lik

    #Computes the unbinned log likelihood

    def get_log_lik_unbinned(self):

        t0 = time.time()

        if self.hmf_matrix is None:

            self.get_hmf()

        t1 = time.time()

        self.time_hmf = t1-t0

        if self.n_tot is None:

            self.get_number_counts()

        #Abundance term

        n_tot = self.n_tot

        if self.cnc_params["non_validated_clusters"] == True:

            n_tot = n_tot + self.n_tot_fd

        log_lik = -n_tot

        if self.cnc_params["non_validated_clusters"] == True:

            if self.catalogue.n_val > 0.5:

                log_lik = log_lik + self.catalogue.n_val*np.log(self.cnc_params["f_true_validated"])

        t2 = time.time()

        self.t_abundance = t2-t1

        #Cluster data term

        log_lik = log_lik + self.get_log_lik_data()

        self.t_data = time.time()-t2

        #Stacked_term

        if self.cnc_params["stacked_likelihood"] == True:

            log_lik = log_lik + self.get_log_lik_stacked()

        if self.cnc_params["likelihood_cal_alt"] == True:

            log_lik = log_lik + self.get_log_lik_calibration_alt()

        return log_lik

    def get_abundance_matrix(self):

        self.abundance_matrix = np.sum(self.abundance_tensor,axis=0)
        self.n_z = integrate.simps(self.abundance_matrix,self.obs_select_vec)

    #Computes the binned log likelihood

    def get_log_lik_binned(self):

        if self.n_tot is None:

            self.get_number_counts()

        log_lik = 0.

        if self.cnc_params["binned_lik_type"] == "z_and_obs_select":

            if self.abundance_matrix is None:

                self.get_abundance_matrix()

            self.n_binned = np.zeros((len(self.cnc_params["bins_edges_z"])-1,len(self.cnc_params["bins_edges_obs_select"])-1))
            self.n_binned_obs = np.zeros((len(self.cnc_params["bins_edges_z"])-1,len(self.cnc_params["bins_edges_obs_select"])-1))

            self.bins_centres_z = (self.cnc_params["bins_edges_z"][1:] + self.cnc_params["bins_edges_z"][0:-1])*0.5
            self.bins_centres_obs = (self.cnc_params["bins_edges_obs_select"][1:] + self.cnc_params["bins_edges_obs_select"][0:-1])*0.5

            n_bins_redshift = int(len(self.redshift_vec)/(len(self.bins_centres_z)-1))
            n_bins_obs_select = int(len(self.obs_select_vec)/(len(self.bins_centres_obs)-1))

            for i in range(0,len(self.cnc_params["bins_edges_z"])-1):

                for j in range(0,len(self.cnc_params["bins_edges_obs_select"])-1):

                    redshift_vec_interp = np.linspace(self.cnc_params["bins_edges_z"][i],self.cnc_params["bins_edges_z"][i+1],n_bins_redshift)
                    obs_select_vec_interp = np.linspace(self.cnc_params["bins_edges_obs_select"][j],self.cnc_params["bins_edges_obs_select"][j+1],n_bins_obs_select)
                    X,Y = np.meshgrid(redshift_vec_interp,obs_select_vec_interp)

                    abundance_matrix_interp = interpolate.RegularGridInterpolator((self.redshift_vec,self.obs_select_vec),self.abundance_matrix)((X,Y))

                    n_theory = integrate.simps(integrate.simps(abundance_matrix_interp,redshift_vec_interp),obs_select_vec_interp)
                    n_observed = self.catalogue.number_counts[i,j]

                    self.n_binned[i,j] = n_theory
                    self.n_binned_obs[i,j] = n_observed

                    log_lik = log_lik - n_theory + n_observed*np.log(n_theory)

        elif self.cnc_params["binned_lik_type"] == "obs_select":

            self.n_binned = np.zeros(len(self.cnc_params["bins_edges_obs_select"])-1)
            self.n_binned_obs = np.zeros(len(self.cnc_params["bins_edges_obs_select"])-1)
            self.bins_centres = (self.cnc_params["bins_edges_obs_select"][1:] + self.cnc_params["bins_edges_obs_select"][0:-1])*0.5
            n_bins_obs_select = int(len(self.obs_select_vec)/(len(self.bins_centres)-1))

            print("n int",n_bins_obs_select)

            for i in range(0,len(self.cnc_params["bins_edges_obs_select"])-1):

                n_obs = self.n_obs

                if self.cnc_params["non_validated_clusters"] == True:

                    n_obs = n_obs + self.n_obs_fd

                obs_select_vec_interp = np.linspace(self.cnc_params["bins_edges_obs_select"][i],self.cnc_params["bins_edges_obs_select"][i+1],n_bins_obs_select)
                n_interp = np.interp(obs_select_vec_interp,self.obs_select_vec,n_obs)

                n_theory = integrate.simps(n_interp,obs_select_vec_interp)
                n_observed = self.catalogue.number_counts[i]
                self.n_binned_obs[i] = n_observed
                self.n_binned[i] = n_theory

                log_lik = log_lik - n_theory + n_observed*np.log(n_theory)

        elif self.cnc_params["binned_lik_type"] == "z":

            if self.abundance_matrix is None:

                self.get_abundance_matrix()

            self.n_binned = np.zeros(len(self.cnc_params["bins_edges_z"])-1)
            self.n_binned_obs = np.zeros(len(self.cnc_params["bins_edges_z"])-1)
            self.bins_centres = (self.cnc_params["bins_edges_z"][1:] + self.cnc_params["bins_edges_z"][0:-1])*0.5
            n_bins_redshift = int(len(self.redshift_vec)/(len(self.bins_centres)-1))
            print("n int",n_bins_redshift)

            for i in range(0,len(self.cnc_params["bins_edges_z"])-1):

                redshift_vec_interp = np.linspace(self.cnc_params["bins_edges_z"][i],self.cnc_params["bins_edges_z"][i+1],n_bins_redshift)
                n_interp = np.interp(redshift_vec_interp,self.redshift_vec,self.n_z)

                n_theory = integrate.simps(n_interp,redshift_vec_interp)
                n_observed = self.catalogue.number_counts[i]
                self.n_binned_obs[i] = n_observed
                self.n_binned[i] = n_theory

                log_lik = log_lik - n_theory + n_observed*np.log(n_theory)

        return log_lik

    def get_c_statistic(self):

        if self.n_binned is None:

            self.get_log_lik_binned()

        n_binned_mean = self.n_binned.flatten()
        n_binned_obs = self.n_binned_obs.flatten()

        self.C,self.C_mean,self.C_std = get_cash_statistic(n_binned_obs,n_binned_mean)

        return (self.C,self.C_mean,self.C_std)
