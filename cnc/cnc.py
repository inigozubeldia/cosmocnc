import numpy as np
import pylab as pl
from multiprocessing import Pool
from .cosmo import *
from .hmf import *
from .sr import *
from .cat import *
from .params import *
import scipy.signal as signal
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.stats as stats
import time
import functools
import copy
import multiprocess as mp
import functools

class cluster_number_counts:

    def __init__(self,cnc_params=None):

        if cnc_params is None:

            cnc_params = cnc_params_default

        self.cnc_params = cnc_params
        self.cosmo_params = cosmo_params_default
        self.scal_rel_params = scaling_relation_params_default

        self.abundance_matrix = None
        self.n_obs_matrix = None
        self.hmf_matrix = None
        self.n_tot = None

        self.hmf_extra_params = {}

    #Loads data (catalogue and scaling relation data)

    def initialise(self):

        self.scaling_relations = {}

        for observable_set in self.cnc_params["observables"]:

            for observable in observable_set:

                self.scaling_relations[observable] = scaling_relations(observable=observable)
                self.scaling_relations[observable].initialise_scaling_relation()

        self.cosmology = cosmology_model(cosmo_params=self.cosmo_params,
        power_spectrum_type=self.cnc_params["power_spectrum_type"],amplitude_parameter=self.cnc_params["cosmo_amplitude_parameter"])

        self.catalogue = cluster_catalogue(catalogue_name=self.cnc_params["cluster_catalogue"],precompute_cnc_quantities=True,
        bins_obs_select_edges=self.cnc_params["bins_edges_obs_select"],bins_z_edges=self.cnc_params["bins_edges_z"],
        observables=self.cnc_params["observables"],obs_select=self.cnc_params["obs_select"])

        self.scatter = scatter(params=self.scal_rel_params)
        self.priors = priors(prior_params={"cosmology":self.cosmology,"theta_mc_prior":self.cnc_params["theta_mc_prior"]})

        if self.cnc_params["hmf_calc"] == "MiraTitan":

            t00 = time.time()

            import MiraTitanHMFemulator

            self.MT_emulator = MiraTitanHMFemulator.Emulator()
            self.hmf_extra_params["emulator"] = self.MT_emulator

    #Updates parameter values (cosmological and scaling relation)

    def reinitialise(self):

        self.abundance_matrix = None
        self.n_obs_matrix = None
        self.hmf_matrix = None
        self.n_tot = None

    def update_params(self,cosmo_params,scal_rel_params):

        self.cosmo_params = cosmo_params
        self.cosmology.update_cosmology(cosmo_params)
        self.scal_rel_params = {}

        for key in scal_rel_params.keys():

            self.scal_rel_params[key] = scal_rel_params[key]

        self.scatter = scatter(params=self.scal_rel_params)

        self.abundance_matrix = None
        self.n_obs_matrix = None
        self.hmf_matrix = None
        self.n_tot = None

    #Computes the hmf as a function of redshift

    def get_hmf(self):

        self.const = constants()

        #Define redshift and observable ranges

        self.redshift_vec = np.linspace(self.cnc_params["z_min"],self.cnc_params["z_max"],self.cnc_params["n_z"])
        self.obs_select_vec = np.linspace(self.cnc_params["obs_select_min"],self.cnc_params["obs_select_max"],self.cnc_params["n_obs_select"])

        #Evaluate some useful quantities (to be passed potentially to scaling relations)

        self.D_A = self.cosmology.background_cosmology.angular_diameter_distance(self.redshift_vec).value
        self.E_z = self.cosmology.background_cosmology.H(self.redshift_vec).value/(self.cosmology.cosmo_params["h"]*100.)
        self.D_l_CMB = self.cosmology.background_cosmology.angular_diameter_distance_z1z2(self.redshift_vec,self.cosmology.z_CMB).value
        self.rho_c = self.cosmology.background_cosmology.critical_density(self.redshift_vec).value*1000.*self.const.mpc**3/self.const.solar

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

    #Computes the cluster abundance across selection observable and redshift

    def get_cluster_abundance(self):

        if self.hmf_matrix is None:

            self.get_hmf()

        t0 = time.time()

        self.scal_rel_selection = self.scaling_relations[self.cnc_params["obs_select"]]
        self.n_patches = len(self.scal_rel_selection.skyfracs)

        if self.cnc_params["parallelise_type"] == "patch":

            indices_split_patches = np.array_split(np.arange(self.n_patches),self.cnc_params["number_cores"])
            indices_split_redshift = [np.arange(len(self.redshift_vec)) for i in range(0,self.cnc_params["number_cores"])]

        elif self.cnc_params["parallelise_type"] == "redshift":

            indices_split_patches = [np.arange(self.n_patches) for i in range(0,self.cnc_params["number_cores"])]
            indices_split_redshift = np.array_split(np.arange(len(self.redshift_vec)),self.cnc_params["number_cores"])

        self.abundance_tensor = np.zeros((self.n_patches,self.cnc_params["n_z"],self.cnc_params["n_obs_select"]))
        self.n_obs_matrix = np.zeros((self.n_patches,self.cnc_params["n_obs_select"]))

        n_cores = self.cnc_params["number_cores"]

        def f_mp(rank,out_q):

            return_dict = {}

            for i in range(0,len(indices_split_patches[rank])):

                abundance_matrix = np.zeros((self.cnc_params["n_z"],self.cnc_params["n_obs_select"]))

                for j in range(0,len(indices_split_redshift[rank])):

                    patch_index = int(indices_split_patches[rank][i])
                    redshift_index = indices_split_redshift[rank][j]

                    other_params = {"D_A": self.D_A[redshift_index],"E_z": self.E_z[redshift_index],"H0": self.cosmology.background_cosmology.H0.value}

                    dn_dx0 = self.hmf_matrix[redshift_index,:]
                    x0 = self.ln_M

                    for k in range(0,self.scal_rel_selection.get_n_layers()):

                        self.scal_rel_selection.precompute_scaling_relation(params=self.scal_rel_params,
                        other_params=other_params,layer=k,patch_index=patch_index)

                        x1 = self.scal_rel_selection.eval_scaling_relation(x0,
                        layer=k,patch_index=patch_index)

                        dx1_dx0 = self.scal_rel_selection.eval_derivative_scaling_relation(x0,
                        layer=k,patch_index=patch_index,
                        scalrel_type_deriv=self.cnc_params["scalrel_type_deriv"])

                        dn_dx1 = dn_dx0/dx1_dx0
                        x1_interp = np.linspace(np.min(x1),np.max(x1),self.cnc_params["n_points"])
                        dn_dx1 = np.interp(x1_interp,x1,dn_dx1)

                        sigma_scatter = np.sqrt(self.scatter.get_cov(observable1=self.cnc_params["obs_select"],
                        observable2=self.cnc_params["obs_select"],
                        layer=k,patch1=patch_index,patch2=patch_index))

                        dn_dx1 = convolve_1d(x1_interp,dn_dx1,sigma_scatter)

                        x0 = x1_interp
                        dn_dx0 = dn_dx1

                    dn_dx1_interp = np.interp(self.obs_select_vec,x0,dn_dx0)
                    abundance = dn_dx1_interp*4.*np.pi*self.scal_rel_selection.skyfracs[patch_index] #number counts per patch

                    return_dict[str(patch_index) + "_" + str(redshift_index)] = abundance  #number counts per patch

                n_obs = integrate.simps(abundance_matrix,self.redshift_vec,axis=0)
                return_dict[str(patch_index) + "_n_obs"] = n_obs

                if n_cores > 1:

                    out_q.put(return_dict)

                else:

                    return return_dict

        return_dict = launch_multiprocessing(f_mp,n_cores)

        for i in range(0,self.n_patches):

            self.n_obs_matrix[i,:] = return_dict[str(i) + "_n_obs"]

            for j in range(0,len(self.redshift_vec)):

                self.abundance_tensor[i,j,:] = return_dict[str(i) + "_" + str(j)]

        t1 = time.time()

        self.t_abundance = t1-t0

        self.abundance_matrix = np.sum(self.abundance_tensor,axis=0)

    #Computes the data (or "mass calibration") part of the unbinned likelihood

    def get_log_lik_data(self):

        indices_no_z = self.catalogue.indices_no_z #indices of clusters with no redshift
        indices_obs_select = self.catalogue.indices_obs_select #indices of clusters with redshift and only the selection observable
        indices_other_obs = self.catalogue.indices_other_obs #indices of clsuters with redshift, the selection observable, and other observables

        #Computes log lik of data for clusters with missing z

        log_lik_data = 0.

        for i in range(0,len(indices_no_z)):

            log_lik_data = log_lik_data + np.log(np.interp(self.catalogue.catalogue[self.cnc_params["obs_select"]][int(indices_no_z[i])],
            self.obs_select_vec,self.n_obs_matrix[self.catalogue.catalogue_patch[self.cnc_params["obs_select"]][int(indices_no_z[i])]]))

        #Computes log lik of data for clusters with z if there's only the selection observable

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

                lik_clusters = abundance_interp(np.transpose(np.array([z_select,obs_select])))

                log_lik_data = log_lik_data + np.sum(np.log(lik_clusters))

        #Computes log lik of data if there are more observables than the selection observable

        elif self.cnc_params["data_lik_from_abundance"] == False:

            indices_other_obs = np.concatenate((indices_other_obs,indices_obs_select))

        if len(indices_other_obs) > 0:

            indices_split = np.array_split(indices_other_obs,self.cnc_params["number_cores"])

            hmf_interp = interpolate.interp1d(self.redshift_vec,self.hmf_matrix,axis=0)
            lnM0 = self.ln_M

            n_cores = self.cnc_params["number_cores"]

            def f_mp(rank,out_q):

                return_dict = {}

                t000 = time.time()

                log_lik_data_rank = 0.

                for cluster_index in indices_split[rank]:

                    t00 = time.time()

                    redshift = self.catalogue.catalogue["z"][cluster_index]
                    cluster_index = int(cluster_index)

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

                        cpdf_product = np.ones(len(lnM0))

                        t0 = time.time()

                        D_A = np.interp(redshift_eval,self.redshift_vec,self.D_A)
                        E_z = np.interp(redshift_eval,self.redshift_vec,self.E_z)
                        D_l_CMB = np.interp(redshift_eval,self.redshift_vec,self.D_l_CMB)
                        rho_c = np.interp(redshift_eval,self.redshift_vec,self.rho_c)

                        other_params = {"D_A": D_A,"E_z": E_z,
                        "H0": self.cosmology.background_cosmology.H0.value,
                        "D_l_CMB":D_l_CMB,"rho_c":rho_c,"D_CMB":self.cosmology.D_CMB}

                        halo_mass_function_z = hmf_interp(redshift_eval)

                        #select observables that are available for cluster cluster_index

                        observables_select = self.catalogue.observable_dict[cluster_index]
                        observable_patches = {}

                        for observable_set in observables_select:

                            for observable in observable_set:

                                observable_patches[observable] = self.catalogue.catalogue_patch[observable][cluster_index]

                                layers = np.arange(self.scaling_relations[observable].get_n_layers())

                                for layer in layers:

                                    self.scaling_relations[observable].precompute_scaling_relation(params=self.scal_rel_params,
                                    other_params=other_params,layer=layer,patch_index=observable_patches[observable])

                        for observable_set in observables_select:

                            x_obs = []

                            for observable in observable_set:

                                x_obs.append(self.catalogue.catalogue[observable][cluster_index])

                            x_obs = np.array(x_obs)

                            covariance = covariance_matrix(self.scatter,observable_set,
                            observable_patches,layer=layers)

                            #Estimation of ln M range

                            obs_mass_def = list(set(self.cnc_params["observables_mass_estimation"]).intersection(observable_set))[0]

                            lnM = np.linspace(np.log(self.cnc_params["M_min"]/1e14),np.log(self.cnc_params["M_max"]/1e14),self.cnc_params["n_points_data_lik"])

                            x0 = lnM

                            for i in layers:

                                x1 = self.scaling_relations[obs_mass_def].eval_scaling_relation(x0,layer=i)
                                x0 = x1

                            i_min = np.argmin(np.abs(x_obs[0]-x1))
                            lnM_centre = lnM[i_min]
                            x0_centre = lnM_centre

                            x_centre_list = []
                            derivative_list = []

                            for i in layers:

                                if self.cnc_params["scalrel_type_deriv"] == "analytical":

                                    x1_centre = self.scaling_relations[obs_mass_def].eval_scaling_relation(x0_centre,layer=i)
                                    der = self.scaling_relations[obs_mass_def].eval_derivative_scaling_relation(x0_centre,layer=i,
                                    scalrel_type_deriv=self.cnc_params["scalrel_type_deriv"])

                                elif self.cnc_params["scalrel_type_deriv"] == "numerical":

                                    x11 = self.scaling_relations[obs_mass_def].eval_scaling_relation(lnM,layer=i)
                                    der = self.scaling_relations[obs_mass_def].eval_derivative_scaling_relation(lnM,layer=i,
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

                            #Backpropagate the scatter

                            n_obs = len(observable_set)

                            if len(layers) == 1:

                                lay = layers[0]

                                x1 = np.zeros((n_obs,len(lnM)))

                                for i in range(0,n_obs):

                                    x1[i,:] = self.scaling_relations[observable_set[i]].eval_scaling_relation(lnM,layer=lay)-x_obs[i]

                                cpdf = stats.multivariate_normal.pdf(np.transpose(x1),cov=covariance.cov[lay])
                                cpdf = np.interp(lnM0,lnM,cpdf,left=0,right=0)

                            elif len(layers) > 1:

                                x_list_linear = []
                                x_list = []
                                x = lnM

                                x = np.zeros((n_obs,len(lnM)))

                                for i in range(0,n_obs):

                                    x[i,:] = lnM

                                for i in layers[0:-1]:

                                    x1 = np.zeros((n_obs,len(lnM)))
                                    x1_linear = np.zeros((n_obs,len(lnM)))

                                    for j in range(0,n_obs):

                                        x1[j,:] = self.scaling_relations[observable_set[j]].eval_scaling_relation(x[j,:],layer=i)
                                        x1_linear[j,:] = np.linspace(x1[j,0],x1[j,-1],self.cnc_params["n_points_data_lik"])

                                    x_list.append(x1)
                                    x_list_linear.append(x1_linear)
                                    x = x1

                                for i in np.flip(layers[0:-1]):

                                    lay = i
                                    x_p = x_list_linear[lay]

                                    if lay == layers[-2]:

                                        x1 = np.zeros((n_obs,len(lnM)))

                                        for j in range(0,n_obs):

                                            x1[j,:] = self.scaling_relations[observable_set[j]].eval_scaling_relation(x_p[j,:],layer=lay+1)-x_obs[j]

                                        x_mesh = get_mesh(x1)
                                        cpdf = eval_gaussian_nd(x_mesh,cov=covariance.cov[lay+1])

                                    x_p_m = np.zeros((n_obs,len(lnM)))

                                    for j in range(0,n_obs):

                                        x_p_m[j,:] = x_p[j,:] - np.mean(x_p[j,:])

                                    x_p_mesh = get_mesh(x_p_m)

                                    kernel = eval_gaussian_nd(x_p_mesh,cov=covariance.cov[lay])

                                    cpdf = apodise(cpdf)
                                    cpdf = signal.convolve(cpdf,kernel,mode="same",method="fft")/np.sum(kernel)
                                    cpdf[np.where(cpdf < 0.)] = 0.

                                    x_mesh_interp = get_mesh(x_list[lay])
                                    shape = x_mesh.shape
                                    x_mesh_interp = np.transpose(x_mesh_interp.reshape(*x_mesh.shape[:-2],-1))

                                    cpdf = interpolate.RegularGridInterpolator(x_p,cpdf,method="linear")(x_mesh_interp)
                                    cpdf = cpdf.reshape(shape[1:])

                                    if lay == layers[0]:

                                        if n_obs > 1:

                                            cpdf = np.diag(cpdf)

                                        cpdf = np.interp(lnM0,lnM,cpdf,left=0,right=0)

                                    else:

                                        cpdf = np.interp(x_list_linear[lay-1],x_list[lay-1],cpdf)

                            cpdf_product = cpdf_product*cpdf

                        patch_select = int(observable_patches[self.cnc_params["obs_select"]])
                        lik_cluster_vec[redshift_error_id] = integrate.simps(cpdf_product*halo_mass_function_z,lnM0)*4.*np.pi*self.scal_rel_selection.skyfracs[patch_select]

                    if self.cnc_params["z_errors"] == True and self.catalogue.catalogue["z_std"][cluster_index] > self.cnc_params["z_error_min"]:

                        lik_cluster = integrate.simps(lik_cluster_vec*z_error_likelihood,z_eval_vec)

                    else:

                        lik_cluster = lik_cluster_vec[0]

                    log_lik_data_rank = log_lik_data_rank + np.log(lik_cluster)

                return_dict[rank] = log_lik_data_rank

                if n_cores > 1:

                    out_q.put(return_dict)

                else:

                    return return_dict

            return_dict = launch_multiprocessing(f_mp,n_cores)

            for key in return_dict:

                log_lik_data = log_lik_data + return_dict[key]

        #    return_dict = None

        return log_lik_data

    #Computes the integrated cluster number counts as a function of redshift, selection
    #observable, and the total number counts

    def get_number_counts(self):

        if self.abundance_matrix is None:

            self.get_cluster_abundance()

        self.n_z = integrate.simps(self.abundance_matrix,self.obs_select_vec)
        self.n_obs = integrate.simps(np.transpose(self.abundance_matrix),self.redshift_vec)
        self.n_tot = integrate.simps(self.n_obs,self.obs_select_vec)

    def get_log_lik_extreme_value(self):

        if self.n_tot is None:

            self.get_number_counts()

        obs_max = self.catalogue.obs_select_max

        self.n_obs = integrate.simps(np.transpose(self.abundance_matrix),self.redshift_vec)

        obs_select_vec_interp = np.linspace(obs_max,self.cnc_params["obs_select_max"],100)
        n_interp = np.interp(obs_select_vec_interp,self.obs_select_vec,self.n_obs)
        n_theory = integrate.simps(n_interp,obs_select_vec_interp)

        log_lik = - n_theory

        return log_lik


    #Reurns log likelihood, with priors

    def get_log_lik(self):

        t0 = time.time()

        if self.cnc_params["priors"] == True:

            log_lik = self.priors.eval_priors(self.cosmo_params,self.scal_rel_params)

        elif self.cnc_params["priors"] == False:

            log_lik = 0.

        if np.isinf(log_lik) == False:

            if self.cnc_params["likelihood_type"] == "unbinned":

                log_lik = log_lik + self.get_log_lik_unbinned()

            elif self.cnc_params["likelihood_type"] == "binned":

                log_lik = log_lik + self.get_log_lik_binned()

            elif self.cnc_params["likelihood_type"] == "extreme_value":

                log_lik = log_lik + self.get_log_lik_extreme_value()

        self.t_total = time.time()-t0

        print("T total",time.time()-t0)#self.t_total)

        if np.isnan(log_lik) == True:

            log_lik = -np.inf

        return log_lik

    #Computes the unbinned log likelihood

    def get_log_lik_unbinned(self):

        t0 = time.time()

        if self.n_tot is None:

            self.get_number_counts()

        #Poisson term

        log_lik = -self.n_tot

        t1 = time.time()

        self.t_poisson = t1-t0

        #Cluster data term

        log_lik = log_lik + self.get_log_lik_data()

        self.t_data = time.time()-t1

        return log_lik

    #Computes the binned log likelihood

    def get_log_lik_binned(self):

        if self.n_tot is None:

            self.get_number_counts()

        log_lik = 0.

        if self.cnc_params["binned_lik_type"] == "z_and_obs_select":

            for i in range(0,len(self.cnc_params["bins_edges_z"])-1):

                for j in range(0,len(self.cnc_params["bins_edges_obs_select"])-1):

                    redshift_vec_interp = np.linspace(self.cnc_params["bins_edges_z"][i],self.cnc_params["bins_edges_z"][i+1],10)
                    obs_select_vec_interp = np.linspace(self.cnc_params["bins_edges_obs_select"][j],self.cnc_params["bins_edges_obs_select"][j+1],100)

                    abundance_matrix_interp = interpolate.RegularGridInterpolator((self.redshift_vec,self.obs_select_vec),self.abundance_matrix)
                    X,Y = np.meshgrid(redshift_vec_interp,obs_select_vec_interp)
                    abundance_matrix_interp = abundance_matrix_interp((X,Y))

                    n_theory = integrate.simps(integrate.simps(abundance_matrix_interp,redshift_vec_interp),obs_select_vec_interp)
                    n_observed = self.catalogue.number_counts[i,j]

                    log_lik = log_lik - n_theory + n_observed*np.log(n_theory)

        elif self.cnc_params["binned_lik_type"] == "obs_select":

            self.n_obs = integrate.simps(np.transpose(self.abundance_matrix),self.redshift_vec)
            self.n_obs_binned = np.zeros(len(self.cnc_params["bins_edges_obs_select"])-1)
            self.n_obs_binned_obs = np.zeros(len(self.cnc_params["bins_edges_obs_select"])-1)

            for i in range(0,len(self.cnc_params["bins_edges_obs_select"])-1):

                obs_select_vec_interp = np.linspace(self.cnc_params["bins_edges_obs_select"][i],self.cnc_params["bins_edges_obs_select"][i+1],100)
                n_interp = np.interp(obs_select_vec_interp,self.obs_select_vec,self.n_obs)

                n_theory = integrate.simps(n_interp,obs_select_vec_interp)
                n_observed = np.sum(self.catalogue.number_counts[:,i])
                self.n_obs_binned_obs[i] = n_observed
                self.n_obs_binned[i] = n_theory

                log_lik = log_lik - n_theory + n_observed*np.log(n_theory)

        elif self.cnc_params["binned_lik_type"] == "z":

            self.n_z = integrate.simps(self.abundance_matrix,self.obs_select_vec)
            self.n_z_binned = np.zeros(len(self.cnc_params["bins_edges_z"])-1)
            self.n_z_binned_obs = np.zeros(len(self.cnc_params["bins_edges_z"])-1)

            for i in range(0,len(self.cnc_params["bins_edges_z"])-1):

                redshift_vec_interp = np.linspace(self.cnc_params["bins_edges_z"][i],self.cnc_params["bins_edges_z"][i+1],10)
                n_interp = np.interp(redshift_vec_interp,self.redshift_vec,self.n_z)

                n_theory = integrate.simps(n_interp,redshift_vec_interp)
                n_observed = np.sum(self.catalogue.number_counts[i,:])
                self.n_z_binned_obs[i] = n_observed
                self.n_z_binned[i] = n_theory

                log_lik = log_lik - n_theory + n_observed*np.log(n_theory)

        return log_lik


class covariance_matrix:

    def __init__(self,scatter,observables,observable_patches,layer=[0,1]):#

        self.layer = layer
        self.cov = []
        self.inv_cov = []

        for k in range(0,len(self.layer)):

            cov_matrix = np.zeros((len(observables),len(observables)))

            for i in range(0,len(observables)):

                for j in range(0,len(observables)):

                    cov_matrix[i,j] = scatter.get_cov(observable1=observables[i],
                    observable2=observables[j],patch1=observable_patches[observables[i]],
                    patch2=observable_patches[observables[j]],layer=self.layer[k])

            self.cov.append(cov_matrix)
            self.inv_cov.append(np.linalg.inv(cov_matrix))

def launch_multiprocessing_old(function,n_cores):

    if n_cores > 1:

        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()

        for rank in range(n_cores):

            p = mp.Process(target=function, args=(rank,return_dict,))
            processes.append(p)

        [x.start() for x in processes]
        [x.join() for x in processes]

    elif n_cores == 1:

        return_dict = {}
        function(0,return_dict)

    return return_dict

def launch_multiprocessing(function,n_cores):

    if n_cores > 1:

        processes = []
        out_q = mp.Queue()

        for rank in range(n_cores):

            p = mp.Process(target=function, args=(rank,out_q,))
            processes.append(p)

        return_dict = {}

        [x.start() for x in processes]

        for p in processes:

            return_dict.update(out_q.get())

        [x.join() for x in processes]

    elif n_cores == 1:

        return_dict = function(0,{})

    return return_dict

def convolve_1d(x,dn_dx,sigma_scatter,inv_cov=None):

    if sigma_scatter > 0.:

        kernel = gaussian_1d(x-np.mean(x),sigma_scatter)
        dn_dx = signal.convolve(dn_dx,kernel,mode="same",method="fft")/np.sum(kernel)

    return dn_dx

def eval_gaussian_nd(x_mesh,cov=None):

    shape = x_mesh.shape
    x_mesh = x_mesh.reshape(*x_mesh.shape[:-2],-1)
    pdf = stats.multivariate_normal.pdf(np.transpose(x_mesh),cov=cov)
    pdf = pdf.reshape(shape[1:])

    return pdf

def get_mesh(x):

    if x.shape[0] == 1:

        x_mesh = np.array(np.meshgrid(x[0,:]))

    elif x.shape[0] == 2:

        x_mesh = np.array(np.meshgrid(x[0,:],x[1,:]))

    elif x.shape[0] == 3:

        x_mesh = np.array(np.meshgrid(x[0,:],x[1,:],x[2,:]))

    return x_mesh

def gaussian_1d(x,sigma):

    return np.exp(-x**2/(2.*sigma**2))/(np.sqrt(2.*np.pi)*sigma)

def apodise(x_map):

    window_1d = signal.windows.tukey(x_map.shape[0],alpha=0.1)
    window = [window_1d for i in range(0,len(x_map.shape))]
    window = functools.reduce(np.multiply, np.ix_(*window))

    return x_map*window
