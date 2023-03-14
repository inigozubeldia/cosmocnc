import numpy as np
import pylab as pl
from multiprocessing import Pool
import cosmo
import hmf
import sr
import scipy.signal as signal
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import time

class cluster_number_counts_params:

    def __init__(self):

        self.params = {
        "number_cores": 1,

        "n_points": 2**12,
        "M_min": 1e13,
        "M_max": 1e16,
        "hmf_type": "Tinker08",
        "mass_definition": "500c",
        "hmf_type_deriv": "numerical",
        "power_spectrum_type": "cosmopower",

        "obs_select_min": 6.,
        "obs_select_max": 100.,
        "n_obs_select": 10000,
        "z_min": 0.01,
        "z_max": 1.01,
        "n_z": 50,

        "obs_select": "q_mmf3_mean",
        "n_patches": 1,#417
        "cov_patch_dependent":False,
        "obs_select_uncorrelated":True,
        "cnc_quantities_precomputed":False
        }

class cluster_number_counts:

    def __init__(self,cosmo_params=None,cnc_params=None,scal_rel_params=None,catalogue=None,cosmology=None):

        if cosmo_params is None:

            cosmo_params = cosmo.cosmo_params_default().params

        if cnc_params is None:

            cnc_params = cluster_number_counts_params().params

        if cosmology is None:

            self.cosmology = cosmo.cosmology_model(cosmo_params=cosmo_params,power_spectrum_type=cnc_params["power_spectrum_type"])

        else:

            self.cosmology = cosmology
            self.cosmology.update_cosmology(cosmo_params)

        self.cnc_params = cnc_params
        self.cosmo_params = cosmo_params
        self.scal_rel_params = scal_rel_params
        self.catalogue = catalogue
        self.cosmology = cosmology

        self.redshift_vec = np.linspace(cnc_params["z_min"],cnc_params["z_max"],cnc_params["n_z"])
        self.obs_select_vec = np.linspace(cnc_params["obs_select_min"],cnc_params["obs_select_max"],cnc_params["n_obs_select"])

        self.D_A = self.cosmology.background_cosmology.angular_diameter_distance(self.redshift_vec).value
        self.E_z = self.cosmology.background_cosmology.H(self.redshift_vec).value/(self.cosmology.cosmo_params["h"]*100.)

        self.halo_mass_function = hmf.halo_mass_function(cosmology=self.cosmology,hmf_type=cnc_params["hmf_type"],
        mass_definition=cnc_params["mass_definition"],M_min=cnc_params["M_min"],
        M_max=cnc_params["M_max"],n_points=cnc_params["n_points"],type_deriv=cnc_params["hmf_type_deriv"])

        #Eval hmf

        self.hmf_matrix = np.zeros((self.cnc_params["n_z"],self.cnc_params["n_points"]))

        indices_split = np.array_split(np.arange(self.cnc_params["n_z"]),self.cnc_params["number_cores"])
        ranks = np.arange(self.cnc_params["number_cores"])

        self.abundance_matrix = None
        self.n_obs_matrix = None

        rank = 0

        t0 = time.time()

        if 1 > 0:

        #def f(rank):

            for i in range(0,len(indices_split[rank])):

                lnmass_vec,hmf_eval = self.halo_mass_function.eval_hmf(self.redshift_vec[indices_split[rank][i]],log=True,volume_element=True)
                self.hmf_matrix[indices_split[rank][i],:] = hmf_eval
                self.ln_M = lnmass_vec

        t1 = time.time()

        print("Time hmf",t1-t0)

        #if __name__ == '__main__':

        #    ranks = np.arange(n_core)
        #    res = Pool().map(f,ranks)

    def get_cluster_abundance(self):

        t0 = time.time()

        self.scal_rel_selection = sr.scaling_relations()
        self.scal_rel_selection.initialise_scaling_relation(observable=self.cnc_params["obs_select"],
        params=self.scal_rel_params)

        self.scatter = sr.scatter()

        indices_split = np.array_split(np.arange(self.cnc_params["n_patches"]),self.cnc_params["number_cores"])
        ranks = np.arange(self.cnc_params["number_cores"])

        self.abundance_tensor = np.zeros((self.cnc_params["n_patches"],self.cnc_params["n_z"],self.cnc_params["n_obs_select"]))

        rank = 0

        if 1 > 0:
        #def f(rank):

            for i in range(0,len(indices_split[rank])):

                for j in range(0,len(self.redshift_vec)):

                    other_params = {"D_A": self.D_A[j],"E_z": self.E_z[j],"H0": self.cosmology.background_cosmology.H0.value}

                    dn_dx0 = self.hmf_matrix[j,:]
                    x0 = self.ln_M

                    for k in range(0,self.scal_rel_selection.get_n_layers(observable=self.cnc_params["obs_select"])):

                        self.scal_rel_selection.precompute_scaling_relation(observable=self.cnc_params["obs_select"],
                        other_params=other_params,layer=k,patch_index=indices_split[rank][i])

                        x1 = self.scal_rel_selection.eval_scaling_relation(x0,observable=self.cnc_params["obs_select"],
                        layer=k,patch_index=indices_split[rank][i])

                        dx1_dx0 = self.scal_rel_selection.eval_derivative_scaling_relation(x0,
                        observable=self.cnc_params["obs_select"],layer=k,patch_index=indices_split[rank][i])

                        dn_dx1 = dn_dx0/dx1_dx0
                        x1_interp = np.linspace(np.min(x1),np.max(x1),self.cnc_params["n_points"])
                        dn_dx1 = np.interp(x1_interp,x1,dn_dx1)

                        sigma_scatter = np.sqrt(self.scatter.get_cov(observable1=self.cnc_params["obs_select"],observable2=self.cnc_params["obs_select"],
                        layer=k,patch1=indices_split[rank][i],patch2=indices_split[rank][i]))

                        dn_dx1 = convolve_dn(x1_interp,dn_dx1,sigma_scatter)

                        x0 = x1_interp
                        dn_dx0 = dn_dx1

                    dn_dx1_interp = np.interp(self.obs_select_vec,x0,dn_dx0)
                    self.abundance_tensor[indices_split[rank][i],j,:] = dn_dx1_interp*4.*np.pi*self.scal_rel_selection.skyfracs[indices_split[rank][i]] #number counts per patch

        t1 = time.time()

        print("Time abundance",t1-t0)

        #if __name__ == '__main__':

            #ranks = np.arange(n_core)
#            res = Pool().map(f,ranks)

        self.abundance_matrix = np.average(self.abundance_tensor,axis=0,weights=self.scal_rel_selection.skyfracs)*float(self.cnc_params["n_patches"])#*4.*np.pi*np.sum(self.scal_rel_selection.skyfracs)

    def get_number_counts(self):

        if self.abundance_matrix is None:

            self.get_cluster_abundance()

        self.n_z = integrate.simps(self.abundance_matrix,self.obs_select_vec)
        self.n_obs = integrate.simps(np.transpose(self.abundance_matrix),self.redshift_vec)
        self.n_tot = integrate.simps(self.n_obs,self.obs_select_vec)

    def log_lik_unbinned(self,observables=None):

        if self.abundance_matrix is None:

            self.get_cluster_abundance()

        if observables is None:

            observables = [self.cnc_params["obs_select"]]

        #Poisson term

        log_lik = -self.n_tot

        #Cluster data term

        log_lik = log_lik + self.get_loglik_data(observables=observables)

    def get_number_counts_per_patch(self):

        if self.abundance_matrix is None:

            self.get_cluster_abundance()

        self.n_obs_matrix = np.sims(self.abundance_tensor,self.redshift_vec,axis=1)

    def get_loglik_data(self,observables=None):

        if observables is None:

            observables = [self.cnc_params["obs_select"]]

        log_lik_mass_cal = 0.

        #Computes log lik of data for clusters with missing z

        if self.cnc_params["cnc_quantities_precomputed"] == False:

            indices_no_z = np.where(self.catalogue.catalogue["z"] < 0)[0]
            indices_with_z = np.where(self.catalogue.catalogue["z"] > 0.)[0]

        else:

            indices_no_z = self.catalogue.indices_no_z
            indices_with_z = self.catalogue.indices_with_z

        if len(indices_no_z) > 0:

            if self.n_obs_matrix is None:

                self.get_number_counts_per_patch(self)

            for i in range(0,len(indices_no_z)):

                log_lik_data = log_lik_data + np.log(np.interp(self.catalogue.catalogue[self.cnc_params["obs_select"]][indices_no_z[i]],self.obs_select_vec,self.n_obs_matrix[self.catalogue.catalogue_patch[self.cnc_params["obs_select"]][indices_no_z[i]]]))

        #Computes log lik of data for clusters with z if there's only the selection observable or if this is uncorrelated with the other observables

        if observables == [self.cnc_params["obs_select"]] or self.cnc_params["obs_select_uncorrelated"] == True:

            if self.cnc_params["cnc_quantities_precomputed"] == False:

                indices_unique = np.unique(self.catalogue.catalogue_patch[self.cnc_params["obs_select"][indices_with_z]])

            else:

                indices_with_z = self.catalogue.indices_wih_z
                indices_unique = self.catalogue.indices_unique
                indices_unique_patch = self.catalogue.indices_unique_patch

            for i in range(0,len(self.indices_unique)):

                patch_index = indices_unique[i]
                abundance_matrix = self.abundance_tensor[patch_index,:,:][indices_with_z[i]]
                abundance_interp = interpolate.RegularGridInterpolator((self.redshift_vec,self.obs_select_vec),abundance_matrix)

                if self.cnc_params["cnc_quantities_precomputed"] == False:

                    indices = np.where(self.catalogue.catalogue_patch[self.cnc_params["obs_select"]][indices_with_z] == patch_index)[0]

                else:

                    indices = indices_unique_patch[i]

                z_select = self.catalogue.catalogue["z"][indices_with_z][indices]
                obs_select = self.catalogue.catalogue[self.cnc_params["obs_select"]][indices_with_z][indices]

                log_lik_data = log_lik_data + np.sum(np.log(abundance_interp(z_select,obs_select)))

        #Computes log lik of data if there's more observables than the selection observable (removing the selection observable if this is uncorrelated with them, as it has been calculated already)

        if observables != [self.cnc_params["obs_select"]]:

            if self.cnc_params["obs_select_uncorrelated"] == True:

                observables.remove(self.cnc_params["obs_select"])

            hmf_interp = interp1d(self.redshift_vec,self.hmf_matrix,axis=0)

            indices_split = np.array_split(np.arange(len(indices_with_z)),self.cnc_params["number_cores"])
            ranks = np.arange(self.cnc_params["number_cores"])

            rank = 0

            for i in range(0,len(indices_split[rank])):

                cluster_index = indices_with_z[indices_split[rank][i]]

                halo_mass_function_z = hmf_interp(self.catalogue.catalogue["z"][cluster_index])
                #select observables that are available for cluster cluster_index

                observables_select = []
                observable_patches = []

                for obs in observables:

                    if self.catalogue.catalogue[obs][cluster_index] > 0:

                        observables_select.append(obs)
                        observable_patches.append(self.catalogue.catalogue_patch[obs])

                observable_patches = np.array(observable_patches)

                for j in range(0,self.scal_rel_selection.get_n_layers(observable=self.cnc_params["obs_select"])):

                    covariance_matrix = get_covariance_matrix(self.scatter,observables_select,observable_patches,layer=j)

                    eigenvalues,A = np.linalg.eig(covariance_matrix)
                    A_inv = np.linalg.inv(A)

                    A_tensor = np.zeros((A.shape[0],A.shape[1],len(halo_mass_function_z)))
                    A_inv_tensor = np.zeros((A.shape[0],A.shape[1],len(halo_mass_function_z)))

                    for k in range(0,len(halo_mass_function_z)):

                        A_tensor[:,:,k] = A
                        A_inv_tensor[:,:,k] = A_inv

                    x0_vec = np.zeros((len(observables_select,len(halo_mass_function_z))))

                    for k in range(0,len(observables_select)):

                        self.scal_rel_selection.precompute_scaling_relation(observable=observables_select[k],
                        other_params=other_params,layer=k,patch_index=observable_patches[k])

                        x1 = self.scal_rel_selection.eval_scaling_relation(x0,observable=observables_select[k],
                        layer=k,patch_index=observable_patches[k])

                        dx1_dx0 = self.scal_rel_selection.eval_derivative_scaling_relation(x0,
                        observable=observables_select[k],layer=k,patch_index=observable_patches[k])

                        dn_dx1 = dn_dx0/dx1_dx0
                        x1_interp = np.linspace(np.min(x1),np.max(x1),self.cnc_params["n_points"])
                        dn_dx1 = np.interp(x1_interp,x1,dn_dx1)

                        sigma_scatter = np.sqrt(self.scatter.get_cov(observable1=self.cnc_params["obs_select"],observable2=self.cnc_params["obs_select"],
                        layer=k,patch1=indices_split[rank][i],patch2=indices_split[rank][i]))

                        dn_dx1 = convolve_dn(x1_interp,dn_dx1,sigma_scatter)

                        x0_vec[k,:] = x1_interp

        return log_lik_data


def get_covariance_matrix(scatter,observables,observable_patches,layer=0):

    covariance_matrix = np.zeros((len(observables),len(observables)))

    for i in range(0,len(observables)):

        for j in range(0,len(observables)):

            covariance_matrix[i,j] = scatter.get_cov(observable1=observables[i],
            observable2=observables[j],patch1=observable_patches[i],patch2=observable_patches[j],layer=layer)

    return covariance_matrix


def convolve_dn(x,dn_dx,sigma_scatter):

    if sigma_scatter > 0.:

        kernel = gaussian_1d(x-np.mean(x),sigma_scatter)
        dn_dx = signal.convolve(dn_dx,kernel,mode="same",method="fft")/np.sum(kernel)

    return dn_dx

def gaussian_1d(x,sigma):

    return np.exp(-x**2/(2.*sigma**2))/(np.sqrt(2.*np.pi)*sigma)
