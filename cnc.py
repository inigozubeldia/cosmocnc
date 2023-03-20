import numpy as np
import pylab as pl
from multiprocessing import Pool
import cosmo
import hmf
import sr
import scipy.signal as signal
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.stats as stats
import time
import functools
import copy

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
        "path_to_cosmopower": "your/path/to/cosmopower",

        "obs_select_min": 6.,
        "obs_select_max": 100.,
        "n_obs_select": 10000,
        "z_min": 0.01,
        "z_max": 1.01,
        "n_z": 50,

        "obs_select": "q_mmf3_mean",
        "n_patches": 1,#417
        "cov_patch_dependent":False,
        "obs_select_uncorrelated":False,
        "other_obs_uncorrelated":False, #if True, all the other observables (other than the selection one) are assumed to be uncorrelated
        }

class cluster_number_counts:

    def __init__(self,cosmo_params=None,cnc_params=None,scal_rel_params=None,catalogue=None,cosmology=None,scaling_relations=None):

        if cosmo_params is None:

            cosmo_params = cosmo.cosmo_params_default().params

        if cnc_params is None:

            cnc_params = cluster_number_counts_params().params

        if cosmology is None:

            self.cosmology = cosmo.cosmology_model(cosmo_params=cosmo_params,
            power_spectrum_type=cnc_params["power_spectrum_type"],path_to_cosmopower=cnc_aprams["path_to_cosmopower"])

        else:

            self.cosmology = cosmology
            self.cosmology.update_cosmology(cosmo_params)

        self.cnc_params = cnc_params
        self.cosmo_params = cosmo_params
        self.scal_rel_params = scal_rel_params
        self.catalogue = catalogue
        self.cosmology = cosmology
        self.scaling_relations = scaling_relations

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

        self.scal_rel_selection = self.scaling_relations[self.cnc_params["obs_select"]]

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

                    for k in range(0,self.scal_rel_selection.get_n_layers()):

                        self.scal_rel_selection.precompute_scaling_relation(
                        other_params=other_params,layer=k,patch_index=indices_split[rank][i])

                        x1 = self.scal_rel_selection.eval_scaling_relation(x0,
                        layer=k,patch_index=indices_split[rank][i])

                        dx1_dx0 = self.scal_rel_selection.eval_derivative_scaling_relation(x0,
                        observable=self.cnc_params["obs_select"],layer=k,patch_index=indices_split[rank][i])

                        dn_dx1 = dn_dx0/dx1_dx0
                        x1_interp = np.linspace(np.min(x1),np.max(x1),self.cnc_params["n_points"])
                        dn_dx1 = np.interp(x1_interp,x1,dn_dx1)

                        sigma_scatter = np.sqrt(self.scatter.get_cov(observable1=self.cnc_params["obs_select"],observable2=self.cnc_params["obs_select"],
                        layer=k,patch1=indices_split[rank][i],patch2=indices_split[rank][i]))

                        dn_dx1 = convolve_dn(x1_interp,dn_dx1,sigma_scatter,dimension="1")

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

        self.n_obs_matrix = integrate.simps(self.abundance_tensor,self.redshift_vec,axis=1)

    def get_loglik_data(self,observables=None):

        if observables is None:

            observables = [self.cnc_params["obs_select"]]

        #Computes log lik of data for clusters with missing z

        if self.catalogue.precompute_cnc_quantities == False:

            indices_no_z = np.where(self.catalogue.catalogue["z"] < 0)[0]
            indices_with_z = np.where(self.catalogue.catalogue["z"] > 0.)[0]

        else:

            indices_no_z = self.catalogue.indices_no_z
            indices_with_z = self.catalogue.indices_with_z

        log_lik_data = 0.

        if len(indices_no_z) > 0:

            print("No z",len(indices_no_z))

            if self.n_obs_matrix is None:

                self.get_number_counts_per_patch()

            for i in range(0,len(indices_no_z)):

                log_lik_data = log_lik_data + np.log(np.interp(self.catalogue.catalogue[self.cnc_params["obs_select"]][indices_no_z[i]],
                self.obs_select_vec,self.n_obs_matrix[self.catalogue.catalogue_patch[self.cnc_params["obs_select"]][indices_no_z[i]]]))

        #Computes log lik of data for clusters with z if there's only the selection observable or if this is uncorrelated with the other observables

        if observables == [self.cnc_params["obs_select"]] or self.cnc_params["obs_select_uncorrelated"] == True:

            print("With z")

            if self.catalogue.precompute_cnc_quantities == False:

                indices_unique = np.unique(self.catalogue.catalogue_patch[self.cnc_params["obs_select"]][indices_with_z])

            else:

                indices_unique = self.catalogue.indices_unique
                indices_unique_patch = self.catalogue.indices_unique_patch

            for i in range(0,len(indices_unique)):

                patch_index = indices_unique[i]
                abundance_matrix = self.abundance_tensor[patch_index,:,:]
                abundance_interp = interpolate.RegularGridInterpolator((self.redshift_vec,self.obs_select_vec),abundance_matrix)

                if self.catalogue.precompute_cnc_quantities == False:

                    indices = np.where(self.catalogue.catalogue_patch[self.cnc_params["obs_select"]][indices_with_z] == patch_index)[0]

                else:

                    indices = indices_unique_patch[i]

                z_select = self.catalogue.catalogue["z"][indices_with_z][indices]
                obs_select = self.catalogue.catalogue[self.cnc_params["obs_select"]][indices_with_z][indices]

                log_lik_data = log_lik_data + np.sum(np.log(abundance_interp(np.transpose(np.array([z_select,obs_select])))))

        #Computes log lik of data if there are more observables than the selection observable
        #(removing the selection observable if this is uncorrelated with them, as this has been calculated already)

        print("Observables",observables,self.cnc_params["other_obs_uncorrelated"])

        if observables != [self.cnc_params["obs_select"]] and self.cnc_params["other_obs_uncorrelated"] == False:

            print("Here")

            if self.cnc_params["obs_select_uncorrelated"] == True:

                observables.remove(self.cnc_params["obs_select"])

            hmf_interp = interpolate.interp1d(self.redshift_vec,self.hmf_matrix,axis=0)

            indices_split = np.array_split(np.arange(len(indices_with_z)),self.cnc_params["number_cores"])
            ranks = np.arange(self.cnc_params["number_cores"])

            rank = 0

            for i in range(0,len(indices_split[rank])):

                cluster_index = indices_with_z[indices_split[rank][i]]
                redshift = self.catalogue.catalogue["z"][cluster_index]

                #print("Cluster index",cluster_index)

                t0 = time.time()

                D_A = np.interp(redshift,self.redshift_vec,self.D_A)
                E_z = np.interp(redshift,self.redshift_vec,self.E_z)

                other_params = {"D_A": D_A,"E_z": E_z,"H0": self.cosmology.background_cosmology.H0.value}

                halo_mass_function_z = hmf_interp(redshift)

                #select observables that are available for cluster cluster_index

                observables_select = []
                observable_patches = []

                for obs in observables:

                    if self.catalogue.catalogue[obs][cluster_index] > 0:

                        observables_select.append(obs)
                        observable_patches.append(self.catalogue.catalogue_patch[obs][cluster_index])

                observable_patches = np.array(observable_patches)

                covariance = covariance_matrix(self.scatter,observables_select,observable_patches,
                layer=range(self.scal_rel_selection.get_n_layers()))

                dn_dx0_vec = halo_mass_function_z

                x0_vec = np.zeros((len(observables_select),len(halo_mass_function_z)))

                for k in range(0,len(observables_select)):

                    x0_vec[k,:] = self.ln_M

                for j in range(0,self.scaling_relations[self.cnc_params["obs_select"]].get_n_layers()):

                    #print("j",j)

                    #Evaluate abundance applying scaling relation

                    x1_vec = np.zeros((len(observables_select),len(halo_mass_function_z)))
                    dx1_dx0_vec = np.zeros((len(observables_select),len(halo_mass_function_z)))

                    t1 = time.time()

                    for k in range(0,len(observables_select)):

                        scaling = self.scaling_relations[observables_select[k]]

                        scaling.precompute_scaling_relation(other_params=other_params,layer=j,patch_index=observable_patches[k])
                        x1 = scaling.eval_scaling_relation(x0_vec[k,:],layer=j,patch_index=observable_patches[k])
                        dx1_dx0 = scaling.eval_derivative_scaling_relation(x0_vec[k,:],layer=j,patch_index=observable_patches[k])

                        dx1_dx0_vec[k,:] = dx1_dx0
                        x1_vec[k,:] = x1

                        """
                        if j == 0:

                            pl.loglog(np.exp(x0_vec[k,:]),np.exp(x1))
                            pl.title(str(j) + " " + str(k))
                            pl.show()

                        else:

                            pl.loglog(np.exp(x0_vec[k,:]),x1)
                            pl.title(str(j) + " " + str(k))
                            pl.show()
                        """


                    if j == 0:

                        dn_dx0_tensor = np.zeros([len(self.ln_M)]*len(observables))
                        dn_dx0_vec = dn_dx0_vec/np.prod(dx1_dx0_vec,axis=0)
                        np.fill_diagonal(dn_dx0_tensor,dn_dx0_vec)

                    else:

                        jacobian = functools.reduce(np.multiply,np.ix_(*dx1_dx0_vec))
                        dn_dx0_tensor = dn_dx0_tensor/jacobian

                    t2 = time.time()

                #    print("Time fill tensor",t2-t1)

                    x1_vec_interp = np.zeros((len(observables_select),len(halo_mass_function_z)))

                    for k in range(0,len(observables_select)):

                        x1_vec_interp[k,:] = np.linspace(np.min(x1_vec[k,:]),np.max(x1_vec[k,:]),self.cnc_params["n_points"])

                    """
                    input_points = (x1_vec[0,:],)
                    interp_points = (x1_vec_interp[0,:],)

                    for k in range(1,len(observables_select)):

                        input_points + (x1_vec[k,:])
                        interp_points + (x1_vec_interp[k,:])
                    """

                    #pl.imshow(dn_dx0_tensor)
                    #pl.show()

                    X,Y = np.meshgrid(x1_vec_interp[0,:],x1_vec_interp[1,:])
                    #X,Y = np.meshgrid(x1_vec[0,:],x1_vec[1,:])

                    #pl.plot(x1_vec[0,:],x1_vec_interp[0,:])
                    #pl.show()

                    dn_dx0_tensor = interpolate.RegularGridInterpolator((x1_vec[0,:],x1_vec[1,:]),dn_dx0_tensor)((X,Y))

                #    pl.imshow(dn_dx0_tensor)
                    #pl.title("Interpolated")
                    #pl.show()

                    t3 = time.time()

                    #print("Time interpolate tensor",t3-t2)

                    #Add scatter

                    inv_cov = covariance.inv_cov[j]
                    cov = covariance.cov[j]
                    dn_dx0_tensor = convolve_dn(x1_vec_interp,dn_dx0_tensor,cov,dimension="n",inv_cov=inv_cov)
                    x0_vec = x1_vec_interp

                #    print(x1_vec_interp[0,:])
                #    print(x1_vec_interp[1,:])

                    #pl.imshow(dn_dx0_tensor)
                    #pl.title("Convolved")
                    #pl.show()

                    #print("Time convolution",time.time()-t3)

                integrated = np.sum(dn_dx0_tensor,axis=1)*(x0_vec[1,1]-x0_vec[1,0])

                #pl.plot(x0_vec[0,:],integrated)
                #pl.xlim([6.,20.])
                #pl.title("Integrated distribution")
                #pl.show()

                #Evaluate observation

                dn_dx0_tensor_interp = interpolate.RegularGridInterpolator(x1_vec_interp,dn_dx0_tensor)


                #obs_values = np.array([self.catalogue.catalogue[obss][cluster_index] for obss in observables_select])
                #log_lik_data = log_lik_data + np.log(dn_dx0_tensor_interp(obs_values))

                t_end = time.time()
                #print("Total time per cluster",t_end-t0)

        return log_lik_data

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
                    observable2=observables[j],patch1=observable_patches[i],patch2=observable_patches[j],layer=self.layer[k])

            self.cov.append(cov_matrix)
            self.inv_cov.append(np.linalg.inv(cov_matrix))

    def diagonalise(self):

        self.eigenvalues = []
        self.A = []
        self.A_inv = []

        for i in range(0,len(self.layer)):

            eigenvalue,A_matrix = np.linalg.eig(self.cov[i])
            A_inv_matrix = np.linalg.inv(A_matrix)

            self.eigenvalues.append(eigenvalue)
            self.A.append(A_matrix)
            self.A_inv.append(A_inv_matrix)

    def repeat(self,n_repeat):

        self.A_repeat = []
        self.A_inv_repeat = []

        for i in range(0,len(self.layer)):

            self.A_repeat.append(np.repeat(self.A[i][:,:,np.newaxis],n_repeat,axis=2))
            self.A_inv_repeat.append(np.repeat(self.A_inv[i][:,:,np.newaxis],n_repeat,axis=2))


def convolve_dn(x,dn_dx,sigma_scatter,dimension="1",inv_cov=None):

    if dimension == "1":

        if sigma_scatter > 0.:

            kernel = gaussian_1d(x-np.mean(x),sigma_scatter)
            dn_dx = signal.convolve(dn_dx,kernel,mode="same",method="fft")/np.sum(kernel)

    elif dimension == "n":

        cov = sigma_scatter

        x_c = copy.deepcopy(x)
        dx = np.zeros(x.shape[0])

        for i in range(0,x.shape[0]):

            mean =  np.mean(x_c[i,:])
            x_c[i,:] = x_c[i,:] - float(mean)
            dx[i] =  x_c[i,1]-x_c[i,0]

        if x.shape[0] == 1:

            x_mesh = np.dstack((np.meshgrid(x_c[0,:])))

        elif x.shape[0] == 2:

            x_mesh = np.dstack((np.meshgrid(x_c[0,:],x_c[1,:])))

        elif x.shape[0] == 3:

            x_mesh = np.dstack((np.meshgrid(x_c[0,:],x_c[1,:],x_c[2,:])))

        kernel = stats.multivariate_normal.pdf(x_mesh,cov=cov)
        dn_dx = signal.convolve(dn_dx,kernel,mode="same",method="fft")*np.prod(dx)#/np.sum(kernel)

    return dn_dx

def gaussian_1d(x,sigma):

    return np.exp(-x**2/(2.*sigma**2))/(np.sqrt(2.*np.pi)*sigma)
