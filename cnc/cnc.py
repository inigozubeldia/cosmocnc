import numpy as np
import pylab as pl
from multiprocessing import Pool
from .cosmo import *
from .hmf import *
from .sr import *
from .cat import *
import scipy.signal as signal
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.stats as stats
import time
import functools
import copy
import multiprocessing as mp

cluster_number_counts_params_default = {

    "number_cores": 10,
    "number_cores_hmf": 8,

    "n_points": 2**12,#2**7,
    "M_min": 1e13,
    "M_max": 1e16,
    "hmf_type": "Tinker08",
    "mass_definition": "500c",
    "hmf_type_deriv": "numerical",
    "power_spectrum_type": "cosmopower",

    #"path_to_cosmopower": "/rds-d4/user/iz221/hpc-work/cosmopower/",

    "obs_select_min": 6.,
    "obs_select_max": 100.,
    "n_obs_select": 10000,
    "z_min": 0.01,
    "z_max": 1.01,
    "n_z": 50,

    "obs_select": "q_mmf3", #"q_mmf3_mean",
    "n_patches": 417,
    "cov_patch_dependent":False,
    "obs_select_uncorrelated":False,
    "all_layers_uncorrelated":False, #if True, all observables have uncorrelated scatter
    "last_layer_uncorrelated":False, #if True, it means that the last layer of the observables is uncorrelated
    "first_layer_power_law":False,
    "obs_mass": ["q_mmf3"],
    "cluster_catalogue":"Planck_MMF3_cosmo",

    "bins_edges_z": np.linspace(0.01,1.01,11),
    "bins_edges_obs_select": np.exp(np.linspace(np.log(6.),np.log(60),6))
    }

class cluster_number_counts:

    def __init__(self,cnc_params=None):

        if cnc_params is None:

            cnc_params = cluster_number_counts_params_default

        self.cnc_params = cnc_params

        self.abundance_matrix = None
        self.n_obs_matrix = None
        self.hmf_matrix = None
        self.n_tot = None

    #Loads data (catalogue and scaling relation data)

    def initialise(self):

        self.scaling_relations = {}

        for i in range(0,len(self.cnc_params["obs_mass"])):

            self.scaling_relations[self.cnc_params["obs_mass"][i]] = scaling_relations(observable=self.cnc_params["obs_mass"][i])
            self.scaling_relations[self.cnc_params["obs_mass"][i]].initialise_scaling_relation()

        cosmo_params = cosmo_params_default().params
        self.cosmology = cosmology_model(cosmo_params=cosmo_params,
        power_spectrum_type=self.cnc_params["power_spectrum_type"])

        self.catalogue = cluster_catalogue(catalogue_name=self.cnc_params["cluster_catalogue"],precompute_cnc_quantities=True,
        bins_obs_select_edges=self.cnc_params["bins_edges_obs_select"],bins_z_edges=self.cnc_params["bins_edges_z"])

        self.scal_rel_params = scaling_relation_params_default
        self.scatter = scatter(params=self.scal_rel_params)

    #Updates parameter values (cosmological and scaling relation)

    def reinitialise_computation(self):

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

        self.redshift_vec = np.linspace(self.cnc_params["z_min"],self.cnc_params["z_max"],self.cnc_params["n_z"])
        self.obs_select_vec = np.linspace(self.cnc_params["obs_select_min"],self.cnc_params["obs_select_max"],self.cnc_params["n_obs_select"])

        self.D_A = self.cosmology.background_cosmology.angular_diameter_distance(self.redshift_vec).value
        self.E_z = self.cosmology.background_cosmology.H(self.redshift_vec).value/(self.cosmology.cosmo_params["h"]*100.)

        self.halo_mass_function = halo_mass_function(cosmology=self.cosmology,hmf_type=self.cnc_params["hmf_type"],
        mass_definition=self.cnc_params["mass_definition"],M_min=self.cnc_params["M_min"],
        M_max=self.cnc_params["M_max"],n_points=self.cnc_params["n_points"],type_deriv=self.cnc_params["hmf_type_deriv"])

        #Eval hmf

        n_cores = self.cnc_params["number_cores_hmf"]
        indices_split = np.array_split(np.arange(self.cnc_params["n_z"]),n_cores)
        ranks = np.arange(n_cores)

        self.hmf_matrix = np.zeros((self.cnc_params["n_z"],self.cnc_params["n_points"]))


        #rank = 0

        t0 = time.time()

        def f_mp(rank,return_dict):

            for i in range(0,len(indices_split[rank])):

                lnmass_vec,hmf_eval = self.halo_mass_function.eval_hmf(self.redshift_vec[indices_split[rank][i]],log=True,volume_element=True)
                ln_M = lnmass_vec

                return_dict[str(indices_split[rank][i])] = hmf_eval
                return_dict["ln_M"] = ln_M


        return_dict = launch_multiprocessing(f_mp,n_cores)

        self.ln_M = return_dict["ln_M"]

        t11 = time.time()

        for i in range(0,len(self.redshift_vec)):

            self.hmf_matrix[i] = return_dict[str(i)]

        t1 = time.time()

        print("Time hmf",t1-t0)

    #Computes the cluster abundance across selection observable and redshift

    def get_cluster_abundance(self):

        if self.hmf_matrix is None:

            self.get_hmf()

        t0 = time.time()

        self.scal_rel_selection = self.scaling_relations[self.cnc_params["obs_select"]]

        indices_split = np.array_split(np.arange(self.cnc_params["n_patches"]),self.cnc_params["number_cores"])
        ranks = np.arange(self.cnc_params["number_cores"])

        self.abundance_tensor = np.zeros((self.cnc_params["n_patches"],self.cnc_params["n_z"],self.cnc_params["n_obs_select"]))
        self.n_obs_matrix = np.zeros((self.cnc_params["n_patches"],self.cnc_params["n_obs_select"]))

        def f_mp(rank,return_dict):

            for i in range(0,len(indices_split[rank])):

                abundance_matrix = np.zeros((self.cnc_params["n_z"],self.cnc_params["n_obs_select"]))

                for j in range(0,len(self.redshift_vec)):

                    #print("Redshift",self.redshift_vec[j])

                    patch_index = indices_split[rank][i]

                    other_params = {"D_A": self.D_A[j],"E_z": self.E_z[j],"H0": self.cosmology.background_cosmology.H0.value}

                    dn_dx0 = self.hmf_matrix[j,:]
                    x0 = self.ln_M

                    for k in range(0,self.scal_rel_selection.get_n_layers()):

                        self.scal_rel_selection.precompute_scaling_relation(params=self.scal_rel_params,
                        other_params=other_params,layer=k,patch_index=patch_index)

                        x1 = self.scal_rel_selection.eval_scaling_relation(x0,
                        layer=k,patch_index=patch_index)

                        dx1_dx0 = self.scal_rel_selection.eval_derivative_scaling_relation(x0,
                        observable=self.cnc_params["obs_select"],layer=k,patch_index=patch_index)

                        dn_dx1 = dn_dx0/dx1_dx0
                        x1_interp = np.linspace(np.min(x1),np.max(x1),self.cnc_params["n_points"])
                        dn_dx1 = np.interp(x1_interp,x1,dn_dx1)

                        sigma_scatter = np.sqrt(self.scatter.get_cov(observable1=self.cnc_params["obs_select"],observable2=self.cnc_params["obs_select"],
                        layer=k,patch1=patch_index,patch2=patch_index))

                        #if j == 5 and k == 0:

                        #    pl.plot(x1_interp,dn_dx1)

                        dn_dx1 = convolve_dn(x1_interp,dn_dx1,sigma_scatter,dimension="1")

                        x0 = x1_interp
                        dn_dx0 = dn_dx1

                    dn_dx1_interp = np.interp(self.obs_select_vec,x0,dn_dx0)

                    #self.abundance_tensor[patch_index,j,:] = dn_dx1_interp*4.*np.pi*self.scal_rel_selection.skyfracs[patch_index] #number counts per patch
                    abundance_matrix[j,:] = dn_dx1_interp*4.*np.pi*self.scal_rel_selection.skyfracs[patch_index] #number counts per patch

                n_obs = integrate.simps(abundance_matrix,self.redshift_vec,axis=0)

                return_dict[str(patch_index)] = abundance_matrix
                return_dict[str(patch_index) + "_n_obs"] = n_obs

        n_cores = self.cnc_params["number_cores"]
        return_dict = launch_multiprocessing(f_mp,n_cores)

        for i in range(0,self.cnc_params["n_patches"]):

            self.abundance_tensor[i,:,:] = return_dict[str(i)]#*self.scal_rel_selection.skyfracs[i]
            self.n_obs_matrix[i,:] = return_dict[str(i) + "_n_obs"]

        t1 = time.time()

        print("Time abundance",t1-t0)

        self.abundance_matrix = np.sum(self.abundance_tensor,axis=0)
        print("Time average",time.time()-t1)

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

            t01 = time.time()

            for i in range(0,len(indices_no_z)):

                log_lik_data = log_lik_data + np.log(np.interp(self.catalogue.catalogue[self.cnc_params["obs_select"]][indices_no_z[i]],
                self.obs_select_vec,self.n_obs_matrix[self.catalogue.catalogue_patch[self.cnc_params["obs_select"]][indices_no_z[i]]]))

        #Computes log lik of data for clusters with z if there's only the selection observable

        if observables == [self.cnc_params["obs_select"]]:

            t0 = time.time()

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

                    indices = indices_unique_patch[patch_index]

                z_select = self.catalogue.catalogue["z"][indices_with_z][indices]
                obs_select = self.catalogue.catalogue[self.cnc_params["obs_select"]][indices_with_z][indices]

                log_lik_data = log_lik_data + np.sum(np.log(abundance_interp(np.transpose(np.array([z_select,obs_select])))))

        #Computes log lik of data if there are more observables than the selection observable

        if observables != [self.cnc_params["obs_select"]]:

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

                #redshift = 0.112

                print("Cluster index",cluster_index)

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

                #If the observables have some correlation

                if self.cnc_params["all_layers_uncorrelated"] == False:

                    covariance = covariance_matrix(self.scatter,observables_select,observable_patches,
                    layer=range(self.scal_rel_selection.get_n_layers()))

                    dn_dx0_vec = halo_mass_function_z

                    x0_vec = np.zeros((len(observables_select),len(halo_mass_function_z)))

                    for k in range(0,len(observables_select)):

                        x0_vec[k,:] = self.ln_M

                    n_layers = self.scaling_relations[self.cnc_params["obs_select"]].get_n_layers()

                    for j in range(0,n_layers):

                        #Evaluate abundance applying scaling relation

                        x1_vec = np.zeros((len(observables_select),len(halo_mass_function_z)))
                        dx1_dx0_vec = np.zeros((len(observables_select),len(halo_mass_function_z)))

                        t1 = time.time()

                        for k in range(0,len(observables_select)):

                            scaling = self.scaling_relations[observables_select[k]]

                            scaling.precompute_scaling_relation(params=self.scal_rel_params,other_params=other_params,layer=j,patch_index=observable_patches[k])
                            x1 = scaling.eval_scaling_relation(x0_vec[k,:],layer=j,patch_index=observable_patches[k])
                            dx1_dx0 = scaling.eval_derivative_scaling_relation(x0_vec[k,:],layer=j,patch_index=observable_patches[k])

                            dx1_dx0_vec[k,:] = dx1_dx0
                            x1_vec[k,:] = x1

                        if j == 0:

                            dn_dx0_tensor = np.zeros([len(self.ln_M)]*len(observables))
                            dn_dx0_vec = dn_dx0_vec/np.prod(dx1_dx0_vec,axis=0)
                            np.fill_diagonal(dn_dx0_tensor,dn_dx0_vec)

                        else:

                            jacobian = functools.reduce(np.multiply,np.ix_(*dx1_dx0_vec))
                            dn_dx0_tensor = dn_dx0_tensor/jacobian

                        t2 = time.time()

                        print("Time fill tensor",t2-t1)

                        if (self.cnc_params["first_layer_power_law"] == True and j == 0) or (self.cnc_params["last_layer_uncorrelated"] == True and j == n_layers-1):

                            x1_vec_interp = x1_vec

                        else:

                            x1_vec_interp = np.zeros((len(observables_select),len(halo_mass_function_z)))

                            for k in range(0,len(observables_select)):

                                x1_vec_interp[k,:] = np.linspace(np.min(x1_vec[k,:]),np.max(x1_vec[k,:]),self.cnc_params["n_points"])

                            X,Y = np.meshgrid(x1_vec_interp[0,:],x1_vec_interp[1,:])

                            dn_dx0_tensor = interpolate.RegularGridInterpolator((x1_vec[0,:],x1_vec[1,:]),dn_dx0_tensor,method="linear")((X,Y))

                        if j == 0:

                            dn_dx0_tensor = dn_dx0_tensor/(x1_vec_interp[1,1]-x1_vec_interp[1,0])

                    #    if j == 0:

                    #        integrated = integrate.simps(np.transpose(dn_dx0_tensor),x1_vec_interp[1,:])#*(x1_vec_interp[1,1]-x1_vec_interp[1,0])
                        #    integrated = integrate.simps(dn_dx0_tensor,x1_vec_interp[1,:])#*(x1_vec_interp[1,1]-x1_vec_interp[1,0])

                    #        pl.plot(x1_vec_interp[0,:],integrated)
                            #pl.loglog()
                            #pl.xlim([6.,20.])
                    #        pl.title("Integrated distribution")
                    #        pl.show()

                        t3 = time.time()

                        print("Time interpolate tensor",t3-t2)

                        #Add scatter

                        if self.cnc_params["last_layer_uncorrelated"] == True and j == n_layers-1:

                            X,Y = np.meshgrid(x1_vec_interp[0,:],x1_vec_interp[1,:])

                            scatter_vec = np.zeros(x1_vec_interp.shape)

                            for k in range(0,len(observables_select)):

                                sigma_scatter = np.sqrt(self.scatter.get_cov(observable1=observables_select[l],observable2=observables_select[k],
                                layer=j,patch1=indices_split[rank][i],patch2=indices_split[rank][i]))

                                scatter_vec[k,:] = gaussian_1d(x1_vec_interp[k,:]-self.catalogue.catalogue[observables_select[k]][cluster_index],sigma_scatter)

                            scatter_tensor = functools.reduce(np.multiply,np.ix_(*scatter_vec))
                            lik = dn_dx0_tensor*scatter_tensor

                            for k in range(0,len(observables_select)):

                                lik = integrate.simps(integrand,x1_vec_interp[k])

                            log_lik_data = log_lik_data + np.log(lik)


                        else:

                            inv_cov = covariance.inv_cov[j]
                            cov = covariance.cov[j]
                            dn_dx0_tensor = convolve_dn(x1_vec_interp,dn_dx0_tensor,cov,dimension="n",inv_cov=inv_cov)

                            dn_dx0_tensor[np.where(dn_dx0_tensor < 0.)] = 0.
                            x0_vec = x1_vec_interp

                        print("Time convolution",time.time()-t3)

                        #pl.show()
                        #pl.imshow(dn_dx0_tensor)
                        #pl.show()

                    #Evaluate observation

                    if self.cnc_params["last_layer_uncorrelated"] == False:

                        dn_dx0_tensor_interp = interpolate.RegularGridInterpolator(x1_vec_interp,dn_dx0_tensor)

                        obs_values = np.array([self.catalogue.catalogue[obss][cluster_index] for obss in observables_select])

                        log_lik_data = log_lik_data + np.log(dn_dx0_tensor_interp(obs_values))

                    t_end = time.time()

                    print("Total time per cluster",t_end-t0)

        return log_lik_data

    #Computes the integrated cluster number counts as a function of redshift, selection
    #observable, and the total number counts

    def get_number_counts(self):

        if self.abundance_matrix is None:

            self.get_cluster_abundance()

        self.n_z = integrate.simps(self.abundance_matrix,self.obs_select_vec)
        self.n_obs = integrate.simps(np.transpose(self.abundance_matrix),self.redshift_vec)
        self.n_tot = integrate.simps(self.n_obs,self.obs_select_vec)

    #Computes the unbinned log likelihood

    def log_lik_unbinned(self):

        if self.n_tot is None:

            self.get_number_counts()

        #Poisson term

        log_lik = -self.n_tot

        #Cluster data term

        log_lik = log_lik + self.get_loglik_data(observables=self.cnc_params["obs_mass"])

        return log_lik

    #Computes the binned log likelihood

    def get_lik_binned(self):

        if self.abundance_matrix is None:

            self.get_cluster_abundance()

        log_lik = 0.

        for i in range(0,len(self.cnc_params["bins_edges_z"])-1):

            for j in range(0,len(self.cnc_params["bins_edges_obs_select"])-1):

                redshift_vec_interp = np.linspace(self.cnc_params["bins_edges_z"][i],self.cnc_params["bins_edges_z"][i+1],10)
                obs_select_vec_interp = np.linspace(self.cnc_params["bins_edges_obs_select"][j],self.cnc_params["bins_edges_obs_select"][j+1],100)

                abundance_matrix_interp = interpolate.RegularGridInterpolator((self.redshift_vec,self.obs_select_vec),self.abundance_matrix)
                X,Y = np.meshgrid(redshift_vec_interp,obs_select_vec_interp)
                abundance_matrix_interp = abundance_matrix_interp((X,Y))

                n_tot = integrate.simps(integrate.simps(abundance_matrix_interp,redshift_vec_interp),obs_select_vec_interp)

                n_obs = self.catalogue.number_counts[i,j]

                log_lik = n_tot + n_obs*np.log(n_tot)

        return log_lik

    #Computes the cluster abundance across selection observable and redshift as a
    #function of patch

    def get_number_counts_per_patch(self):

        if self.abundance_matrix is None:

            print("seriosuly computed twice")

            self.get_cluster_abundance()

        t0 = time.time()

        self.n_obs_matrix = integrate.simps(self.abundance_tensor,self.redshift_vec,axis=1)

        t_integral = time.time()

        print("Time INT",t_integral-t0)

    #Computes the data (or "mass calibration") part of the unbinned likelihood


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

def launch_multiprocessing(function,n_cores):

    processes = []
    manager = mp.Manager()
    return_dict = manager.dict()

    for rank in range(n_cores):

        p = mp.Process(target=function, args=(rank,return_dict,))
        processes.append(p)

    [x.start() for x in processes]
    [x.join() for x in processes]

    return return_dict

def convolve_dn(x,dn_dx,sigma_scatter,dimension="1",inv_cov=None):

    if dimension == "1":

        if sigma_scatter > 0.:

            kernel = gaussian_1d(x-np.mean(x),sigma_scatter)
            dn_dx = signal.convolve(dn_dx,kernel,mode="same",method="fft")/np.sum(kernel)

    elif dimension == "n":

        cov = sigma_scatter

        x_c = copy.deepcopy(x)

        for i in range(0,x.shape[0]):

            mean =  np.mean(x_c[i,:])
            x_c[i,:] = x_c[i,:] - float(mean)

        if x.shape[0] == 1:

            x_mesh = np.dstack((np.meshgrid(x_c[0,:])))

        elif x.shape[0] == 2:

            x_mesh = np.dstack((np.meshgrid(x_c[0,:],x_c[1,:])))

        elif x.shape[0] == 3:

            x_mesh = np.dstack((np.meshgrid(x_c[0,:],x_c[1,:],x_c[2,:])))

        kernel = stats.multivariate_normal.pdf(x_mesh,cov=cov)
        dn_dx = signal.convolve(dn_dx,kernel,mode="same",method="fft")/np.sum(kernel)

    return dn_dx

def gaussian_1d(x,sigma):

    return np.exp(-x**2/(2.*sigma**2))/(np.sqrt(2.*np.pi)*sigma)
