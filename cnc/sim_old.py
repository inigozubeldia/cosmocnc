import numpy as np
import pylab as pl
from .cnc import *
from .sr import *
import numba
import time

class catalogue_generator:

    def __init__(self,n_catalogues=1,params_cnc=None,seed=None,scal_rel_params=None,
    cosmo_params=None):

        self.n_data = 0
        self.n_hmf = 0

        self.n_catalogues = n_catalogues

        if params_cnc is None:

            params_cnc = cnc_params_default

        if seed is not None:

            np.random.seed(seed=seed)

        if scal_rel_params is None:

            scal_rel_params = scaling_relation_params_default

        if cosmo_params is None:

            cosmo_params = cosmo_params_default

        self.params_cnc = params_cnc

        number_counts = cluster_number_counts(cnc_params=self.params_cnc)
        number_counts.scal_rel_params = scal_rel_params
        number_counts.cosmo_params = cosmo_params

        number_counts.initialise()
        number_counts.get_number_counts()
        self.n_tot = number_counts.n_tot
        self.number_counts = number_counts

        print("N tot",self.n_tot)

        self.get_number_clusters()

        self.scaling_relations = {}

        self.scaling_relations = self.number_counts.scaling_relations
        self.scatter = self.number_counts.scatter

        self.skyfracs = self.scaling_relations[self.params_cnc["obs_select"]].skyfracs

        hmf_matrix = number_counts.hmf_matrix
        self.ln_M = number_counts.ln_M
        self.redshift_vec = number_counts.redshift_vec
        self.hmf_interp = interpolate.interp2d(self.ln_M,self.redshift_vec,hmf_matrix)
        self.hmf_range = np.array([np.min(hmf_matrix),np.max(hmf_matrix)])


    def get_number_clusters(self):

        self.n_tot_obs = np.random.poisson(lam=self.n_tot,size=self.n_catalogues)

    def get_sky_patches(self):

        self.sky_patches = {}
        p = self.skyfracs/np.sum(self.skyfracs)

        for i in range(0,self.n_catalogues):

            patches = np.random.multinomial(self.n_tot_obs[i],p)
            self.sky_patches[i] = np.repeat(np.arange(len(patches)),patches)

            print(self.sky_patches[i])

    #@numba.jit
    def get_individual_sample_hmf(self):

        self.n_data = self.n_data + 1

        hmf_eval = 0.
        hmf_sample = 1.

        while hmf_sample > hmf_eval:

            self.n_hmf = self.n_hmf + 1

            ln_M_sample = np.random.rand()*(self.ln_M[-1]-self.ln_M[0])+self.ln_M[0]
            redshift_sample = np.random.rand()*(self.redshift_vec[-1]-self.redshift_vec[0])+self.redshift_vec[0]
            hmf_sample = np.random.rand()*(self.hmf_range[-1]-self.hmf_range[0])+self.hmf_range[0]
            hmf_eval = self.hmf_interp(ln_M_sample,redshift_sample)

        t1 = time.time()

        return (ln_M_sample,redshift_sample)

    def get_individual_sample(self,patch_index=0):

        n_layers = self.scaling_relations[self.params_cnc["obs_select"]].get_n_layers()

        observable_patches = {}

        for observable in self.params_cnc["observables"][0]:

            observable_patches[observable] = 0

        observable_patches[self.params_cnc["obs_select"]] = patch_index

        covariance = covariance_matrix(self.scatter,self.params_cnc["observables"][0],
        observable_patches=observable_patches,layer=np.arange(n_layers))

        x_select = 0.

        self.n_data = 0
        self.n_hmf = 0

        while x_select < self.params_cnc["obs_select_min"]:

            (ln_M_sample,redshift_sample) = self.get_individual_sample_hmf()
            n_observables = len(self.params_cnc["observables"][0])

            x0 = np.repeat(ln_M_sample,n_observables)

            D_A = np.interp(redshift_sample,self.redshift_vec,self.number_counts.D_A)
            E_z = np.interp(redshift_sample,self.redshift_vec,self.number_counts.E_z)
            D_l_CMB = np.interp(redshift_sample,self.redshift_vec,self.number_counts.D_l_CMB)
            rho_c = np.interp(redshift_sample,self.redshift_vec,self.number_counts.rho_c)

            other_params = {"D_A": D_A,"E_z": E_z,
            "H0": self.number_counts.cosmology.background_cosmology.H0.value,
            "D_l_CMB":D_l_CMB,"rho_c":rho_c,"D_CMB":self.number_counts.cosmology.D_CMB}

            for i in range(0,n_layers):

                x1 = np.zeros(n_observables)

                for j in range(0,n_observables):

                    scal_rel = self.scaling_relations[self.params_cnc["observables"][0][j]]

                    scal_rel.precompute_scaling_relation(params=self.number_counts.scal_rel_params,
                    other_params=other_params,patch_index=patch_index)

                    x1[j] = scal_rel.eval_scaling_relation(x0[j],
                    layer=i,patch_index=patch_index)

                cov = covariance.cov[i]
                noise = np.random.multivariate_normal(np.zeros(n_observables),cov)
                x1 = x1 + noise
                x0 = x1

            x_select = x1[0]

        print("n data",self.n_data)
        print("n hmf",self.n_hmf)

        return x1,ln_M_sample,redshift_sample

    def get_observables(self):

        self.catalogue_list = []
        self.catalogue_patch_list = []

        for i in range(0,self.n_catalogues):

            catalogue = {}
            n_tot = self.n_tot_obs[i]

            for observable in self.params_cnc["observables"][0]:

                catalogue[observable] = -np.ones(n_tot)
                catalogue[observable + "_patch"] = np.zeros(n_tot)

            catalogue["z"] = -np.ones(n_tot)
            catalogue["M"] = -np.ones(n_tot)

            catalogue[self.params_cnc["obs_select"] + "_patch"] = self.sky_patches[i]

            for j in range(0,n_tot):

                x,lnM,z = self.get_individual_sample(patch_index=int(self.sky_patches[i][j]))

                print(j,x,z)

                for k in range(0,len(self.params_cnc["observables"][0])):

                    catalogue[self.params_cnc["observables"][0][k]][j] = x[k]

                catalogue["z"][j] = z
                catalogue["M"][j] = np.exp(lnM)

            self.catalogue_list.append(catalogue)

    def generate_catalogues(self):

        self.get_number_clusters()
        self.get_sky_patches()
        self.get_observables()

def convert_array(a):
    b = np.repeat(np.arange(len(a)), a)
    return b

def inverse_cpdf_sampling(n_samples,x,pdf):

    a = 1.
