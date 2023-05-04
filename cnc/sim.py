import numpy as np
import pylab as pl
from .cnc import *
from .sr import *

params_generator_default = {
"observables": ["q_mmf3_mean","p_zc19"], #selection observable should always go first
"obs_select": "q_mmf3_mean",
"obs_select_min":6.,
"n_catalogues":1
}

class catalogue_generator:

    def __init__(self,params=None,params_cnc=None,seed=None):

        if params is None:

            params = params_generator_default

        if params_cnc is None:

            params_cnc = cnc_params_default

        if seed is not None:

            np.random.seed(seed=seed)

        self.params = params
        self.params_cnc = params_cnc

        self.params_cnc["obs_select"] = self.params["obs_select"]
        self.params_cnc["obs_select_min"] = self.params["obs_select_min"]
        self.params_cnc["M_min"] = 4e13

        number_counts = cluster_number_counts(cnc_params=self.params_cnc)
        number_counts.initialise()
        number_counts.get_number_counts()
        self.n_tot = number_counts.n_tot
        self.number_counts = number_counts

        print("N tot",self.n_tot)

        self.get_number_clusters()

        self.scaling_relations = {}

        for observable in self.params["observables"]:

            self.scaling_relations[observable] = scaling_relations(observable=observable)
            self.scaling_relations[observable].initialise_scaling_relation()

        self.scatter = scatter(params=self.number_counts.scal_rel_params)


        self.skyfracs = self.scaling_relations[self.params["obs_select"]].skyfracs

        hmf_matrix = number_counts.hmf_matrix
        self.ln_M = number_counts.ln_M
        self.redshift_vec = number_counts.redshift_vec
        self.hmf_interp = interpolate.interp2d(self.ln_M,self.redshift_vec,hmf_matrix)
        self.hmf_range = np.array([np.min(hmf_matrix),np.max(hmf_matrix)])


    def get_number_clusters(self):

        self.n_tot_obs = np.random.poisson(lam=self.n_tot,size=self.params["n_catalogues"])

    def get_sky_patches(self):

        self.sky_patches = {}
        p = self.skyfracs/np.sum(self.skyfracs)

        for i in range(0,self.params["n_catalogues"]):

            patches = np.random.multinomial(np.arange(0,len(self.skyfracs)),p,size=self.n_tot_obs[i])
            patches_vec = np.zeros(len(patches))

            for j in range(0,len(patches)):

                patches_vec[j] = patches[j][0]

            self.sky_patches[i] = patches_vec

    def get_individual_sample_hmf(self):

        hmf_eval = 0.
        hmf_sample = 1.

        while hmf_sample > hmf_eval:

            ln_M_sample = np.random.rand()*(self.ln_M[-1]-self.ln_M[0])+self.ln_M[0]
            redshift_sample = np.random.rand()*(self.redshift_vec[-1]-self.redshift_vec[0])+self.redshift_vec[0]
            hmf_sample = np.random.rand()*(self.hmf_range[-1]-self.hmf_range[0])+self.hmf_range[0]
            hmf_eval = self.hmf_interp(ln_M_sample,redshift_sample)

        return (ln_M_sample,redshift_sample)

    def get_individual_sample(self,patch_index=0):

        n_layers = self.scaling_relations[self.params["obs_select"]].get_n_layers()

        observable_patches = {}

        for observable in self.params["observables"]:

            observable_patches[observable] = 0

        observable_patches[self.params["obs_select"]] = patch_index

        covariance = covariance_matrix(self.scatter,self.params["observables"],
        observable_patches=observable_patches,layer=np.arange(n_layers))

        x_select = 0.

        while x_select < self.params["obs_select_min"]:

            (ln_M_sample,redshift_sample) = self.get_individual_sample_hmf()
            n_observables = len(self.params["observables"])

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

                    scal_rel = self.scaling_relations[self.params["observables"][j]]

                    scal_rel.precompute_scaling_relation(params=self.number_counts.scal_rel_params,
                    other_params=other_params,layer=i,patch_index=patch_index)

                    x1[j] = scal_rel.eval_scaling_relation(x0[j],
                    layer=i,patch_index=patch_index)

                noise = np.random.multivariate_normal(np.zeros(n_observables),covariance.cov[i])
                x1 = x1 + noise
                x0 = x1

            x_select = x1[0]

        return x1,redshift_sample

    def get_observables(self):

        self.catalogue_list = []
        self.catalogue_patch_list = []

        for i in range(0,self.params["n_catalogues"]):

            catalogue = {}
            n_tot = self.n_tot_obs[i]

            for observable in self.params["observables"]:

                catalogue[observable] = -np.ones(n_tot)
                catalogue[observable + "_patch"] = np.zeros(n_tot)

            catalogue["z"] = -np.ones(n_tot)

            catalogue[self.params["obs_select"] + "_patch"] = self.sky_patches[i]

            for j in range(0,n_tot):

                x,z = self.get_individual_sample(patch_index=int(self.sky_patches[i][j]))

                print(j,x,z)

                for k in range(0,len(self.params["observables"])):

                    catalogue[self.params["observables"][k]][j] = x[k]

                catalogue["z"][j] = z

            self.catalogue_list.append(catalogue)

    def generate_catalogues(self):

        self.get_number_clusters()
        self.get_sky_patches()
        self.get_observables()
