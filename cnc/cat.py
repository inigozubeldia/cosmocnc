import numpy as np
import pylab as pl
from astropy.io import fits
from .config import *

class cluster_catalogue:

    def __init__(self,catalogue_name="Planck_MMF3_cosmo",precompute_cnc_quantities=True,
    bins_obs_select_edges=np.linspace(0.01,1.01,11),bins_z_edges=np.exp(np.linspace(np.log(6.),np.log(100),6)),
    observables=None,obs_select=None):

        self.catalogue_name = catalogue_name
        self.catalogue = {}
        self.catalogue_patch = {}
        self.precompute_cnc_quantities = precompute_cnc_quantities
        self.bins_obs_select_edges = bins_obs_select_edges
        self.bins_z_edges = bins_z_edges
        self.observables = observables
        self.obs_select =  obs_select

        if self.catalogue_name == "Planck_MMF3_cosmo":

            #SZ data from Planck Legacy Archive

            threshold = 6.

            fit_union = fits.open(root_path + 'data/HFI_PCCS_SZ-union_R2.08.fits')
            fit_mmf3 = fits.open(root_path + 'data/HFI_PCCS_SZ-MMF3_R2.08.fits')

            data_union = fit_union[1].data
            data_mmf3 = fit_mmf3[1].data

            indices_mmf3 = []
            indices_union = []

            for i in range(0,len(data_mmf3["SNR"])):

                if (data_mmf3["SNR"][i] > threshold) and (data_union["COSMO"][data_mmf3["INDEX"][i]-1] == True):

                    indices_union.append(data_mmf3["INDEX"][i]-1)
                    indices_mmf3.append(i)

            observable = self.obs_select

            self.catalogue[observable] = data_mmf3["SNR"][indices_mmf3]
            self.catalogue["z"] = data_union["REDSHIFT"][indices_union]
            self.catalogue_patch[observable] = np.zeros(len(self.catalogue[observable])).astype(np.int)

            indices_no_z = np.where(self.catalogue["z"] < 0.)[0]

            self.catalogue["z"][indices_no_z] = None

            indices_z = np.argwhere(~np.isnan(self.catalogue["z"]))[:,0]

            patch_index_vec = np.load("/home/iz221/bayesian_bias/cluster_patch_index.npy")

            if self.obs_select == "q_mmf3":

                self.catalogue_patch[observable][indices_z] = patch_index_vec

            #Fake lensing data

            self.catalogue["m_lens"] = data_union["MSZ"][indices_union]
            self.catalogue_patch["m_lens"] = np.zeros(len(self.catalogue[observable])).astype(np.int)
            self.catalogue["m_lens"][indices_no_z] = None

            #CMB lensing data from Zubeldia & Challinor 2019

            [m_cmb_obs,sigma_cmb_obs,m_xry] = np.load("/home/iz221/bayesian_bias/mass_estimates_paper.npy")
            p_obs = m_cmb_obs/sigma_cmb_obs
            cmb_lensing_patches = np.arange(len(p_obs))

            p_zc19 = -np.ones(len(self.catalogue["z"]))
            p_zc19_patches = -np.ones(len(self.catalogue["z"]))

            p_zc19[indices_z] = p_obs
            p_zc19_patches[indices_z] = cmb_lensing_patches
            self.catalogue["p_zc19"] = p_zc19
            self.catalogue_patch["p_zc19"] = p_zc19_patches


        elif self.catalogue_name[0:14] == "zc19_simulated":

            catalogue = np.load(root_path + "data/catalogues_sim/catalogue_" + self.catalogue_name + "_paper.npy",allow_pickle=True)[0]

            self.catalogue = {}
            self.catalogue_patch = {}

            self.catalogue["q_mmf3_mean"] = catalogue["q_mmf3_mean"]
            self.catalogue_patch["q_mmf3_mean"] = catalogue["q_mmf3_mean_patch"]
            self.catalogue["p_zc19"] = catalogue["p_zc19"]
            self.catalogue_patch["p_zc19"] = catalogue["p_zc19_patch"]
            self.catalogue["z"] = catalogue["z"]
            self.catalogue["z_std"] = np.ones(len(self.catalogue["z"]))*0.02

            self.obs_select = "q_mmf3_mean"

        elif self.catalogue_name == "q_mlens_simulated":

            catalogue = np.load(root_path + "data/catalogues_sim/catalogue_q_mlens_simulated_lowsnr.npy",allow_pickle=True)[0]
        #    catalogue = np.load(root_path + "data/catalogues_sim/catalogue_q_mlens_simulated.npy",allow_pickle=True)[0]

            self.catalogue = {}
            self.catalogue_patch = {}

            self.catalogue["q_mmf3_mean"] = catalogue["q_mmf3_mean"]
            self.catalogue_patch["q_mmf3_mean"] = catalogue["q_mmf3_mean_patch"]
            self.catalogue["m_lens"] = catalogue["m_lens"]
            self.catalogue_patch["m_lens"] = catalogue["m_lens_patch"]
            self.catalogue["z"] = catalogue["z"]


        self.n_clusters = len(self.catalogue[self.obs_select])

        if self.precompute_cnc_quantities == True:

            self.get_precompute_cnc_quantities()


    def get_precompute_cnc_quantities(self):

        self.indices_no_z = np.argwhere(np.isnan(self.catalogue["z"]))[:,0]
        self.indices_with_z = np.argwhere(~np.isnan(self.catalogue["z"]))[:,0]

        self.number_counts = np.zeros((len(self.bins_z_edges)-1,len(self.bins_obs_select_edges)-1))

        for i in range(0,len(self.bins_z_edges)-1):

            for j in range(0,len(self.bins_obs_select_edges)-1):

                indices = np.where((self.catalogue[self.obs_select] > self.bins_obs_select_edges[j]) & (self.catalogue[self.obs_select] < self.bins_obs_select_edges[j+1])
                & (self.catalogue["z"] > self.bins_z_edges[i]) & (self.catalogue["z"] < self.bins_z_edges[i+1]))[0]
                self.number_counts[i,j] = len(indices)

        self.obs_select_max = np.max(self.catalogue[self.obs_select])

        self.observable_dict = {}
        self.indices_obs_select = []
        self.indices_other_obs = []

        for i in self.indices_with_z:

            observables_cluster = []

            for observable_set in self.observables:

                observable_set_cluster = []

                for observable in observable_set:

                    if np.isnan(self.catalogue[observable][i]) == False:

                        observable_set_cluster.append(observable)

                observables_cluster.append(observable_set_cluster)

            self.observable_dict[i] = observables_cluster

            if observables_cluster == [[self.obs_select]]:

                self.indices_obs_select.append(i)

            else:

                self.indices_other_obs.append(i)

        self.indices_obs_select = np.array(self.indices_obs_select)
        self.indices_other_obs = np.array(self.indices_other_obs)

        #For clusters with only selection observable, sort them by patch

        self.indices_unique = []
        self.indices_unique_dict = {}

        if len(self.indices_obs_select) > 0:

            self.indices_unique = np.unique(self.catalogue_patch[self.obs_select][self.indices_obs_select])

            for i in range(0,len(self.indices_unique)):

                patch_index = self.indices_unique[i]
                indices = np.where(self.catalogue_patch[self.obs_select][self.indices_obs_select] == patch_index)[0]

                self.indices_unique_dict[str(int(patch_index))] = indices
