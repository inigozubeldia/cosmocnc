import numpy as np
import pylab as pl
from astropy.io import fits
from .config import *

class cluster_catalogue:

    def __init__(self,catalogue_name="Planck_MMF3_cosmo",precompute_cnc_quantities=True,
    bins_obs_select_edges=np.linspace(0.01,1.01,11),bins_z_edges=np.exp(np.linspace(np.log(6.),np.log(100),6))):

        self.catalogue_name = catalogue_name
        self.catalogue = {}
        self.catalogue_patch = {}
        self.precompute_cnc_quantities = precompute_cnc_quantities
        self.bins_obs_select_edges = bins_obs_select_edges
        self.bins_z_edges = bins_z_edges

        if self.catalogue_name == "Planck_MMF3_cosmo":

            threshold = 6.

            fit_union = fits.open(root_path +  'data/HFI_PCCS_SZ-union_R2.08.fits')
            fit_mmf3 = fits.open(root_path +  'data/HFI_PCCS_SZ-MMF3_R2.08.fits')

            data_union = fit_union[1].data
            data_mmf3 = fit_mmf3[1].data

            indices_mmf3 = []
            indices_union = []

            for i in range(0,len(data_mmf3["SNR"])):

                if (data_mmf3["SNR"][i] > threshold) and (data_union["COSMO"][data_mmf3["INDEX"][i]-1] == True):

                    indices_union.append(data_mmf3["INDEX"][i]-1)
                    indices_mmf3.append(i)

            observable = "q_mmf3"

            self.catalogue[observable] = data_mmf3["SNR"][indices_mmf3]
            self.catalogue["z"] = data_union["REDSHIFT"][indices_union]
            self.catalogue_patch[observable] = np.zeros(len(self.catalogue[observable])).astype(np.int)
            self.catalogue["m_lens"] = data_union["MSZ"][indices_union]
            self.catalogue_patch["m_lens"] = np.zeros(len(self.catalogue[observable])).astype(np.int)

            self.n_clusters = len(self.catalogue[observable])
            self.obs_select =  observable

        if self.precompute_cnc_quantities == True:

            self.get_precompute_cnc_quantities()

    def get_precompute_cnc_quantities(self):

        self.indices_no_z = np.where(self.catalogue["z"] < 0)[0]
        self.indices_with_z = np.where(self.catalogue["z"] > 0.)[0]
        self.indices_unique = np.unique(self.catalogue_patch[self.obs_select][self.indices_with_z])

        self.indices_unique_patch = []

        for i in range(0,len(self.indices_unique)):

            patch_index = self.indices_unique[i]
            indices = np.where(self.catalogue_patch[self.obs_select][self.indices_with_z] == patch_index)[0]
            self.indices_unique_patch.append(indices)

        self.number_counts = np.zeros((len(self.bins_z_edges)-1,len(self.bins_obs_select_edges)-1))

        for i in range(0,len(self.bins_z_edges)-1):

            for j in range(0,len(self.bins_obs_select_edges)-1):

                indices = np.where((self.catalogue[self.obs_select] > self.bins_obs_select_edges[j]) & (self.catalogue[self.obs_select] < self.bins_obs_select_edges[j+1])
                & (self.catalogue["z"] > self.bins_z_edges[i]) & (self.catalogue["z"] < self.bins_z_edges[i+1]))[0]
                self.number_counts[i,j] = len(indices)
