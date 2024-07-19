import numpy as np
import pylab as pl
import scipy.integrate as integrate
from astropy.io import fits
from astropy.table import Table
from .config import *
from .sr import *
from .utils import *
from .params import *
import pickle

class cluster_catalogue:

    def __init__(self,catalogue_name=None,
                 precompute_cnc_quantities=True,
                 bins_obs_select_edges=np.linspace(0.01,1.01,11),
                 bins_z_edges=np.exp(np.linspace(np.log(6.),np.log(100),6)),
                 observables=None, # "q_mmf3_mean","p_zc19", etc
                 obs_select=None,
                 cnc_params=None,
                 scal_rel_params=None):

        if scal_rel_params is None:

            scal_rel_params = scaling_relation_params_default

        self.catalogue_name = catalogue_name
        self.catalogue = {}
        self.catalogue_patch = {}
        self.precompute_cnc_quantities = precompute_cnc_quantities
        self.cnc_params = cnc_params
        self.scal_rel_params = scal_rel_params


        if isinstance(bins_obs_select_edges,str):

            bins_obs_select_edges = eval(bins_obs_select_edges)

        if isinstance(bins_z_edges,str):

            bins_z_edges = eval(bins_z_edges)

        if isinstance(observables,str):

            observables = eval(observables)

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
            self.catalogue_patch[observable] = np.zeros(len(self.catalogue[observable])).astype(np.int64)

            indices_no_z = np.where(self.catalogue["z"] < 0.)[0]

            self.catalogue["z"][indices_no_z] = None
            self.catalogue["z_std"] = np.zeros(len(self.catalogue["z"]))

            indices_z = np.argwhere(~np.isnan(self.catalogue["z"]))[:,0]

            if self.obs_select == "q_mmf3":

                patch_index_vec = np.load(root_path + "data/cluster_patch_index.npy")

                downsample_flag = self.cnc_params["catalogue_params"]["downsample"]

                if downsample_flag == False:

                    self.catalogue_patch[observable][indices_z] = patch_index_vec

                elif downsample_flag == True:

                    (self.sigma_matrix,self.skyfracs,original_tile_vec) = np.load(root_path + "data/mmf3_noise_downsampled.npy",allow_pickle=True)

                    patch_index_downsampled = np.zeros(len(patch_index_vec))

                    for i in range(0,len(patch_index_vec)):

                        patch_index_downsampled[i] = original_tile_vec[int(patch_index_vec[i])]

                    self.catalogue_patch[observable][indices_z] = patch_index_downsampled


            #Fake lensing data

            self.catalogue["m_lens"] = data_union["MSZ"][indices_union]
            self.catalogue_patch["m_lens"] = np.zeros(len(self.catalogue[observable])).astype(np.int64)
            self.catalogue["m_lens"][indices_no_z] = None

            #CMB lensing data from Zubeldia & Challinor 2019

            [m_cmb_obs,sigma_cmb_obs,m_xry] = np.load(root_path + "data/mass_estimates_paper.npy")
            p_obs = m_cmb_obs/sigma_cmb_obs
            cmb_lensing_patches = np.arange(len(p_obs))

            p_zc19 = -np.ones(len(self.catalogue["z"]))
            p_zc19_patches = -np.ones(len(self.catalogue["z"]))

            p_zc19[indices_z] = p_obs
            p_zc19_patches[indices_z] = cmb_lensing_patches
            p_zc19[indices_no_z] = None
            p_zc19_patches[indices_no_z] = None

            self.catalogue["p_zc19"] = p_zc19
            self.catalogue_patch["p_zc19"] = p_zc19_patches

            #Stacked CMB lensing data

            self.stacked_data_labels = ["p_zc19_stacked"]

            self.catalogue_patch["p_zc19_stacked"] = p_zc19_patches #if one wants to use p with just one layer

            self.stacked_data = {"p_zc19_stacked":{}}

            self.stacked_data["p_zc19_stacked"]["data_vec"] = np.mean(p_obs)
            self.stacked_data["p_zc19_stacked"]["inv_cov"] = float(len(p_obs))
            self.stacked_data["p_zc19_stacked"]["cluster_index"] = indices_z
            self.stacked_data["p_zc19_stacked"]["observable"] = "p_zc19"

            self.catalogue["validated"] = np.ones(len(self.catalogue[self.obs_select]))

        elif self.catalogue_name == "planck_szifi_validation":

            path = "/rds-d4/user/iz221/hpc-work/catalogues_szifi_planck/"

            #prename = "planck_it_find_apodold_planckcibparams_sim_val_bias062_theta45"
            #prename = "planck_it_find_apodold_planckcibparams_sim_val_fixed"
            #prename = "planck_it_find_apodold_planckcibparams_sim_val_bias062"
            theta_label = self.cnc_params["catalogue_params"]["theta_label"]

            prename = "planck_it_find_apodold_planckcibparams_sim_val_bias062_" + theta_label

            suffix = self.cnc_params["catalogue_params"]["filter"]

            import szifi

        #    name = path + prename + "_" + suffix + "_processed_id5arcmin.npy"
            #name = path + prename + "_" + suffix + "_processed_cross_matched.npy"
            #name = path + prename + "_" + suffix + "_processed_cross_matched_noid.npy"
            # name = path + prename + "_" + suffix + "_processed_cross_matched_id1arcmin.npy"
            # name = path + prename + "_" + suffix + "_processed_cross_matched_id3arcmin.npy"
            #name = path + prename + "_" + suffix + "_processed_cross_matched_idmask1theta.npy"
            #name = path + prename + "_" + suffix + "_processed_cross_matched_idmaskfixed5arcmin.npy"
            #name = path + prename + "_" + suffix + "_processed_idmaskfixed5arcmin.npy"

            suffix_merge = self.cnc_params["catalogue_params"]["suffix_merge"]

            if theta_label == "":

                name = path + prename + "" + suffix + "_processed_cross_matched_" + suffix_merge + ".npy"

            else:

                name = path + prename + "_" + suffix + "_processed_cross_matched_" + suffix_merge + ".npy"

            #name = path + prename + "_" + suffix + "_processed.npy"

            print(name)

            (catalogue_obs_it,catalogue_obs_noit) = np.load(name,allow_pickle=True)[()]

            if self.cnc_params["catalogue_params"]["iterative"] == True:

                catalogue_obs = catalogue_obs_it
                it_label = "it"

            if self.cnc_params["catalogue_params"]["iterative"] == False:

                catalogue_obs = catalogue_obs_noit
                it_label = "noit"

            #catalogue_obs.catalogue["validation"] = np.ones(len(catalogue_obs.catalogue["z"])) #REMOVEEEE

            if self.cnc_params["catalogue_params"]["only_validated"] == True:

                indices_val = np.where(catalogue_obs.catalogue["validation"] > 0.)[0]
                catalogue_obs = szifi.get_catalogue_indices(catalogue_obs,indices_val)

            indices = np.where((catalogue_obs.catalogue["z"] > self.cnc_params["z_min"]) & (catalogue_obs.catalogue["z"] < self.cnc_params["z_max"]) & (catalogue_obs.catalogue["q_opt"] > self.cnc_params["obs_select_min"]))
            catalogue_obs = szifi.get_catalogue_indices(catalogue_obs,indices)

            self.catalogue = {}
            self.catalogue_patch = {}

            n_clusters = len(catalogue_obs.catalogue["q_opt"])
            self.catalogue["q_szifi_val"] = catalogue_obs.catalogue["q_opt"]
            self.catalogue["z"] = catalogue_obs.catalogue["z"]

            if self.cnc_params["catalogue_params"]["downsample"] == False:

                self.catalogue_patch["q_szifi_val"] = np.zeros(n_clusters)

            elif self.cnc_params["catalogue_params"]["downsample"] == True:

                #indices_mapping = np.load(path + "indices_mapping_" + suffix + "_val_down20_" + it_label + "_bias062_theta32_" + suffix_merge + ".npy")
                indices_mapping = np.load(path + "indices_mapping_" + suffix + "_val_down20_" + it_label + "_bias062_" + theta_label + "_" + suffix_merge + ".npy")

                self.catalogue_patch["q_szifi_val"] = indices_mapping[catalogue_obs.catalogue["pixel_ids"].astype(np.int64)]


        elif self.catalogue_name[0:20] == "zc19_simulated_paper":

            catalogue = np.load(root_path + "data/catalogues_sim/catalogue_" + self.catalogue_name + ".npy",allow_pickle=True)[0]
            self.catalogue = {}
            self.catalogue_patch = {}

            self.catalogue["q_mmf3_mean"] = catalogue["q_mmf3_mean"]
            self.catalogue_patch["q_mmf3_mean"] = catalogue["q_mmf3_mean_patch"]
            self.catalogue["p_zc19_mean"] = catalogue["p_zc19_mean"]
            self.catalogue_patch["p_zc19_mean"] = catalogue["p_zc19_mean_patch"]
            self.catalogue["z"] = catalogue["z"]

        elif self.catalogue_name[0:14] == "zc19_simulated":

            catalogue = np.load(root_path + "data/catalogues_sim/catalogue_" + self.catalogue_name + "_paper.npy",allow_pickle=True)[0]

            self.catalogue = {}
            self.catalogue_patch = {}

            self.catalogue["q_mmf3_mean"] = catalogue["q_mmf3_mean"]
            self.catalogue_patch["q_mmf3_mean"] = catalogue["q_mmf3_mean_patch"]
            self.catalogue["p_zc19"] = catalogue["p_zc19"]
            self.catalogue_patch["p_zc19"] = catalogue["p_zc19_patch"]
            self.catalogue["z"] = catalogue["z"]

            n_fd = 0
            self.catalogue["z"][0:n_fd] = [None]*n_fd
            self.catalogue["z_std"] = np.ones(len(self.catalogue["z"]))*1e-2
            self.catalogue["validated"] = np.ones(len(self.catalogue["z"])) #1. or 0.
            self.catalogue["validated"][0:n_fd] = np.zeros(n_fd)

            self.obs_select = "q_mmf3_mean"

            #Stacked CMB lensing data

            self.stacked_data_labels = ["p_zc19_stacked"]

            self.catalogue_patch["p_zc19_stacked"] = self.catalogue_patch["p_zc19"] #if one wants to use p with just one layer

            self.stacked_data = {"p_zc19_stacked":{}}

            self.stacked_data["p_zc19_stacked"]["data_vec"] = np.mean(self.catalogue["p_zc19"])
            self.stacked_data["p_zc19_stacked"]["inv_cov"] = float(len(self.catalogue["p_zc19"]))
            self.stacked_data["p_zc19_stacked"]["cluster_index"] = np.arange(len(self.catalogue["p_zc19"]))
            self.stacked_data["p_zc19_stacked"]["observable"] = "p_zc19"

            #False detections (to test false detection likelihood implementation)

            #Add non-validated clusters for clusters with q < 7

            if "non_val" in self.cnc_params["catalogue_params"]:

                if self.cnc_params["catalogue_params"]["non_val"] == True:

                    N_td_nonval = self.cnc_params["catalogue_params"]["N_td_nonval"]
                    N_fd = self.cnc_params["catalogue_params"]["N_fd"]

                    np.random.seed(seed=0)

                    #indices = np.where(self.catalogue["q_mmf3_mean"] < 7.)[0]
                    indices = np.arange(len(self.catalogue["q_mmf3_mean"]))

                    indices_nonval = np.random.choice(indices,N_td_nonval,replace=False)

                    self.catalogue["validated"][indices_nonval] = np.zeros(N_td_nonval)
                    self.catalogue["z"][indices_nonval] = np.array([float('nan')]*N_td_nonval)

                    f_v = (len(self.catalogue["z"])-N_td_nonval)/len(self.catalogue["z"])

                    #Add false detections

                    q_vec = np.linspace(6.,10.,self.cnc_params["n_points"])
                    pdf_fd = np.exp(-(q_vec-3.)**2/1.5**2)
                    pdf_fd = pdf_fd/integrate.simps(pdf_fd,q_vec)
                    self.pdf_false_detection = [q_vec,pdf_fd]

                    q_fd = rejection_sample_1d(q_vec,pdf_fd,N_fd)

                    self.catalogue["q_mmf3_mean"] = np.concatenate((self.catalogue["q_mmf3_mean"],q_fd))
                    self.catalogue_patch["q_mmf3_mean"] = np.concatenate((self.catalogue_patch["q_mmf3_mean"],np.zeros(len(q_fd))))
                    self.catalogue["z"] = np.concatenate((self.catalogue["z"],np.array([float('nan')]*N_fd)))
                    self.catalogue["z_std"] = np.concatenate((self.catalogue["z_std"],np.array([float('nan')]*N_fd)))
                    self.catalogue["p_zc19"] = np.concatenate((self.catalogue["p_zc19"],np.array([float('nan')]*N_fd)))
                    self.catalogue_patch["p_zc19"] = np.concatenate((self.catalogue_patch["p_zc19"],np.array([float('nan')]*N_fd)))

                    self.catalogue["validated"] = np.concatenate((self.catalogue["validated"],np.zeros(N_fd)))

                    if self.cnc_params["catalogue_params"]["none_validated"] == True:

                        self.catalogue["validated"] = np.zeros(len(self.catalogue["validated"]))
                        self.catalogue["z"] = np.array([float('nan')]*len(self.catalogue["z"]))
                        f_v = 0.

                    self.cnc_params["f_true_validated"] = f_v

        elif self.catalogue_name == "zc19_lensboosted_simulated" or self.catalogue_name == "zc19_lensboosted40_simulated":

            if self.catalogue_name == "zc19_lensboosted_simulated":

                catalogue = np.load(root_path + "data/catalogues_sim/catalogue_zc19_simulated_3_alens1.npy",allow_pickle=True)[0]

            elif self.catalogue_name == "zc19_lensboosted40_simulated":

                catalogue = np.load(root_path + "data/catalogues_sim/catalogue_zc19_simulated_3_alens40.npy",allow_pickle=True)[0]

            self.catalogue = {}
            self.catalogue_patch = {}

            self.catalogue["q_mmf3_mean"] = catalogue["q_mmf3_mean"]
            self.catalogue_patch["q_mmf3_mean"] = catalogue["q_mmf3_mean_patch"]
            self.catalogue["p_zc19"] = catalogue["p_zc19"]
            self.catalogue_patch["p_zc19"] = catalogue["p_zc19_patch"]
            self.catalogue["z"] = catalogue["z"]

            n_fd = 0
            self.catalogue["z"][0:n_fd] = [None]*n_fd
            self.catalogue["z_std"] = np.ones(len(self.catalogue["z"]))*1e-2
            self.catalogue["validated"] = np.ones(len(self.catalogue["z"])) #1. or 0.
            self.catalogue["validated"][0:n_fd] = np.zeros(n_fd)

            self.catalogue["M"] = catalogue["M"]

            np.random.seed(seed=0)
            M_lens = np.exp(np.log(self.catalogue["M"]) + np.random.normal(scale=0.2,size=len(self.catalogue["M"]))) + np.random.normal(scale=0.5,size=len(self.catalogue["M"]))

            self.catalogue["m_lens"] = M_lens
            self.catalogue_patch["m_lens"] = np.zeros(len(self.catalogue["m_lens"]))

            self.obs_select = "q_mmf3_mean"

            #Stacked CMB lensing data

            self.stacked_data_labels = ["p_zc19_stacked"]

            self.catalogue_patch["p_zc19_stacked"] = self.catalogue_patch["p_zc19"] #if one wants to use p with just one layer

            self.stacked_data = {"p_zc19_stacked":{}}

            self.stacked_data["p_zc19_stacked"]["data_vec"] = np.mean(self.catalogue["p_zc19"])
            self.stacked_data["p_zc19_stacked"]["inv_cov"] = float(len(self.catalogue["p_zc19"]))
            self.stacked_data["p_zc19_stacked"]["cluster_index"] = np.arange(len(self.catalogue["p_zc19"]))
            self.stacked_data["p_zc19_stacked"]["observable"] = "p_zc19"

        elif self.catalogue_name[0:11] == "SO_sim_sbi_":

            catalogue = np.load("/rds-d4/user/iz221/hpc-work/catalogues_so_sbi/catalogue_so_simulated_sbi_" + str(self.catalogue_name[11:]) + ".npy",allow_pickle=True)[0]

            self.catalogue = {}
            self.catalogue["q_so_sim"] = catalogue["q_so_sim"]
            self.catalogue["z"] = catalogue["z"]
            self.catalogue["z_std"] = np.zeros(len(self.catalogue["z"]))
            self.catalogue["p_so_sim"] = catalogue["p_so_sim"]

            print("Min max q",np.min(catalogue["q_so_sim"]),np.max(catalogue["q_so_sim"]))
            print("Min max z",np.min(catalogue["z"]),np.max(catalogue["z"]))
            print("N clusters",len(catalogue["z"]))

            self.catalogue_patch = {}
            self.catalogue_patch["q_so_sim"] = catalogue["q_so_sim_patch"]
            self.catalogue_patch["p_so_sim"] = catalogue["p_so_sim_patch"]

            self.M = catalogue["M"]

            #Stacked CMB lensing

            self.stacked_data_labels = ["p_so_sim_stacked"]

            self.catalogue_patch["p_so_sim_stacked"] = np.zeros(len(self.catalogue["p_so_sim"])) #if one wants to use p with just one layer
            self.stacked_data = {"p_so_sim_stacked":{}}

            self.stacked_data["p_so_sim_stacked"]["data_vec"] = np.mean(self.catalogue["p_so_sim"])
            self.stacked_data["p_so_sim_stacked"]["inv_cov"] = float(len(self.catalogue["p_so_sim"]))
            self.stacked_data["p_so_sim_stacked"]["cluster_index"] = np.arange(len(self.catalogue["z"]))
            self.stacked_data["p_so_sim_stacked"]["observable"] = "p_so_sim"

        elif self.catalogue_name[0:14] == "SO_sim_goal3yr":

            catalogue = np.load("/home/iz221/cnc/data/catalogues_sim/catalogue_so_simulated_" + str(self.catalogue_name[15:]) + "_goal3yr.npy",allow_pickle=True)[0]

            self.catalogue = {}
            self.catalogue["q_so_goal3yr_sim"] = catalogue["q_so_goal3yr_sim"]
            self.catalogue["z"] = catalogue["z"]
            self.catalogue["z_std"] = np.zeros(len(self.catalogue["z"]))
            self.catalogue["p_so_goal3yr_sim"] = catalogue["p_so_goal3yr_sim"]

            self.catalogue_patch = {}
            self.catalogue_patch["q_so_goal3yr_sim"] = catalogue["q_so_goal3yr_sim_patch"]
            self.catalogue_patch["p_so_goal3yr_sim"] = catalogue["p_so_goal3yr_sim_patch"]

            self.M = catalogue["M"]

            #Stacked CMB lensing

            self.stacked_data_labels = ["p_so_sim_stacked"]

            self.catalogue_patch["p_so_goal3yr_sim_stacked"] = np.zeros(len(self.catalogue["p_so_goal3yr_sim"])) #if one wants to use p with just one layer
            self.stacked_data = {"p_so_goal3yr_sim_stacked":{}}

            self.stacked_data["p_so_goal3yr_sim_stacked"]["data_vec"] = np.mean(self.catalogue["p_so_goal3yr_sim"])
            self.stacked_data["p_so_goal3yr_sim_stacked"]["inv_cov"] = float(len(self.catalogue["p_so_goal3yr_sim"]))
            self.stacked_data["p_so_goal3yr_sim_stacked"]["cluster_index"] = np.arange(len(self.catalogue["z"]))
            self.stacked_data["p_so_goal3yr_sim_stacked"]["observable"] = "p_so_goal3yr_sim"

        elif self.catalogue_name[0:7] == "SO_sim_":

            catalogue = np.load(root_path + "data/catalogues_sim/catalogue_so_simulated_" + str(self.catalogue_name[7:]) + "_simple.npy",allow_pickle=True)[0]

            self.catalogue = {}
            self.catalogue["q_so_sim"] = catalogue["q_so_sim"]
            self.catalogue["z"] = catalogue["z"]
            self.catalogue["z_std"] = np.zeros(len(self.catalogue["z"]))*1e-2
            self.catalogue["p_so_sim"] = catalogue["p_so_sim"]

            self.catalogue_patch = {}
            self.catalogue_patch["q_so_sim"] = catalogue["q_so_sim_patch"]
            self.catalogue_patch["p_so_sim"] = catalogue["p_so_sim_patch"]

            self.M = catalogue["M"]

            #Stacked CMB lensing

            self.stacked_data_labels = ["p_so_sim_stacked"]

            self.catalogue_patch["p_so_sim_stacked"] = np.zeros(len(self.catalogue["p_so_sim"])) #if one wants to use p with just one layer
            self.stacked_data = {"p_so_sim_stacked":{}}

            self.stacked_data["p_so_sim_stacked"]["data_vec"] = np.mean(self.catalogue["p_so_sim"])
            self.stacked_data["p_so_sim_stacked"]["inv_cov"] = float(len(self.catalogue["p_so_sim"]))
            self.stacked_data["p_so_sim_stacked"]["cluster_index"] = np.arange(len(self.catalogue["z"]))
            self.stacked_data["p_so_sim_stacked"]["observable"] = "p_so_sim"

            if "non_val" in self.cnc_params["catalogue_params"]:

                self.catalogue["validated"] = np.ones(len(self.catalogue["p_so_sim"]))

                if self.cnc_params["catalogue_params"]["non_val"] == True:

                    #Add some non-validated true detections

                    N_td_nonval = self.cnc_params["catalogue_params"]["N_td_nonval"]
                    N_fd = self.cnc_params["catalogue_params"]["N_fd"]

                    np.random.seed(seed=1)

                    indices = np.arange(len(self.catalogue["q_so_sim"]))
                    indices_nonval = np.random.choice(indices,N_td_nonval,replace=False)

                    self.catalogue["validated"][indices_nonval] = np.zeros(N_td_nonval)
                    self.catalogue["z"][indices_nonval] = np.array([float('nan')]*N_td_nonval)

                    f_v = (len(self.catalogue["z"])-N_td_nonval)/len(self.catalogue["z"])

                    #Add false detections

                    q_vec = np.linspace(5.,10.,self.cnc_params["n_points"])
                    pdf_fd = np.exp(-(q_vec-3.)**2/1.5**2)
                    pdf_fd = pdf_fd/integrate.simps(pdf_fd,q_vec)
                    self.pdf_false_detection = [q_vec,pdf_fd]

                    q_fd = rejection_sample_1d(q_vec,pdf_fd,N_fd)

                    self.catalogue["q_so_sim"] = np.concatenate((self.catalogue["q_so_sim"],q_fd))
                    self.catalogue_patch["q_so_sim"] = np.concatenate((self.catalogue_patch["q_so_sim"],np.zeros(len(q_fd))))
                    self.catalogue["z"] = np.concatenate((self.catalogue["z"],np.array([float('nan')]*N_fd)))
                    self.catalogue["p_so_sim"] = np.concatenate((self.catalogue["p_so_sim"],np.array([float('nan')]*N_fd)))
                    self.catalogue_patch["p_so_sim"] = np.concatenate((self.catalogue_patch["p_so_sim"],np.array([float('nan')]*N_fd)))

                    self.catalogue["validated"] = np.concatenate((self.catalogue["validated"],np.zeros(N_fd)))

                    if self.cnc_params["catalogue_params"]["none_validated"] == True:

                        self.catalogue["validated"] = np.zeros(len(self.catalogue["validated"]))
                        self.catalogue["z"] = np.array([float('nan')]*len(self.catalogue["z"]))
                        f_v = 0.

                    self.cnc_params["f_true_validated"] = f_v


        elif self.catalogue_name[0:12] == "SO_sim_mass_":

            catalogue = np.load(root_path + "data/catalogues_sim/catalogue_so_simulated_mass_" + str(self.catalogue_name[12:]) + ".npy",allow_pickle=True)[0]

            self.catalogue = {}
            self.catalogue["ln_M"] = catalogue["ln_M"]
            self.catalogue["z"] = catalogue["z"]
            self.catalogue["z_std"] = np.zeros(len(self.catalogue["z"]))
            #self.catalogue["p_so_sim"] = catalogue["p_so_sim"]

            self.catalogue_patch = {}
            self.catalogue_patch["ln_M"] = catalogue["ln_M"]
        #    self.catalogue_patch["p_so_sim_simple"] = catalogue["p_so_sim_patch"]


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

        elif self.catalogue_name[0:12] == "planck_szifi":

            import sys
            sys.path.insert(0,'/home/iz221/planck_sz/')
            import cat

            path = "/rds-d4/user/iz221/hpc-work/catalogues_def/planck_hpc/"

            mask_type = self.cnc_params["catalogue_params"]["mask_type"]

            if mask_type == "cosmology_2":

                (catalogue_master) = np.load(path + "master_catalogue_planck_withfixed_planckcibparams_withnoit_cross_validated_cosmologymask_withdownsamplednoise.npy",allow_pickle=True)[()]

            if mask_type == "cosmology_3":

                (catalogue_master) = np.load(path + "master_catalogue_planck_withfixed_planckcibparams_withnoit_cross_validated_cosmologymask3_withdownsamplednoise.npy",allow_pickle=True)[()]

            dep_type = self.catalogue_name[13:]

            self.catalogue = {}
            self.catalogue_patch = {}

            q_szifi = catalogue_master.catalogue_master.catalogue["q_opt_" + dep_type]
            z = catalogue_master.catalogue_master.catalogue["redshift"]

            q_th = self.cnc_params["obs_select_min"]

            validated = catalogue_master.catalogue_master.catalogue["validated"]

            include_non_validated = self.cnc_params["catalogue_params"]["include_non_validated"]

            if include_non_validated == True:

                indices = np.where((q_szifi > q_th))

            elif include_non_validated == False:

                indices = np.where((q_szifi > q_th) & (validated > 0.))

            self.catalogue["q_szifi"] = q_szifi[indices]
            self.catalogue["z"] = z[indices]

            self.catalogue["z"][np.where(self.catalogue["z"] < 0.)] = None

            self.catalogue["z_std"] = np.zeros(len(indices[0]))

            downsample_flag = self.cnc_params["catalogue_params"]["downsample"]

            if downsample_flag == False:

                sigma_matrix = catalogue_master.sigma_matrices[dep_type]
                q_patches = catalogue_master.catalogue_master.catalogue["pixel_ids"][indices]
                skyfracs = catalogue_master.skyfracs["cosmology"]

            elif downsample_flag == True:

                sigma_matrix = catalogue_master.sigma_matrices[dep_type + "_downnoisebased"]
                pixel_ids_original = catalogue_master.catalogue_master.catalogue["pixel_ids"][indices]
                original_tile_vec = catalogue_master.original_tile_vecs["cosmology" + "_" + dep_type + "_downnoisebased"]
                skyfracs = catalogue_master.skyfracs["cosmology" + "_" + dep_type + "_downnoisebased"]

                q_patches = np.zeros(len(indices[0]))

                for i in range(0,len(indices[0])):

                    q_patches[i] = original_tile_vec[int(pixel_ids_original[i])]

    #        indices_sigma = np.where(skyfracs > 1e-10)
            indices_sigma = np.where(skyfracs > 2e-4)


            q_patches_new = np.zeros(len(indices[0]))

            for i in range(0,len(indices[0])):

                q_patches_new[i] = np.where(indices_sigma[0] == int(q_patches[i]))[0]

            self.catalogue_patch["q_szifi"] = q_patches_new

        elif self.catalogue_name == "SPT2500d":

            # here are some spt-specific  quantities.
            # these are pasted from Bocquet's SPT_SZ_cluster_likelihood/SPTcluster_data.py
            SPTfieldNames = ('ra5h30dec-55', 'ra23h30dec-55', 'ra21hdec-60', 'ra3h30dec-60',
                             'ra21hdec-50', 'ra4h10dec-50', 'ra0h50dec-50', 'ra2h30dec-50',
                             'ra1hdec-60', 'ra5h30dec-45', 'ra6h30dec-55', 'ra3h30dec-42.5',
                             'ra23hdec-62.5','ra21hdec-42.5', 'ra1hdec-42.5', 'ra22h30dec-55',
                             'ra23hdec-45','ra6h30dec-45', 'ra6hdec-62.5')


            SPTdoubleCount = [('SPT-CLJ0000-5748', 'ra1hdec-60'),
                            ('SPT-CLJ0001-5440', 'ra23h30dec-55'),
                            ('SPT-CLJ0047-4506', 'ra0h50dec-50'),
                            ('SPT-CLJ0344-5452', 'ra3h30dec-60'),
                            ('SPT-CLJ0426-5455', 'ra4h10dec-50'),
                            ('SPT-CLJ2040-4451', 'ra21hdec-42.5'),
                            ('SPT-CLJ2056-5459', 'ra21hdec-50'),
                            ('SPT-CLJ2232-5959', 'ra23hdec-62.5')]

            SPTnFalse_alpha = (16.79, 17.58, 25.64, 20.53, 25.28, 16.75, 20.76, 14.98,
                17.25, 15.91, 17.77, 16.85, 14.90, 17.11, 18.41, 16.45, 17.00, 14.78, 16.53)

            SPTnFalse_beta = (4.60, 4.03, 4.07, 4.70, 4.14, 5.48, 5.11, 4.78, 4.38, 4.81,
                4.58, 4.31, 4.92, 4.49, 5.55, 5.23, 5.20, 4.23, 4.70)

            # now we load the catalogue
            SPTcatalogfile = root_path + "data/spt/SPT2500d.fits"
            spt_catalog = Table.read(SPTcatalogfile)

            threshold = self.cnc_params['obs_select_min']
            indices_catalog = []
            bounds_vec = []

            for i in range(0,len(spt_catalog["xi"])):
                ## add a seperate zmin threshold, for observed clusters.
                if (spt_catalog["xi"][i] > threshold):

                    if spt_catalog['redshift'][i]>self.cnc_params["z_min"]:

                        indices_catalog.append(i)
                        bounds_vec.append(False)

                    elif spt_catalog['redshift'][i] < 1e-8:

                        indices_catalog.append(i)
                        bounds_vec.append(True)

            self.catalogue["z_bounds"] = bounds_vec
            self.catalogue["low_z"] = np.asarray(spt_catalog['redshift_lim'][indices_catalog])
            self.catalogue["up_z"] = np.ones(len(indices_catalog))*self.cnc_params["z_max"]

            validated_vec = [True]*len(indices_catalog)

            for i in range(0,len(validated_vec)):

                if self.catalogue["low_z"][i] < 1e-8:

                    validated_vec[i] = False

            self.catalogue["validated"] = validated_vec

            self.catalogue["z"] = np.asarray(spt_catalog['redshift'][indices_catalog])
            indices_no_z = np.where(self.catalogue["z"] < 1e-8)[0]
            self.catalogue["z"][indices_no_z] = None

            self.catalogue["z_std"] = np.asarray(spt_catalog['redshift_err'][indices_catalog])
            self.catalogue["xi"] = np.asarray(spt_catalog['xi'][indices_catalog])

            self.catalogue_patch['xi'] = np.zeros(len(self.catalogue['xi'])).astype(np.int64)
            for id,field in enumerate(spt_catalog['field'][indices_catalog]):
                self.catalogue_patch['xi'][id] = SPTfieldNames.index(field)

            if 'Yx' in self.observables[0]:

                self.catalogue["Yx"] = np.asarray(spt_catalog['Yx_fid'][indices_catalog])
                self.catalogue["Yx_std"] = np.asarray(spt_catalog['Yx_err'][indices_catalog])
                self.catalogue["r500"] = np.asarray(spt_catalog['r500'][indices_catalog])


                indices_no_Yx = np.where(self.catalogue["Yx"] <= 0.)[0]
                self.catalogue["Yx"][indices_no_Yx] = None
                self.catalogue_patch["Yx"] = np.arange(len(self.catalogue["Yx"])).astype(np.int64)# index of all clusters.


            if 'WLMegacam' or 'WLHST' in self.observables[0]:
                # WL simulation calibration data --  same as Bocquet's code
                # WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
                WLsimcalibfile = root_path + "data/spt/WLsimcalib_data.py"
                import warnings
                from contextlib import contextmanager

                @contextmanager
                def suppress_warnings():
                    warnings.filterwarnings("ignore")
                    try:
                        yield
                    finally:
                        warnings.resetwarnings()
                with suppress_warnings():
                    import imp
                WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)

                self.WLcalib = WLsimcalib.WLcalibration
                spt_catalog['WLdata'] = [None for i in range(len(spt_catalog['SPT_ID']))]

                # exit(0)

            if 'WLMegacam' in self.observables[0]:
                spt_catalog['WLMegacam'] = [None for i in range(len(spt_catalog['SPT_ID']))]
                spt_catalog['WLMegacam_std'] = [None for i in range(len(spt_catalog['SPT_ID']))]

                # --  same as Bocquet's code
                self.MegacamDir = root_path + "data/spt/Megacam"
                for i,name in enumerate(spt_catalog['SPT_ID']):
                    prefix = self.MegacamDir+'/'+name+'/'+name
                    if os.path.isfile(prefix+'_shear.txt'):
                        shear = np.loadtxt(prefix+'_shear.txt', unpack=True)
                        Nz = np.loadtxt(prefix+'_Nz.txt', unpack=True)
                        spt_catalog['WLMegacam'][i] = shear[1]

                        spt_catalog['WLMegacam_std'][i] = shear[2]
                        spt_catalog['WLdata'][i] = {'datatype':'Megacam',
                                                       'r_deg':shear[0],
                                                       'shear':shear[1],
                                                       'shearerr':shear[2],
                                                       'redshifts':Nz[0],
                                                       'Nz':Nz[1],
                                                       'Ntot':np.sum(Nz[1]),
                                                       'massModelErr': (self.WLcalib['MegacamSim'][1]**2 + self.WLcalib['MegacamMcErr']**2 + self.WLcalib['MegacamCenterErr']**2)**.5,
                                                       'zDistShearErr': (self.WLcalib['MegacamzDistErr']**2 + self.WLcalib['MegacamShearErr']**2)**.5}

            # if 'WLMegacam' or 'WLHST' in self.observables[0]:

                self.catalogue['WLMegacam'] = np.asarray(list(map(lambda x: np.nan if x is None else x, spt_catalog['WLMegacam'][indices_catalog])))
                self.catalogue['WLMegacam_std'] = np.asarray(list(map(lambda x: np.nan if x is None else x, spt_catalog['WLMegacam_std'][indices_catalog])))
                self.catalogue_patch['WLMegacam'] = np.arange(len(self.catalogue['WLMegacam'])).astype(np.int64)# index of all clusters.


            if 'WLHST' in self.observables[0]:

                spt_catalog['WLHST'] = [None for i in range(len(spt_catalog['SPT_ID']))]
                spt_catalog['WLHST_std'] = [None for i in range(len(spt_catalog['SPT_ID']))]
                prefix =  root_path + "data/spt/"
                self.HSTfile = prefix +'hst_20160930_xray.pkl'
                HSTdata = pickle.load(open(self.HSTfile, 'rb'),encoding='latin1') ## added 16/05/23
                for i,name in enumerate(spt_catalog['SPT_ID']):
                    if name in HSTdata.keys():
                        Ntot, pzs = {}, {}
                        for j in HSTdata[name]['pzs'].keys():
                            pzs[j] = np.sum(HSTdata[name]['pzs'][j], axis=0)
                            Ntot[j] = np.sum(HSTdata[name]['pzs'][j])
                        spt_catalog['WLHST'][i] = HSTdata[name]['shear']
                        spt_catalog['WLHST_std'][i] = HSTdata[name]['shearerr']

                        spt_catalog['WLdata'][i] = {'datatype':'HST',
                                                'center':HSTdata[name]['center'],
                                                'r_deg':HSTdata[name]['r_deg'],
                                                'shear':HSTdata[name]['shear'],
                                                'shearerr':HSTdata[name]['shearerr'],
                                                'magbinids':HSTdata[name]['magbinids'],
                                                'redshifts':HSTdata[name]['redshifts'],
                                                'pzs':pzs,
                                                'magcorr':HSTdata[name]['magnificationcorr'],
                                                'Ntot':Ntot,
                                                'massModelErr': (self.WLcalib['HSTsim'][name][1]**2 + self.WLcalib['HSTmcErr']**2 + self.WLcalib['HSTcenterErr']**2)**.5,
                                                'zDistShearErr': (self.WLcalib['HSTzDistErr']**2 + self.WLcalib['HSTshearErr']**2)**.5}

                self.catalogue['WLHST'] = np.asarray(list(map(lambda x: np.nan if x is None else x, spt_catalog['WLHST'][indices_catalog])))
                self.catalogue['WLHST_std'] = np.asarray(list(map(lambda x: np.nan if x is None else x, spt_catalog['WLHST_std'][indices_catalog])))
                self.catalogue_patch['WLHST'] = np.arange(len(self.catalogue['WLHST'])).astype(np.int64)# index of all clusters.
                self.catalogue['SPT_ID'] = np.asarray(list(map(lambda x: np.nan if x is None else x, spt_catalog['SPT_ID'][indices_catalog])))

            if 'WLMegacam' or 'WLHST' in self.observables[0]:

                self.catalogue['WLdata'] = np.asarray(spt_catalog['WLdata'][indices_catalog])

        elif self.catalogue_name[0:11] == "act_dr5_sim":

            #SZ data from ACT

            threshold = 5.

            # act_cat = np.loadtxt(root_path + "data/act_dr5/SZ_cat_nemosimkit_130923.txt").transpose()
            act_cat = np.loadtxt(root_path + "data/SZ_cat_nemosimkit_19jul24_100bins_sim1.txt").transpose()
            data_act = {}
            data_act["SNR"] = act_cat[2]
            data_act["z"] = act_cat[0]

            indices_act = []

            for i in range(0,len(data_act["SNR"])):

                if (data_act["SNR"][i] > threshold):

                    indices_act.append(i)

            observable = self.obs_select

            self.catalogue[observable] = data_act["SNR"][indices_act]
            self.catalogue["z"] = data_act["z"][indices_act]
            self.catalogue_patch[observable] = np.zeros(len(self.catalogue[observable])).astype(np.int64)

            indices_no_z = np.where(self.catalogue["z"] < 0.)[0]

            self.catalogue["z"][indices_no_z] = None
            self.catalogue["z_std"] = np.zeros(len(self.catalogue["z"]))

            indices_z = np.argwhere(~np.isnan(self.catalogue["z"]))[:,0]

            if self.obs_select == "q_act":

                # patch_index_vec = np.load(root_path + "data/act_dr5/SZ_cat_nemosimkit_cluster_patch_indices.npy")
                patch_index_vec = np.load(root_path + "data/SZ_cat_nemosimkit_cluster_patch_indices_19jul24_100bins_sim1.npy")
                patch_index_vec = patch_index_vec[indices_act]
                self.catalogue_patch[observable][indices_z] = patch_index_vec[indices_z]



        #End of catalogue names

        if self.precompute_cnc_quantities == True:

            self.get_precompute_cnc_quantities()


        self.n_clusters = len(self.catalogue[self.obs_select])


    def bin_number_counts(self):

            #Compute binned number counts

            if self.cnc_params["binned_lik_type"] == "obs_select":

                self.number_counts = np.histogram(self.catalogue[self.obs_select],bins=self.bins_obs_select_edges)[0]

            elif self.cnc_params["binned_lik_type"] == "z":

                self.number_counts = np.histogram(self.catalogue["z"],bins=self.bins_z_edges)[0]

            elif self.cnc_params["binned_lik_type"] == "z_and_obs_select":

                self.number_counts = np.zeros((len(self.bins_z_edges)-1,len(self.bins_obs_select_edges)-1))

                for i in range(0,len(self.bins_z_edges)-1):

                    for j in range(0,len(self.bins_obs_select_edges)-1):

                        indices = np.where((self.catalogue[self.obs_select] > self.bins_obs_select_edges[j]) & (self.catalogue[self.obs_select] < self.bins_obs_select_edges[j+1])
                        & (self.catalogue["z"] > self.bins_z_edges[i]) & (self.catalogue["z"] < self.bins_z_edges[i+1]))[0]
                        self.number_counts[i,j] = len(indices)


    def get_precompute_cnc_quantities(self):

        self.indices_no_z = np.argwhere(np.isnan(self.catalogue["z"]))[:,0]
        self.indices_with_z = np.argwhere(~np.isnan(self.catalogue["z"]))[:,0]

        if self.cnc_params["non_validated_clusters"] == True:

            self.n_val = len(np.where(self.catalogue["validated"] > 0.5)[0])

        self.n_tot = len(self.catalogue["z"])

        #Bin number counts

        self.bin_number_counts()

        #Precompute other quantities

        self.obs_select_max = np.max(self.catalogue[self.obs_select])

        self.observable_dict = {}
        self.indices_obs_select = []
        self.indices_other_obs = []

        for i in self.indices_with_z:

            observables_cluster = []

            for observable_set in self.observables:

                observable_set_cluster = []

                for observable in observable_set:

                    if np.any(np.isnan(self.catalogue[observable][i])) == False:
                    #if np.isnan(self.catalogue[observable][i]) == False:

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
