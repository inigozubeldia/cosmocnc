import numpy as np
import pylab as pl
from astropy.io import fits
from astropy.table import Table
from .config import *
import imp
import pickle


class cluster_catalogue:

    def __init__(self,catalogue_name="Planck_MMF3_cosmo",
                 precompute_cnc_quantities=True,
                 bins_obs_select_edges=np.linspace(0.01,1.01,11),
                 bins_z_edges=np.exp(np.linspace(np.log(6.),np.log(100),6)),
                 observables=None, # "q_mmf3_mean","p_zc19", etc
                 obs_select=None,
                 cnc_params=None):

        self.catalogue_name = catalogue_name
        self.catalogue = {}
        self.catalogue_patch = {}
        self.precompute_cnc_quantities = precompute_cnc_quantities
        self.cnc_params = cnc_params

        if isinstance(bins_obs_select_edges,str):
            bins_obs_select_edges = eval(bins_obs_select_edges)
        if isinstance(bins_z_edges,str):
            bins_z_edges = eval(bins_z_edges)
        # print(bins_z_edges)
        # print(observables)
        if isinstance(observables,str):
            observables = eval(observables)
        # print(observables[0])
        # exit(0)

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
            # print('loading simulated catalogue')
            catalogue = np.load(root_path + "data/catalogues_sim/catalogue_" + self.catalogue_name + "_paper.npy",allow_pickle=True)[0]

            self.catalogue = {}
            self.catalogue_patch = {}

            self.catalogue["q_mmf3_mean"] = catalogue["q_mmf3_mean"]
            self.catalogue_patch["q_mmf3_mean"] = catalogue["q_mmf3_mean_patch"]
            self.catalogue["p_zc19"] = catalogue["p_zc19"]
            self.catalogue_patch["p_zc19"] = catalogue["p_zc19_patch"]
            self.catalogue["z"] = catalogue["z"]
            self.catalogue["z_std"] = np.ones(len(self.catalogue["z"]))*1e-2

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

        elif self.catalogue_name == "SPT2500d":
            # print('loading spt catalogue')
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
            for i in range(0,len(spt_catalog["xi"])):
                ## add a seperate zmin threshold, for observed clusters.
                if (spt_catalog["xi"][i] > threshold) and (spt_catalog['redshift'][i]>self.cnc_params["z_min"]):
                    indices_catalog.append(i)
            # exit(0)



            self.catalogue["z"] = np.asarray(spt_catalog['redshift'][indices_catalog])
            # print('z:',self.catalogue["z"][:10])
            indices_no_z = np.where(self.catalogue["z"] == 0.)[0]

            # print('indices_no_z',indices_no_z)
            self.catalogue["z"][indices_no_z] = None

            self.catalogue["z_std"] = np.asarray(spt_catalog['redshift_err'][indices_catalog])
            # print('z_std:',len(self.catalogue["z_std"]))
            # print('z_std:',self.catalogue["z_std"][:10])


            self.catalogue["xi"] = np.asarray(spt_catalog['xi'][indices_catalog])

            # print('xi:',len(self.catalogue["xi"]))
            # print('xi:',self.catalogue["xi"][:10])

            self.catalogue_patch['xi'] = np.zeros(len(self.catalogue['xi'])).astype(np.int)
            for id,field in enumerate(spt_catalog['field'][indices_catalog]):
                self.catalogue_patch['xi'][id] = SPTfieldNames.index(field)
            # print('patches xi:',len(self.catalogue_patch["xi"]))
            # print('patches xi:',self.catalogue_patch["xi"][:10])

            # print('self.obs_select',self.obs_select)

            # print('threshold =',self.cnc_params['obs_select_threshold'])
            # exit(0)

            # print(self.observables)
            # exit(0)
            if 'Yx' in self.observables[0]:
                # print('adding Yx data')
                self.catalogue["Yx"] = np.asarray(spt_catalog['Yx_fid'][indices_catalog])
                # print()
                self.catalogue["Yx_std"] = np.asarray(spt_catalog['Yx_err'][indices_catalog])

                indices_no_Yx = np.where(self.catalogue["Yx"] <= 0.)[0]
                self.catalogue["Yx"][indices_no_Yx] = None


                self.catalogue_patch["Yx"] = np.arange(len(self.catalogue["Yx"])).astype(np.int)# index of all clusters.
                # print(self.catalogue["Yx"])
                # exit(0)
                # print(self.catalogue["Yx_std"])
                # print(self.catalogue_patch["Yx"])
                # exit(0)

            if 'WLMegacam' or 'WLHST' in self.observables[0]:
                # WL simulation calibration data --  same as Bocquet's code
                # WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
                WLsimcalibfile = root_path + "data/spt/WLsimcalib_data.py"
                WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
                # print("WLsimcalib:",WLsimcalib)

                self.WLcalib = WLsimcalib.WLcalibration
                # print("WLcalib:",self.WLcalib)
                spt_catalog['WLdata'] = [None for i in range(len(spt_catalog['SPT_ID']))]

                # exit(0)

            if 'WLMegacam' in self.observables[0]:
                # print('collecting WLMegacam data')
                spt_catalog['WLMegacam'] = [None for i in range(len(spt_catalog['SPT_ID']))]
                spt_catalog['WLMegacam_std'] = [None for i in range(len(spt_catalog['SPT_ID']))]

                # --  same as Bocquet's code
                self.MegacamDir = root_path + "data/spt/Megacam"
                for i,name in enumerate(spt_catalog['SPT_ID']):
                    prefix = self.MegacamDir+'/'+name+'/'+name
                    if os.path.isfile(prefix+'_shear.txt'):
                        shear = np.loadtxt(prefix+'_shear.txt', unpack=True)
                        # print('loading shear data:',prefix+'_shear.txt')
                        Nz = np.loadtxt(prefix+'_Nz.txt', unpack=True)
                        spt_catalog['WLMegacam'][i] = shear[1]
                        # print('shear[1]',shear[1])
                        # exit(0)
                        spt_catalog['WLMegacam_std'][i] = shear[2]
                        # print('shear size',i,len(spt_catalog['WLMegacam_std'][i]))
                        spt_catalog['WLdata'][i] = {'datatype':'Megacam',
                                                       'r_deg':shear[0],
                                                       'shear':shear[1],
                                                       'shearerr':shear[2],
                                                       'redshifts':Nz[0],
                                                       'Nz':Nz[1],
                                                       'Ntot':np.sum(Nz[1]),
                                                       'massModelErr': (self.WLcalib['MegacamSim'][1]**2 + self.WLcalib['MegacamMcErr']**2 + self.WLcalib['MegacamCenterErr']**2)**.5,
                                                       'zDistShearErr': (self.WLcalib['MegacamzDistErr']**2 + self.WLcalib['MegacamShearErr']**2)**.5}

                # print(spt_catalog['WLdata'])
            # if 'WLMegacam' or 'WLHST' in self.observables[0]:
                # self.catalogue['WLMegacam'] = np.asarray(spt_catalog['WLMegacam'][indices_catalog])
                self.catalogue['WLMegacam'] = np.asarray(list(map(lambda x: np.nan if x is None else x, spt_catalog['WLMegacam'][indices_catalog])))
                self.catalogue['WLMegacam_std'] = np.asarray(list(map(lambda x: np.nan if x is None else x, spt_catalog['WLMegacam_std'][indices_catalog])))
                self.catalogue_patch['WLMegacam'] = np.arange(len(self.catalogue['WLMegacam'])).astype(np.int)# index of all clusters.
                # print(self.catalogue_patch['WLMegacam'])
                # print(self.catalogue['WLMegacam'])
                # print(np.shape(self.catalogue['WLMegacam_std']))
                # exit(0)

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
                        # print(name,len(spt_catalog['WLHST'][i]))
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
                self.catalogue_patch['WLHST'] = np.arange(len(self.catalogue['WLHST'])).astype(np.int)# index of all clusters.
                self.catalogue['SPT_ID'] = np.asarray(list(map(lambda x: np.nan if x is None else x, spt_catalog['SPT_ID'][indices_catalog])))

                # print(self.catalogue['WLHST'],self.catalogue['WLHST_std'])
                # print(self.catalogue['WLHST'],self.catalogue['WLHST_std'])
                # exit(0)


            if 'WLMegacam' or 'WLHST' in self.observables[0]:
                self.catalogue['WLdata'] = np.asarray(spt_catalog['WLdata'][indices_catalog])
                # print("self.catalogue['WLdata']",self.catalogue['WLdata'])
                # exit(0)
            # exit(0)



        self.n_clusters = len(self.catalogue[self.obs_select])
        # print('self.n_clusters',self.n_clusters)

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

                    # print( observable,type(self.catalogue[observable][i]))
                    # exit(0)

                    # if np.isnan(self.catalogue[observable][i]) == False:
                    if np.any(np.isnan(self.catalogue[observable][i])) == False: ## np.any to allow for observable being a set of numbers, e.g. shear

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
