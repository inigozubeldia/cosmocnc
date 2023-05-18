import numpy as np
import pylab as pl
import warnings
warnings.filterwarnings("ignore")
import cosmopower
import os
import subprocess
from cosmopower import cosmopower_NN
from cosmopower import cosmopower_PCAplusNN
from .config import *
import scipy.optimize as optimize

#cosmo_model = "lcdm", "mnu", "neff", "wcdm"

class cosmopower:

    def __init__(self,cosmo_model="lcdm"):

        #path_to_cosmopower_organization = path_to_cosmopower

        path_to_emulators = path_to_cosmopower_organization + cosmo_model + "/"
        str_cmd_subprocess = ["ls",path_to_emulators]

        emulator_dict = {}
        emulator_dict[cosmo_model] = {}

        emulator_dict[cosmo_model]['TT'] = 'TT_v1'
        emulator_dict[cosmo_model]['TE'] = 'TE_v1'
        emulator_dict[cosmo_model]['EE'] = 'EE_v1'
        emulator_dict[cosmo_model]['PP'] = 'PP_v1'
        emulator_dict[cosmo_model]['PKNL'] = 'PKNL_v1'
        emulator_dict[cosmo_model]['PKL'] = 'PKL_v1'
        emulator_dict[cosmo_model]['DER'] = 'DER_v1'
        emulator_dict[cosmo_model]['DAZ'] = 'DAZ_v1'
        emulator_dict[cosmo_model]['HZ'] = 'HZ_v1'
        emulator_dict[cosmo_model]['S8Z'] = 'S8Z_v1'

        #LOAD THE EMULATORS

        self.cp_tt_nn = {}
        self.cp_te_nn = {}
        self.cp_ee_nn = {}
        self.cp_pp_nn = {}
        self.cp_pknl_nn = {}
        self.cp_pkl_nn = {}
        #self.cp_der_nn = {}
        #self.cp_da_nn = {}
        #self.cp_h_nn = {}
        #self.cp_s8_nn = {}

        self.mp = cosmo_model

        path_to_emulators = path_to_cosmopower_organization + self.mp +'/'

        self.cp_tt_nn[self.mp] = cosmopower_NN(restore=True,
                                 restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[self.mp]['TT'])

        self.cp_te_nn[self.mp] = cosmopower_PCAplusNN(restore=True,
                                        restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[self.mp]['TE'])

        self.cp_ee_nn[self.mp] = cosmopower_NN(restore=True,
                                 restore_filename=path_to_emulators + 'TTTEEE/' + emulator_dict[self.mp]['EE'])

        self.cp_pp_nn[self.mp] = cosmopower_NN(restore=True,
                                 restore_filename=path_to_emulators + 'PP/' + emulator_dict[self.mp]['PP'])

        self.cp_pknl_nn[self.mp] = cosmopower_NN(restore=True,
                                   restore_filename=path_to_emulators + 'PK/' + emulator_dict[self.mp]['PKNL'])

        self.cp_pkl_nn[self.mp] = cosmopower_NN(restore=True,
                                  restore_filename=path_to_emulators + 'PK/' + emulator_dict[self.mp]['PKL'])

        self.cp_der_nn = cosmopower_NN(restore=True,restore_filename=path_to_emulators + 'derived-parameters/DER_v1',)

    def get_sigma_8(self):

        return self.cp_der_nn.ten_to_predictions_np(self.params_cp)[0][1]

    def set_cosmology(self,H0=67.37,Ob0=0.02233/0.6737**2,Oc0=0.1198/0.6737**2,ln10A_s=3.043,tau_reio=0.0540,n_s=0.9652): # LambdaCDM parameters last column of Table 1 of https://arxiv.org/pdf/1807.06209.pdf:

        h = H0/100.

        self.params_settings = {
                           'H0': H0,
                           'omega_b':Ob0*h**2,
                           'omega_cdm': Oc0*h**2,
                           'ln10^{10}A_s':ln10A_s,
                           'tau_reio': tau_reio,
                           'n_s': n_s,
                           }

        self.params_cp = {}

        for key,value in self.params_settings.items():

            self.params_cp[key] = [value]

    def find_As(self,sigma_8):

        params_cp = self.params_cp

        def to_root(ln10_10_As):

            params_cp["ln10^{10}A_s"] = ln10_10_As

            return self.cp_der_nn.ten_to_predictions_np(params_cp)[0][1]-sigma_8

        A_s = np.exp(optimize.root(to_root,x0=3.04,method="hybr").x)/1e10

        return A_s[0]

    def get_linear_power_spectrum(self,redshift):

        predicted_pkl_spectrum = {}
        params_cp_pk = self.params_cp.copy()

        params_cp_pk['z_pk_save_nonclass'] = [redshift]
        predicted_pkl_spectrum[str(redshift)] = self.cp_pkl_nn[self.mp].predictions_np(params_cp_pk)

        ndspl = 10

        k_arr = np.geomspace(1e-4,50.,5000)[::ndspl]
        ls = np.arange(2,5000+2)[::ndspl]
        dls = ls*(ls+1.)/2./np.pi

        pkl = 10.**np.asarray(predicted_pkl_spectrum[str(redshift)][0])
        pkl =  ((dls)**-1*pkl)

        #k_arr_re = np.exp(np.linspace(np.log(1e-4),np.log(5.),10000))
        #pkl_re = np.interp(k_arr_re,k_arr,pkl)

        k_arr_re = k_arr
        pkl_re = pkl
        # print('got pkl_re',pkl_re)

        return (k_arr_re,pkl_re)
