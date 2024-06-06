import numpy as np
import pylab as pl
import warnings
warnings.filterwarnings("ignore")
import warnings
from contextlib import contextmanager
import logging

# Suppress absl warnings
@contextmanager
def suppress_warnings():
    warnings.filterwarnings("ignore")
    try:
        yield
    finally:
        warnings.resetwarnings()
import absl.logging
absl.logging.set_verbosity('error')
# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
with suppress_warnings():
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


import cosmopower
import os
import subprocess
from cosmopower import cosmopower_NN
from cosmopower import cosmopower_PCAplusNN
from .config import *
import scipy.optimize as optimize
import os
#cosmo_model = "lcdm", "mnu", "neff", "wcdm"

class cosmopower:

    def __init__(self,cosmo_model="lcdm",path=None):

        self.cosmo_params = None


        path_to_emulators = path + cosmo_model + "/"
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

        path_to_emulators = path + self.mp +'/'

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

        # k_cutoff = self.cosmo_params["k_cutoff"]
        # ps_cutoff = self.cosmo_params["ps_cutoff"]
        #
        # if k_cutoff < 10:
        #
        #     x = np.linspace(np.log10(0.1),np.log10(10.))
        #
        #     centre = np.log10(0.5)
        #     max = 1
        #     min = 0.7
        #     width = (np.log10(10.)-np.log10(0.1))
        #
        #     suppression = -np.tanh((x-centre)/width*4)*0.5*(max-min)+min+(max-min)*0.5
        #
        #     ps_cutoff = np.interp(np.log10(k_arr_re),x+np.log10(0.677),suppression)
        #
        #     # pl.figure()
        #     # pl.semilogx(k_arr_re/0.677,ps_cutoff)
        #     # pl.xlim([0.01,100])
        #     # pl.xlabel("$k$ ($h$ Mpc)")
        #     # pl.ylabel("Power spectrum suppression")
        #     # pl.savefig("/home/iz221/cnc/figures/test_ps.pdf")
        #     # pl.show()
        #
        #
        # #indices = np.where(k_arr_re > k_cutoff)
        # #pkl_re[indices] = pkl_re[indices]*ps_cutoff
        #
        # pkl_re = pkl_re*ps_cutoff

        return (k_arr_re,pkl_re)
