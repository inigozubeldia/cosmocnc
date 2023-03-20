import numpy as np
import sys
from .ps import *
import astropy.cosmology as cp
from .config import *

#for now only lcdm

class cosmo_params_default:

    def __init__(self):

        self.params = {"Om0":0.3096,"Ob0":0.04897,"h":0.6766,"A_s":1.9687e-9,"n_s":0.96,"m_nu":0.06}

class cosmology_model:

    # def __init__(self,cosmo_params=None,power_spectrum_type="cosmopower",path_to_cosmopower="/rds-d4/user/iz221/hpc-work/cosmopower/"):
    def __init__(self,cosmo_params=None,power_spectrum_type="cosmopower"):#,
                 # path_to_cosmopower=path_to_cosmopower_organization
                 # ):

        if cosmo_params is None:

            #m_nu is sum of the three neutrino masses

            cosmo_params = {"Om0":0.3096,"Ob0":0.04897,"h":0.6766,"A_s":1.9687e-9,"n_s":0.96,"m_nu":0.06}

        self.cosmo_params = cosmo_params
        self.power_spectrum_type = power_spectrum_type

        self.T_CMB_0 = 2.7255
        self.N_eff = 3.046

        self.background_cosmology = cp.FlatLambdaCDM(self.cosmo_params["h"]*100.,
        self.cosmo_params["Om0"],Ob0=self.cosmo_params["Ob0"],Tcmb0=self.T_CMB_0,
        Neff=self.N_eff,m_nu=self.cosmo_params["m_nu"]/3.)

        if self.power_spectrum_type == "cosmopower":

            # self.power_spectrum = ps.cosmopower(cosmo_model="lcdm",path_to_cosmopower=path_to_cosmopower)
            self.power_spectrum = cosmopower(cosmo_model="lcdm")#,path_to_cosmopower=path_to_cosmopower)
            self.power_spectrum.set_cosmology(H0=self.cosmo_params["h"]*100.,Ob0=self.cosmo_params["Ob0"],
            Oc0=self.cosmo_params["Om0"]-self.cosmo_params["Ob0"],ln10A_s=np.log(self.cosmo_params["A_s"]*1e10),
            n_s=self.cosmo_params["n_s"])

    def update_cosmology(self,cosmo_params_new):

        self.cosmo_params = cosmo_params_new
        self.background_cosmology = cp.FlatLambdaCDM(self.cosmo_params["h"]*100.,
        self.cosmo_params["Om0"],Ob0=self.cosmo_params["Ob0"],Tcmb0=self.T_CMB_0,
        Neff=self.N_eff,m_nu=self.cosmo_params["m_nu"]/3.)

        if self.power_spectrum_type == "cosmopower":

            self.power_spectrum.set_cosmology(H0=self.cosmo_params["h"]*100.,Ob0=self.cosmo_params["Ob0"],
            Oc0=self.cosmo_params["Om0"]-self.cosmo_params["Ob0"],ln10A_s=np.log(self.cosmo_params["A_s"]*1e10),
            n_s=self.cosmo_params["n_s"])
