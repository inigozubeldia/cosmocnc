import numpy as np
import astropy.cosmology as cp
import sys
from .ps import *
from .config import *
from .hmf import *
import scipy.integrate as integrate
import time

#for now only lcdm

class cosmology_model:

    def __init__(self,cosmo_params=None,power_spectrum_type="cosmopower",amplitude_parameter="sigma_8"):
        # print('cosmo_params',cosmo_params)
        if cosmo_params is None:
            cosmo_params = cosmo_params_default

        self.cosmo_params = cosmo_params
        self.power_spectrum_type = power_spectrum_type
        self.amplitude_parameter = amplitude_parameter

        self.T_CMB_0 = 2.7255
        self.N_eff = 3.046

        self.background_cosmology = cp.FlatLambdaCDM(self.cosmo_params["h"]*100.,
        self.cosmo_params["Om0"],Ob0=self.cosmo_params["Ob0"],Tcmb0=self.T_CMB_0,
        Neff=self.N_eff,m_nu=self.cosmo_params["m_nu"]/3.)

        self.z_CMB = self.get_z_cmb()
        self.D_CMB = self.background_cosmology.angular_diameter_distance(self.z_CMB).value

        if self.power_spectrum_type == "cosmopower":
            self.power_spectrum = cosmopower(cosmo_model="lcdm")
            self.power_spectrum.set_cosmology(H0=self.cosmo_params["h"]*100.,Ob0=self.cosmo_params["Ob0"],
            Oc0=self.cosmo_params["Om0"]-self.cosmo_params["Ob0"],ln10A_s=np.log(self.cosmo_params["A_s"]*1e10),
            n_s=self.cosmo_params["n_s"])

            if self.amplitude_parameter == "sigma_8":

                self.sigma_8 = self.cosmo_params["sigma_8"]
                self.cosmo_params["A_s"] = self.power_spectrum.find_As(self.sigma_8)

                self.power_spectrum.set_cosmology(H0=self.cosmo_params["h"]*100.,Ob0=self.cosmo_params["Ob0"],
                Oc0=self.cosmo_params["Om0"]-self.cosmo_params["Ob0"],ln10A_s=np.log(self.cosmo_params["A_s"]*1e10),
                n_s=self.cosmo_params["n_s"])

            elif self.amplitude_parameter == "A_s":

                self.sigma_8 = self.power_spectrum.get_sigma_8()

    def update_cosmology(self,cosmo_params_new):

        self.cosmo_params = cosmo_params_new
        self.background_cosmology = cp.FlatLambdaCDM(self.cosmo_params["h"]*100.,
        self.cosmo_params["Om0"],Ob0=self.cosmo_params["Ob0"],Tcmb0=self.T_CMB_0,
        Neff=self.N_eff,m_nu=self.cosmo_params["m_nu"]/3.)

        self.z_CMB = self.get_z_cmb()
        self.D_CMB = self.background_cosmology.angular_diameter_distance(self.z_CMB).value

        if self.power_spectrum_type == "cosmopower":
            self.power_spectrum.set_cosmology(H0=self.cosmo_params["h"]*100.,Ob0=self.cosmo_params["Ob0"],
            Oc0=self.cosmo_params["Om0"]-self.cosmo_params["Ob0"],ln10A_s=np.log(self.cosmo_params["A_s"]*1e10),
            n_s=self.cosmo_params["n_s"])

            if self.amplitude_parameter == "sigma_8":

                self.sigma_8 = self.cosmo_params["sigma_8"]
                self.cosmo_params["A_s"] = self.power_spectrum.find_As(self.sigma_8)
                # print('got:',self.cosmo_params["A_s"])
                self.power_spectrum.set_cosmology(H0=self.cosmo_params["h"]*100.,Ob0=self.cosmo_params["Ob0"],
                Oc0=self.cosmo_params["Om0"]-self.cosmo_params["Ob0"],ln10A_s=np.log(self.cosmo_params["A_s"]*1e10),
                n_s=self.cosmo_params["n_s"])

            elif self.amplitude_parameter == "A_s":

                self.sigma_8 = self.power_spectrum.get_sigma_8()

        # print('computing thetamc for params:',self.cosmo_params)
        theta_mc = self.get_theta_mc()
        # print('got: ',theta_mc)

    def get_theta_mc(self):
        Ogamma0 = 2.47282*10.**(-5)/self.cosmo_params["h"]**2
        Orad0 =  4.18343*10.**(-5)/self.cosmo_params["h"]**2
        Om0 = self.cosmo_params["Om0"]
        Ob0 = self.cosmo_params["Ob0"]
        OL0 = 1.-Om0-Orad0

        a_cmb = 1./(1.+self.z_CMB)

        def sound_horizon_integrand(x):

            return 1./np.sqrt((1.+3.*Ob0*x/(4.*Ogamma0))*(OL0*x**4+Om0*x+Orad0))

        r_sound = integrate.quad(sound_horizon_integrand,0.,a_cmb)[0]/(self.cosmo_params["h"]*100.*np.sqrt(3.))*constants().c_light/1e3/self.z_CMB
        theta_mc = r_sound/self.D_CMB

        return theta_mc

    def get_z_cmb(self):

        Ob0h2 = self.cosmo_params["Ob0"]*self.cosmo_params["h"]**2
        Om0h2 = self.cosmo_params["Om0"]*self.cosmo_params["h"]**2

        g1 = 0.0783*(Ob0h2)**(-0.238)/(1.+39.5*(Ob0h2)**0.763)
        g2 = 0.56/(1.+21.1*(Ob0h2)**1.81)
        z_cmb = 1048.*(1.+0.00124*(Ob0h2)**(-0.738))*(1.+g1*Om0h2**g2)

        return z_cmb
