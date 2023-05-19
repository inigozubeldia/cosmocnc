import numpy as np

import sys
from .ps import *
from .config import *
from .hmf import *
import scipy.integrate as integrate
import time

#for now only lcdm


class cosmology_model:

    def __init__(self,cosmo_params=None,cosmology_tool = "astropy", power_spectrum_type="cosmopower",amplitude_parameter="sigma_8"):
        # print('cosmo_params',cosmo_params)
        if cosmo_params is None:
            cosmo_params = cosmo_params_default
        self.cosmo_params = cosmo_params
        self.amplitude_parameter = amplitude_parameter
        if cosmology_tool == "classy_sz":
            # print('using classy_sz')
            from classy_sz import Class
            self.classy = Class()
            # print("tau_reio:",self.cosmo_params["tau_reio"])
            self.classy.set({
                           'H0': self.cosmo_params["h"]*100.,
                           'omega_b': self.cosmo_params["Ob0"]*self.cosmo_params["h"]**2,
                           'omega_cdm': (self.cosmo_params["Om0"]-self.cosmo_params["Ob0"])*self.cosmo_params["h"]**2,
                           'ln10^{10}A_s':np.log(self.cosmo_params["A_s"]*1e10),
                           'tau_reio':  self.cosmo_params["tau_reio"],
                           'n_s': self.cosmo_params["n_s"],

                           'N_ncdm' : 1,
                           'N_ur' : 2.0328,
                           'm_ncdm' : 0.06,
                           'T_ncdm' : 0.71611,


                          'output': 'mPk',
                          'skip_background_and_thermo': 0,
                          'skip_chi': 1,
                          'skip_hubble': 1,
                          'skip_cmb': 1,
                          'skip_pknl': 1,
                          'skip_pkl': 0,
                          'skip_sigma8_and_der': 0,
                          'skip_sigma8_at_z': 1,
                          # 'class_sz_verbose': 1,
                          # 'background_verbose':3,
                          # 'thermodynamics_verbose':3
                          })
            self.classy.compute_class_szfast()
            # pktest = self.classy.pk_lin(1e-3,0)
            self.T_CMB_0 = self.classy.T_cmb()
            # print("self.T_CMB_0:",self.T_CMB_0)
            self.N_eff = self.classy.get_current_derived_parameters(['Neff'])['Neff']
            # print("self.N_eff:",self.N_eff)
            # self.z_CMB = self.classy.get_current_derived_parameters(['z_star'])['z_star']
            # print("self.z_CMB:",self.z_CMB)
            self.z_CMB = self.classy.get_current_derived_parameters(['z_rec'])['z_rec']
            # print("self.z_CMB:",self.z_CMB)
            self.D_CMB = self.classy.get_current_derived_parameters(['da_rec'])['da_rec']
            # print("self.D_CMB:",self.D_CMB)
            self.sigma8 = self.classy.get_current_derived_parameters(['sigma8'])['sigma8']
            # print("self.sigma8:",self.sigma8)
            self.power_spectrum = classy_sz(self.classy)
            # k,ps = self.power_spectrum.get_linear_power_spectrum(0.)
            # print('k,ps',k,ps)
            self.background_cosmology = classy_sz(self.classy)
            self.background_cosmology.H0.value = self.classy.h()*100.

            # print('H0',self.background_cosmology.H0.value)
            # exit(0)

            # exit(0)
        if cosmology_tool == "astropy":
            # print("importing astroppy")
            import astropy.cosmology as cp
            self.cosmology_tool = cp



            self.power_spectrum_type = power_spectrum_type


            self.T_CMB_0 = 2.7255
            self.N_eff = 3.046

            self.background_cosmology = self.cosmology_tool.FlatLambdaCDM(self.cosmo_params["h"]*100.,
                                                                          self.cosmo_params["Om0"],
                                                                          Ob0=self.cosmo_params["Ob0"],
                                                                          Tcmb0=self.T_CMB_0,
                                                                          Neff=self.N_eff,
                                                                          m_nu=self.cosmo_params["m_nu"]/3.)

            self.z_CMB = self.get_z_cmb()
            # print('z_cmb',self.z_CMB)
            self.D_CMB = self.background_cosmology.angular_diameter_distance(self.z_CMB).value
            # print('D_CMB',self.D_CMB)
            if self.power_spectrum_type == "cosmopower":
                self.power_spectrum = cosmopower(cosmo_model="lcdm")
                self.power_spectrum.set_cosmology(H0=self.cosmo_params["h"]*100.,
                                                  Ob0=self.cosmo_params["Ob0"],
                                                  Oc0=self.cosmo_params["Om0"]-self.cosmo_params["Ob0"],
                                                  ln10A_s=np.log(self.cosmo_params["A_s"]*1e10),
                                                  n_s=self.cosmo_params["n_s"])

                if self.amplitude_parameter == "sigma_8":

                    self.sigma_8 = self.cosmo_params["sigma_8"]
                    self.cosmo_params["A_s"] = self.power_spectrum.find_As(self.sigma_8)

                    self.power_spectrum.set_cosmology(H0=self.cosmo_params["h"]*100.,
                                                      Ob0=self.cosmo_params["Ob0"],
                                                      Oc0=self.cosmo_params["Om0"]-self.cosmo_params["Ob0"],
                                                      ln10A_s=np.log(self.cosmo_params["A_s"]*1e10),
                                                      n_s=self.cosmo_params["n_s"])

                elif self.amplitude_parameter == "A_s":

                    self.sigma_8 = self.power_spectrum.get_sigma_8()
                # print('sigma8:',self.sigma_8)
            # exit(0)

    def update_cosmology(self,cosmo_params_new,cosmology_tool = "astropy"):

        self.cosmo_params = cosmo_params_new
        if cosmology_tool == "classy_sz":
            # print('updating classy_sz')
            # print('using classy_sz')
            # from classy_sz import Class
            # self.classy = Class()
            # print("tau_reio:",self.cosmo_params["tau_reio"])
            classy_params = {
                           'H0': self.cosmo_params["h"]*100.,
                           'omega_b': self.cosmo_params["Ob0"]*self.cosmo_params["h"]**2,
                           'omega_cdm': (self.cosmo_params["Om0"]-self.cosmo_params["Ob0"])*self.cosmo_params["h"]**2,
                           # 'ln10^{10}A_s':np.log(self.cosmo_params["A_s"]*1e10),
                           'tau_reio':  self.cosmo_params["tau_reio"],
                           'n_s': self.cosmo_params["n_s"],

                           'N_ncdm' : 1,
                           'N_ur' : 2.0328,
                           'm_ncdm' : 0.06,
                           'T_ncdm' : 0.71611,


                          'output': 'mPk',
                          'skip_background_and_thermo': 0,
                          'skip_chi': 1,
                          'skip_hubble': 1,
                          'skip_cmb': 1,
                          'skip_pknl': 1,
                          'skip_pkl': 0,
                          'skip_sigma8_and_der': 0,
                          'skip_sigma8_at_z': 1,
                          # 'class_sz_verbose': 1,
                          # 'background_verbose':3,
                          # 'thermodynamics_verbose':3
                          }
            if self.amplitude_parameter == "sigma_8":
                # print('self.cosmo_params["sigma_8"]',self.cosmo_params["sigma_8"])
                classy_params['sigma8'] = self.cosmo_params["sigma_8"]
            elif self.amplitude_parameter == "A_s":
                classy_params['ln10^{10}A_s'] = np.log(self.cosmo_params["A_s"]*1e10)
            # print('classy_params:',classy_params)
            self.classy.set(classy_params)
            self.classy.compute_class_szfast()
            # pktest = self.classy.pk_lin(1e-3,0)
            self.T_CMB_0 = self.classy.T_cmb()
            # print("self.T_CMB_0:",self.T_CMB_0)
            self.N_eff = self.classy.get_current_derived_parameters(['Neff'])['Neff']
            # print("self.N_eff:",self.N_eff)
            # self.z_CMB = self.classy.get_current_derived_parameters(['z_star'])['z_star']
            # print("self.z_CMB:",self.z_CMB)
            self.z_CMB = self.classy.get_current_derived_parameters(['z_rec'])['z_rec']
            # print("self.z_CMB:",self.z_CMB)
            self.D_CMB = self.classy.get_current_derived_parameters(['da_rec'])['da_rec']
            # print("self.D_CMB:",self.D_CMB)
            self.sigma8 = self.classy.get_current_derived_parameters(['sigma8'])['sigma8']
            # print("self.sigma8:",self.sigma8)
            self.power_spectrum = classy_sz(self.classy)
            # k,ps = self.power_spectrum.get_linear_power_spectrum(0.)
            # print('k,ps',k,ps)
            self.background_cosmology = classy_sz(self.classy)
            self.background_cosmology.H0.value = self.classy.h()*100.

        if cosmology_tool == "astropy":
            # print("updating astroppy")
            self.background_cosmology = self.cosmology_tool.FlatLambdaCDM(self.cosmo_params["h"]*100.,
                                                                          self.cosmo_params["Om0"],
                                                                          Ob0=self.cosmo_params["Ob0"],
                                                                          Tcmb0=self.T_CMB_0,
                                                                          Neff=self.N_eff,
                                                                          m_nu=self.cosmo_params["m_nu"]/3.)

            self.z_CMB = self.get_z_cmb()
            self.D_CMB = self.background_cosmology.angular_diameter_distance(self.z_CMB).value

            if self.power_spectrum_type == "cosmopower":
                self.power_spectrum.set_cosmology(H0=self.cosmo_params["h"]*100.,
                                                  Ob0=self.cosmo_params["Ob0"],
                                                  Oc0=self.cosmo_params["Om0"]-self.cosmo_params["Ob0"],
                                                  ln10A_s=np.log(self.cosmo_params["A_s"]*1e10),
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


class classy_sz:
    def __init__(self,classy):
        self.const = constants()

        ndspl = 10 # fixed by cosmopower -- dont change !
        self.k_arr = np.geomspace(1e-4,50.,5000)[::ndspl] # fixed by cosmopower -- dont change !

        self.classy = classy
        self.h = self.classy.h()*100.
        ###
    def get_linear_power_spectrum(self,redshift):
        return (self.k_arr,np.vectorize(self.classy.pk_lin)(self.k_arr,redshift))

    def critical_density(self,z):
        # conv_fac = 1.999999999e30/(3.08567758128e22)**3/self.classy.h()**4
        # conv_fac = 1./(3.08567758128e22)**3/self.classy.h()**4
        # print('z',z)
        conv_fac = 1./(1000.*self.const.mpc**3/self.const.solar)
        class result:
            value = np.vectorize(self.classy.get_rho_crit_at_z)(z)*conv_fac*self.classy.h()**2
        return result

    def differential_comoving_volume(self,z):
        # conv_fac = 1.999999999e30/(3.08567758128e22)**3/self.classy.h()**4
        # conv_fac = 1./(3.08567758128e22)**3/self.classy.h()**4
        # print('z',z)
        # conv_fac = 1./(1000.*self.const.mpc**3/self.const.solar)
        class result:
            value = np.vectorize(self.classy.get_volume_dVdzdOmega_at_z)(z)*self.classy.h()**-3#*conv_fac*self.classy.h()**2
        return result

    def angular_diameter_distance(self,z):
        class result:
            value = np.vectorize(self.classy.angular_distance)(z)
        return result
    def angular_diameter_distance_z1z2(self,z1,z2):
        # print('z1',z1)
        # print('z2',z2)
        # print('Daz2',np.vectorize(self.classy.angular_distance)(z2)/(1.+z2))
        # print('Daz1',np.vectorize(self.classy.angular_distance)(z1)/(1.+z1))
        # print('Dazinf',np.vectorize(self.classy.angular_distance)(1e-6))
        # exit(0)
        class result:
            value = -(1./(1.+z2))*(np.vectorize(self.classy.angular_distance)(z1)*(1.+z1)-np.vectorize(self.classy.angular_distance)(z2)*(1.+z2))
        return result
        # return {'DA':np.vectorize(self.classy.angular_distance)(z[0])}
    def H(self,z):
        conv_fac = 299792.458
        class result:
            value = np.vectorize(self.classy.Hubble)(z)*conv_fac
        # return {'hubble':np.vectorize(self.classy.Hubble)(z)}
        return result

    class H0:
        # def __init__(self):
            # conv_fac = 299792.458
            # class result:
        value = 0
        # return {'hubble':np.vectorize(self.classy.Hubble)(z)}
        # return result
