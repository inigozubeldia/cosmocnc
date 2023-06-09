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

        if cosmo_params is None:
            cosmo_params = cosmo_params_default
        self.cosmo_params = cosmo_params
        self.amplitude_parameter = amplitude_parameter
        if cosmology_tool == "classy_sz":
            from classy_sz import Class

            self.classy_sptref = Class()

            spt_cosmoRef = {'Omega_m':.3,
                            'Omega_l':.7,
                            'h':.7,
                            'w0':-1.,
                            'wa':0,
                            # "Ob0":
                            }
            self.classy_sptref.set({
                           'H0': spt_cosmoRef["h"]*100.,
                           'omega_b': self.cosmo_params["Ob0"]*spt_cosmoRef["h"]**2,
                           'omega_cdm': (spt_cosmoRef['Omega_m']-self.cosmo_params["Ob0"])*spt_cosmoRef["h"]**2,
                           'output': ' '
                           }
                           )
            self.classy_sptref.compute()
            self.background_cosmology_sptref = classy_sz(self.classy_sptref)
            # print('spt cosmoref computed')

            self.classy = Class()


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

            self.T_CMB_0 = self.classy.T_cmb()
            self.N_eff = self.classy.get_current_derived_parameters(['Neff'])['Neff']
            self.z_CMB = self.classy.get_current_derived_parameters(['z_rec'])['z_rec']
            self.D_CMB = self.classy.get_current_derived_parameters(['da_rec'])['da_rec']
            self.sigma8 = self.classy.get_current_derived_parameters(['sigma8'])['sigma8']
            self.power_spectrum = classy_sz(self.classy)
            self.background_cosmology = classy_sz(self.classy)
            self.background_cosmology.H0.value = self.classy.h()*100.


            # print('cosmo computed',self.sigma8)





        if cosmology_tool == "astropy":

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
            self.D_CMB = self.background_cosmology.angular_diameter_distance(self.z_CMB).value

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

    def update_cosmology(self,cosmo_params_new,cosmology_tool = "astropy"):

        self.cosmo_params = cosmo_params_new

        if cosmology_tool == "classy_sz":
            ts = time.time()

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

                          # for mass conversion routines:
                          'output': 'mPk,m500c_to_m200c',
                          'M_min' : 1e9,
                          'M_max' : 1e16,
                          'z_min' : 0.,
                          'z_max' : 2.,
                          'ndim_redshifts' :50,
                          'ndim_masses' :50,
                          'concentration parameter':'D08'
                          }

            if self.amplitude_parameter == "sigma_8":

                classy_params['sigma8'] = self.cosmo_params["sigma_8"]

            elif self.amplitude_parameter == "A_s":
                classy_params['ln10^{10}A_s'] = np.log(self.cosmo_params["A_s"]*1e10)
            # print('classy_params:',classy_params)



            spt_cosmoRef = {'Omega_m':.3,
                            'Omega_l':.7,
                            'h':.7,
                            'w0':-1.,
                            'wa':0,
                            # "Ob0":
                            }
            self.spt_cosmoRef = spt_cosmoRef
            self.classy_sptref.set({
                                   'H0': spt_cosmoRef["h"]*100.,
                                   'omega_b': self.cosmo_params["Ob0"]*spt_cosmoRef["h"]**2,
                                   'omega_cdm': (spt_cosmoRef['Omega_m']-self.cosmo_params["Ob0"])*spt_cosmoRef["h"]**2,
                                   'output': ' '
                                   }
                                   )
            # print('recomputing spt ref cosmo')
            self.classy_sptref.compute()
            # print('spt ref cosmo recomputed')
            self.background_cosmology_sptref = classy_sz(self.classy_sptref)
            # self.background_cosmology_sptref.H0.value = (self.classy_sptref.h()*100.)

            spt_zs = np.linspace(1e-5,5,200)
            self.spt_ln1pzs = np.log(1.+spt_zs)
            self.spt_lndas_hmpc = np.log(self.background_cosmology_sptref.angular_diameter_distance(spt_zs).value*self.classy_sptref.h())


            # cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1, 'wa':0}
            spt_cosmoRef_masscal = {'Omega_m':.272,
                                    'Omega_l':.728,
                                    'h':.702,
                                    'w0':-1.,
                                    'wa':0,
                                    # "Ob0":
                                    }
            self.spt_cosmoRef_masscal = spt_cosmoRef_masscal
            self.classy_sptref.set({
                                   'H0': spt_cosmoRef_masscal["h"]*100.,
                                   'omega_b': self.cosmo_params["Ob0"]*spt_cosmoRef_masscal["h"]**2,
                                   'omega_cdm': (spt_cosmoRef_masscal['Omega_m']-self.cosmo_params["Ob0"])*spt_cosmoRef_masscal["h"]**2,
                                   'output': ' '
                                   }
                                   )
            # print('recomputing spt ref cosmo')
            self.classy_sptref.compute()
            # print('spt ref cosmo recomputed')
            self.background_cosmology_sptref_masscal = classy_sz(self.classy_sptref)
            # self.background_cosmology_sptref.H0.value = (self.classy_sptref.h()*100.)

            spt_zs_masscal = np.linspace(1e-5,5,200)
            self.spt_ln1pzs_masscal = np.log(1.+spt_zs_masscal)
            self.spt_lndas_hmpc_masscal = np.log(self.background_cosmology_sptref_masscal.angular_diameter_distance(spt_zs_masscal).value*self.classy_sptref.h())




            # print('spt cosmoref re-computed',self.background_cosmology_sptref.H0.value/100.)
            # exit(0)
            # self.classy.set(classy_params)
            self.classy.set(classy_params)
            self.classy.compute_class_szfast()
            self.T_CMB_0 = self.classy.T_cmb()
            self.N_eff = self.classy.get_current_derived_parameters(['Neff'])['Neff']
            self.z_CMB = self.classy.get_current_derived_parameters(['z_rec'])['z_rec']
            self.D_CMB = self.classy.get_current_derived_parameters(['da_rec'])['da_rec']
            self.sigma8 = self.classy.get_current_derived_parameters(['sigma8'])['sigma8']
            self.power_spectrum = classy_sz(self.classy)
            self.background_cosmology = classy_sz(self.classy)
            self.background_cosmology.H0.value = self.classy.h()*100.
            self.get_m500c_to_m200c_at_z_and_M = np.vectorize(self.classy.get_m500c_to_m200c_at_z_and_M)
            self.get_c200c_at_m_and_z = np.vectorize(self.classy.get_c200c_at_m_and_z_D08)
            # print('spt cosmoref re-computed 2',self.background_cosmology_sptref.H0.value/100.)
            # exit(0)

            te = time.time()
            print('time to do cosmo:  %.3e s'%(te-ts))



        if cosmology_tool == "astropy":

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

                    self.power_spectrum.set_cosmology(H0=self.cosmo_params["h"]*100.,Ob0=self.cosmo_params["Ob0"],
                    Oc0=self.cosmo_params["Om0"]-self.cosmo_params["Ob0"],ln10A_s=np.log(self.cosmo_params["A_s"]*1e10),
                    n_s=self.cosmo_params["n_s"])

                elif self.amplitude_parameter == "A_s":

                    self.sigma_8 = self.power_spectrum.get_sigma_8()

        theta_mc = self.get_theta_mc()

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

class classy_sz(object):

    def __init__(self,classy):

        self.const = constants()

        ndspl = 10 # fixed by cosmopower -- dont change !
        self.k_arr = np.geomspace(1e-4,50.,5000)[::ndspl] # fixed by cosmopower -- dont change !

        self.classy = classy
        self.h = self.classy.h()*100.

    def get_linear_power_spectrum(self,redshift):

        return (self.k_arr,np.vectorize(self.classy.pk_lin)(self.k_arr,redshift))

    def critical_density(self,z):

        conv_fac = 1./(1000.*self.const.mpc**3/self.const.solar)

        class result:

            value = np.vectorize(self.classy.get_rho_crit_at_z)(z)*conv_fac*self.classy.h()**2

        return result

    def differential_comoving_volume(self,z):

        class result:

            value = np.vectorize(self.classy.get_volume_dVdzdOmega_at_z)(z)*self.classy.h()**-3

        return result

    def angular_diameter_distance(self,z):

        class result:

            value = np.vectorize(self.classy.angular_distance)(z)

        return result

    def angular_diameter_distance_z1z2(self,z1,z2):

        class result:

            value = -(1./(1.+z2))*(np.vectorize(self.classy.angular_distance)(z1)*(1.+z1)-np.vectorize(self.classy.angular_distance)(z2)*(1.+z2))

        return result

    def H(self,z):

        conv_fac = 299792.458

        class result:

            value = np.vectorize(self.classy.Hubble)(z)*conv_fac

        return result

    class H0:
        # def __init__(self):
            # conv_fac = 299792.458
            # class result:
        value = 0
        # return {'hubble':np.vectorize(self.classy.Hubble)(z)}
        # return result
