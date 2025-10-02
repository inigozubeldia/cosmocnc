import numpy as np

import sys
from .ps import *
from .config import *
from .hmf import *
import scipy.integrate as integrate
import time

#for now only lcdm


class cosmology_model:

    def __init__(self,cosmo_params=None,cosmology_tool="astropy",power_spectrum_type="cosmopower",
    amplitude_parameter="sigma_8",cnc_params = None,logger = None):

        self.cnc_params = cnc_params

        self.logger = logging.getLogger(__name__)

        # if cosmo_params is None:

        #     cosmo_params = cosmo_params_default

        self.cosmo_params = cosmo_params
        self.amplitude_parameter = amplitude_parameter

        self.logger.info(f'Cosmology params: {self.cosmo_params}')
        # if self.cnc_params["cosmo_model"] != self.cnc_params["class_sz_cosmo_model"]:
        #     self.logger.warning(f'Cosmology model in cosmocnc params and classy_sz params do not match. Using classy_sz params.')
        #     self.cnc_params["class_sz_cosmo_model"] = self.cnc_params["cosmo_model"]
        #

        if cosmology_tool == "classy_sz":

            from classy_sz import Class as Class_sz

            self.classy = Class_sz()

            if self.amplitude_parameter == "sigma_8":

                self.classy.set({"sigma8": self.cosmo_params["sigma_8"]})

            elif self.amplitude_parameter == "A_s":

                self.classy.set({"ln10^{10}A_s": np.log(self.cosmo_params["A_s"]*1e10)})

            if self.cnc_params["cosmo_param_density"] == "critical":

                self.classy.set({'omega_b': self.cosmo_params["Ob0"]*self.cosmo_params["h"]**2,
                           'omega_cdm': (self.cosmo_params["Om0"]-self.cosmo_params["Ob0"])*self.cosmo_params["h"]**2})

                self.cosmo_params["Ob0h2"] = self.cosmo_params["Ob0"]*self.cosmo_params["h"]**2
                self.cosmo_params["Oc0h2"] = (self.cosmo_params["Om0"]-self.cosmo_params["Ob0"])*self.cosmo_params["h"]**2

            #This needs to be updated
            #Om0 = Oc0 + Ob0 + cosmo_params['N_ncdm']*cosmo_params['m_ncdm']/93.14/h()**2

            elif self.cnc_params["cosmo_param_density"] == "physical":

                self.classy.set({'omega_b': self.cosmo_params["Ob0h2"],
                           'omega_cdm': self.cosmo_params["Oc0h2"]})

                self.cosmo_params["Ob0"] = self.cosmo_params["Ob0h2"]/self.cosmo_params["h"]**2
                self.cosmo_params["Om0"] = (self.cosmo_params["Oc0h2"]+self.cosmo_params["Ob0h2"])/self.cosmo_params["h"]**2

            if self.cnc_params["cosmo_param_density"] == "mixed":

                self.cosmo_params["Ob0"] = self.cosmo_params["Ob0h2"]/self.cosmo_params["h"]**2
                self.cosmo_params["Oc0h2"] = (self.cosmo_params["Om0"]-self.cosmo_params["Ob0"])*self.cosmo_params["h"]**2

                self.classy.set({'omega_b': self.cosmo_params["Ob0h2"],
                           'omega_cdm': self.cosmo_params["Oc0h2"]})
                
            self.cosmo_model_dict = {'lcdm' : 0,
                                     'mnu'  : 1,
                                     'neff' : 2,
                                     'wcdm' : 3,
                                     'ede'  : 4,
                                     'mnu-3states' : 5,
                                     'ede-v2'  : 6,
                                     }

            self.classy.set({
                           'H0': self.cosmo_params["h"]*100.,
                           'tau_reio':  self.cosmo_params["tau_reio"],
                           'n_s': self.cosmo_params["n_s"],

                          'output': self.cnc_params["class_sz_output"],



                          'HMF_prescription_NCDM': 1,
                          'no_spline_in_tinker': 1,

                          'M_min' : self.cnc_params["M_min"]*0.5, #This is in units M_Sun/h
                          'M_max' : self.cnc_params["M_max"]*1.2,
                          'z_min' : self.cnc_params["z_min"]*0.8,
                          'z_max' : self.cnc_params["z_max"]*1.2,

                          'ndim_redshifts' : self.cnc_params["n_z"],
                          'ndim_masses' : self.cnc_params["class_sz_ndim_masses"], # automatically set in fast mode
                          'concentration_parameter': self.cnc_params["class_sz_concentration_parameter"],
                          'cosmo_model': self.cosmo_model_dict[self.cnc_params['cosmo_model']],
                          'mass_function' : self.cnc_params["class_sz_hmf"],

                          'use_m500c_in_ym_relation' : self.cnc_params["class_sz_use_m500c_in_ym_relation"],
                          'use_m200c_in_ym_relation' : self.cnc_params["class_sz_use_m200c_in_ym_relation"],

                          })

            if  self.cnc_params['cosmo_model'] == "wcdm":

                self.classy.set({
                    'Omega_Lambda' : 0.,
                    'w0_fld' : self.cosmo_params["w0"],
                })

            if  self.cnc_params['hmf_calc'] == "classy_sz":
                self.logger.info('adding dndlnM to class_sz output')
                self.classy.set({
                    'output': self.cnc_params["class_sz_output"] + ",dndlnm"
                })

            self.logger.info('computing class_szfast')

            self.classy.compute_class_szfast()

            self.logger.info('computing class_szfast done')

            self.T_CMB_0 = self.classy.T_cmb()
            self.N_eff = self.classy.get_current_derived_parameters(['Neff'])['Neff']
            self.z_CMB = self.classy.get_current_derived_parameters(['z_rec'])['z_rec']
            self.D_CMB = self.classy.get_current_derived_parameters(['da_rec'])['da_rec']
            self.sigma8 = self.classy.get_current_derived_parameters(['sigma8'])['sigma8']
            self.As = np.exp(self.classy.get_current_derived_parameters(["ln10^{10}A_s"])["ln10^{10}A_s"])/1e10
            self.cosmo_params["A_s"] = self.As
            self.cosmo_params["sigma_8"] = self.sigma8

            self.Omega_nu = self.classy.Omega_nu
            self.cosmo_params["Onu0"] = self.Omega_nu

            self.power_spectrum = classy_sz(self.classy)
            self.background_cosmology = classy_sz(self.classy)
            self.background_cosmology.H0.value = self.classy.h()*100.

            self.logger.debug(f'Got: {self.T_CMB_0}, {self.sigma8}')

            self.get_m500c_to_m200c_at_z_and_M = np.vectorize(self.classy.get_m500c_to_m200c_at_z_and_M)
            self.get_m200c_to_m500c_at_z_and_M = np.vectorize(self.classy.get_m200c_to_m500c_at_z_and_M)
            self.get_c200c_at_m_and_z = np.vectorize(self.classy.get_c200c_at_m_and_z_D08)
            self.get_dndlnM_at_z_and_M = np.vectorize(self.classy.get_dndlnM_at_z_and_M)
            self.get_delta_mean_from_delta_crit_at_z = np.vectorize(self.classy.get_delta_mean_from_delta_crit_at_z)

            # self.logger.debug(f'collected hmf')

        elif cosmology_tool == "astropy":

            import astropy.cosmology as cp

            self.cosmology_tool = cp
            self.power_spectrum_type = power_spectrum_type

            self.T_CMB_0 = 2.7255

            self.background_cosmology = self.cosmology_tool.FlatLambdaCDM(self.cosmo_params["h"]*100.,
                                                                          self.cosmo_params["Om0"],
                                                                          Ob0=self.cosmo_params["Ob0"],
                                                                          Tcmb0=self.T_CMB_0,
                                                                          Neff=self.cosmo_params["N_eff"],
                                                                          m_nu=self.cosmo_params["m_nu"]/3.)

            self.z_CMB = self.get_z_cmb()
            self.D_CMB = self.background_cosmology.angular_diameter_distance(self.z_CMB).value

            if self.power_spectrum_type == "cosmopower":

                self.power_spectrum = cosmopower(cosmo_model=self.cnc_params["cosmo_model"],path=self.cnc_params["path_to_cosmopower_organization"])
                self.power_spectrum.set_cosmology(H0=self.cosmo_params["h"]*100.,
                                                  Ob0=self.cosmo_params["Ob0"],
                                                  Oc0=self.cosmo_params["Om0"]-self.cosmo_params["Ob0"],
                                                  ln10A_s=np.log(self.cosmo_params["A_s"]*1e10),
                                                  n_s=self.cosmo_params["n_s"])

                self.power_spectrum.cosmo_params = self.cosmo_params

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
                    self.cosmo_params["sigma_8"] = self.sigma_8

        elif cosmology_tool == "cobaya":

            cobaya_cosmology = cobaya_cosmo(self.cnc_params)

            self.power_spectrum = cobaya_cosmology
            self.background_cosmology = cobaya_cosmology
            self.background_cosmology.H0.value = cobaya_cosmology.H(0).value
            h = self.background_cosmology.H0.value/100.

            self.cosmo_params["Om0"] = cobaya_cosmology.Om(0.)
            self.cosmo_params["Ob0"] = cobaya_cosmology.Ob(0.)
            self.cosmo_params["sigma_8"] = cobaya_cosmology.sigma8(0.)
            self.cosmo_params["Onu0"] = cobaya_cosmology.Omega_nu_massive(0.)
            self.cosmo_params["n_s"] = cobaya_cosmology.ns
            self.cosmo_params["h"] = h


            self.z_CMB = cobaya_cosmology.z_cmb
            self.D_CMB = self.background_cosmology.angular_diameter_distance(self.z_CMB).value


        print("cosmo params",self.cosmo_params)


    def update_cosmology(self,cosmo_params_new,cosmology_tool = "astropy"):

        self.cosmo_params = cosmo_params_new

        if cosmology_tool == "classy_sz":

            classy_params = {
                           'H0': self.cosmo_params["h"]*100.,
                           'tau_reio':  self.cosmo_params["tau_reio"],
                           'n_s': self.cosmo_params["n_s"],
                           'm_ncdm' : self.cosmo_params["m_nu"],


                           'output': self.cnc_params["class_sz_output"],

                          'HMF_prescription_NCDM': 1,
                          'no_spline_in_tinker': 1,


                          # for mass conversion routines:
                          'output': self.cnc_params["class_sz_output"],
                          'M_min' : self.cnc_params["M_min"]*0.5,
                          'M_max' : self.cnc_params["M_max"]*1.2,
                          'z_min' : self.cnc_params["z_min"]*0.8,
                          'z_max' : self.cnc_params["z_max"]*1.2,

                          'ndim_redshifts' : self.cnc_params["n_z"],
                          'ndim_masses' : self.cnc_params["class_sz_ndim_masses"], # automatically set in fast mode
                          'concentration_parameter': self.cnc_params["class_sz_concentration_parameter"],

                          'cosmo_model': self.cosmo_model_dict[self.cnc_params['cosmo_model']],
                          'mass_function' : self.cnc_params["class_sz_hmf"],

                        #   'classy_sz_verbose': 'none',
                          'use_m500c_in_ym_relation' : self.cnc_params["class_sz_use_m500c_in_ym_relation"],
                          'use_m200c_in_ym_relation' : self.cnc_params["class_sz_use_m200c_in_ym_relation"],

                          }

            if  self.cnc_params['cosmo_model'] == "wcdm":

                classy_params.update({
                    'Omega_Lambda' : 0.,
                    'w0_fld' : self.cosmo_params["w0"],
                })

            if  self.cnc_params['hmf_calc'] == "classy_sz":
                self.logger.info('adding dndlnM to class_sz output in update_cosmology')
                self.classy.set({
                    'output': self.cnc_params["class_sz_output"] + ",dndlnm"
                })
            if self.amplitude_parameter == "sigma_8":

                classy_params['sigma8'] = self.cosmo_params["sigma_8"]

            elif self.amplitude_parameter == "A_s":

                classy_params['ln10^{10}A_s'] = np.log(self.cosmo_params["A_s"]*1e10)

            if self.cnc_params["cosmo_param_density"] == "critical":

                classy_params['omega_b'] = self.cosmo_params["Ob0"]*self.cosmo_params["h"]**2
                classy_params['omega_cdm'] = (self.cosmo_params["Om0"]-self.cosmo_params["Ob0"])*self.cosmo_params["h"]**2
                self.cosmo_params["Ob0h2"] = self.cosmo_params["Ob0"]*self.cosmo_params["h"]**2
                self.cosmo_params["Oc0h2"] = classy_params['omega_cdm']

            elif self.cnc_params["cosmo_param_density"] == "physical":

                classy_params['omega_b'] = self.cosmo_params["Ob0h2"]
                classy_params['omega_cdm'] = self.cosmo_params["Oc0h2"]
                self.cosmo_params["Ob0"] = self.cosmo_params["Ob0h2"]/self.cosmo_params["h"]**2
                self.cosmo_params["Om0"] = (self.cosmo_params["Oc0h2"]+self.cosmo_params["Ob0h2"])/self.cosmo_params["h"]**2


            if self.cnc_params["cosmo_param_density"] == "mixed":

                classy_params['omega_b'] = self.cosmo_params["Ob0h2"]
                self.cosmo_params["Ob0"] = self.cosmo_params["Ob0h2"]/self.cosmo_params["h"]**2

                classy_params['omega_cdm'] = (self.cosmo_params["Om0"]-self.cosmo_params["Ob0"])*self.cosmo_params["h"]**2
                self.cosmo_params["Oc0h2"] = classy_params['omega_cdm']

            self.classy.set(classy_params)

            self.classy.compute_class_szfast()

            self.T_CMB_0 = self.classy.T_cmb()
            self.N_eff = self.classy.get_current_derived_parameters(['Neff'])['Neff']
            self.z_CMB = self.classy.get_current_derived_parameters(['z_rec'])['z_rec']
            self.D_CMB = self.classy.get_current_derived_parameters(['da_rec'])['da_rec']
            self.sigma8 = self.classy.get_current_derived_parameters(['sigma8'])['sigma8']
            self.As = np.exp(self.classy.get_current_derived_parameters(["ln10^{10}A_s"])["ln10^{10}A_s"])/1e10
            self.cosmo_params["A_s"] = self.As
            self.cosmo_params["sigma_8"] = self.sigma8

            self.Omega_nu = self.classy.Omega_nu
            self.cosmo_params["Onu0"] = self.Omega_nu

            self.power_spectrum = classy_sz(self.classy)
            self.background_cosmology = classy_sz(self.classy)
            self.background_cosmology.H0.value = self.classy.h()*100.
            self.get_m500c_to_m200c_at_z_and_M = np.vectorize(self.classy.get_m500c_to_m200c_at_z_and_M)
            self.get_c200c_at_m_and_z = np.vectorize(self.classy.get_c200c_at_m_and_z_D08)
            self.get_dndlnM_at_z_and_M = np.vectorize(self.classy.get_dndlnM_at_z_and_M)

        elif cosmology_tool == "astropy":

            self.background_cosmology = self.cosmology_tool.FlatLambdaCDM(self.cosmo_params["h"]*100.,
                                                                          self.cosmo_params["Om0"],
                                                                          Ob0=self.cosmo_params["Ob0"],
                                                                          Tcmb0=self.T_CMB_0,
                                                                          Neff=self.cosmo_params["N_eff"],
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
                    self.cosmo_params["sigma_8"] = self.sigma_8

        elif cosmology_tool == "cobaya":

            cobaya_cosmology = cobaya_cosmo(self.cnc_params)

            self.power_spectrum = cobaya_cosmology
            self.background_cosmology = cobaya_cosmology
            self.background_cosmology.H0.value = cobaya_cosmology.H(0).value
            h = self.background_cosmology.H0.value/100.

            self.cosmo_params["Om0"] = cobaya_cosmology.Om(0.)
            self.cosmo_params["Ob0"] = cobaya_cosmology.Ob(0.)
            self.cosmo_params["sigma_8"] = cobaya_cosmology.sigma8(0.)
            self.cosmo_params["Onu0"] = cobaya_cosmology.Omega_nu_massive(0.)
            self.cosmo_params["n_s"] = cobaya_cosmology.ns
            self.cosmo_params["h"] = h

            self.z_CMB = cobaya_cosmology.z_cmb
            self.D_CMB = self.background_cosmology.angular_diameter_distance(self.z_CMB).value

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

    def get_Omega_nu(self):

        return self.Omega_nu


class classy_sz:

    def __init__(self,classy):

        self.const = constants()

        ndspl = 10 # fixed by cosmopower -- dont change !
        self.k_arr = np.geomspace(1e-4,50.,5000)[::ndspl] # fixed by cosmopower -- dont change !

        self.classy = classy

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


class cobaya_cosmo:

    def __init__(self,cnc_params):

        self.cnc_params = cnc_params
        self.const = constants()
        self.k_arr = np.geomspace(1e-4,50.,500) # same as in cosmopower, in Mpc-1
        self.provider = self.cnc_params["cobaya_provider"]
        self.cnc_params = cnc_params
        self.z_vec =  np.linspace(self.cnc_params["z_min"],self.cnc_params["z_max"],self.cnc_params["n_z"])
        self.z_vec = np.concatenate([[0.],self.z_vec])
        self.ns = self.provider.get_param("ns")
        self.z_cmb = 1100.

    def get_linear_power_spectrum(self,redshift):

       # ps = self.provider.get_Pk_interpolator(nonlinear=False)(redshift,self.k_arr)[0,:]

        (k, z, ps) = self.provider.get_Pk_grid(nonlinear=False)
        index = np.abs(z-redshift).argmin()
        ps = np.interp(self.k_arr,k,ps[index,:])

        return (self.k_arr,ps)

    def critical_density(self,z):

        class result:

            conv_fac = self.const.solar/(1000.*self.const.mpc**3)
            Hz = self.H(z).value
            G = 4.301e-9  # Mpc M_sun^-1 (km/s)^2
            rho_crit = 3*Hz**2/(8*np.pi*G)
    
            value = rho_crit*conv_fac

        return result

    def differential_comoving_volume(self,z):

        class result:

            Hz = self.H(z).value  # km/s/Mpc
            DM = self.provider.get_comoving_radial_distance(z)  # Mpc
            c = 299792.458  # km/s
            value = c / Hz * DM**2 

        return result

    def angular_diameter_distance(self,z):
        
        class result:

            value = self.provider.get_angular_diameter_distance(z)

        return result

    def angular_diameter_distance_z1z2(self,z1,z2):

        z_pairs = [(z, z2) for z in z1]

        class result:

            value = self.provider.get_angular_diameter_distance_2(z_pairs)

        return result

    def H(self,z):

        class result:

            value =  self.provider.get_Hubble(z)
        return result

    class H0:
        # def __init__(self):
            # conv_fac = 299792.458
            # class result:
        value = 0
        # return {'hubble':np.vectorize(self.classy.Hubble)(z)}
        # return result

    def Oc(self,z):

        #return np.interp(z,self.z_vec,self.provider.get_Omega_cdm())
        return self.provider.get_Omega_cdm(z)

    def Ob(self,z):

        #return np.interp(z,self.z_vec,self.provider.get_Omega_b())
        return self.provider.get_Omega_b(z)

    def Om(self,z):

        return self.Oc(z)+self.Ob(z)

    def Omega_nu_massive(self,z):

        #return np.interp(z,self.z_vec,self.provider.get_Omega_nu_massive())       
        return self.provider.get_Omega_nu_massive(z)    

    def sigma8(self,z):

        #return np.interp(z,self.z_vec,self.provider.sigma8_z())       
        return self.provider.get_sigma8_z(z)     
