import numpy as np
import pylab as pl
from .config import *

class scaling_relations:

    def __init__(self,observable="q_mmf3"):

        self.observable = observable

    def get_n_layers(self):

        observable = self.observable

        if observable == "q_mmf3" or observable == "q_mmf3_mean":

            n_layers = 2

        return n_layers

    def initialise_scaling_relation(self,params=None):

        observable = self.observable

        if params is None:

            params = scaling_relation_params(observable=observable).params

        self.params = params

        if observable == "q_mmf3" or observable == "q_mmf3_mean":

            f = open(root_path+"/data/noise_planck.txt","r")
            sigma_matrix_flat = np.array(f.readlines()).astype(np.float)
            f.close()

            f = open(root_path+"/data/thetas_planck_arcmin.txt","r")
            self.theta_500_vec = np.array(f.readlines()).astype(np.float)
            f.close()

            f = open(root_path+"/data/skyfracs_planck.txt","r")
            self.skyfracs = np.array(f.readlines()).astype(np.float)
            f.close()

            self.sigma_matrix = sigma_matrix_flat.reshape((80,417))

        if observable == "q_mmf3_mean":

            sigma_matrix_0 = np.zeros((self.sigma_matrix.shape[0],1))
            sigma_matrix_0[:,0] = np.average(self.sigma_matrix,axis=1,weights=self.skyfracs)
            self.sigma_matrix = sigma_matrix_0
            self.skyfracs = np.array([np.sum(self.skyfracs)])

    def precompute_scaling_relation(self,other_params=None,layer=0,patch_index=0):

        observable = self.observable

        if observable == "q_mmf3" or observable == "q_mmf3_mean":

            H0 = other_params["H0"]
            E_z = other_params["E_z"]
            D_A = other_params["D_A"]

            self.prefactor_Y_500 = ((1.-self.params["b"])/6e14)**self.params["alpha"]*(H0/70.)**(-2.+self.params["alpha"])*self.params["Y_star"]*E_z**self.params["beta"]*0.00472724*(D_A/500.)**(-2.)
            self.prefactor_M_500_to_theta = 6.997*(H0/70.)**(-2./3.)*((1.-self.params["b"])/3e14)**(1./3.)*E_z**(-2./3.)*(500./D_A)

    def eval_scaling_relation(self,x0,observable="q_mmf3",layer=0,patch_index=0,other_params=None):

        observable = self.observable

        if observable == "q_mmf3" or observable == "q_mmf3_mean":

            if layer == 0:

                #x0 is ln M_500

                self.M_500 = np.exp(x0)
                Y_500 = self.prefactor_Y_500*self.M_500**self.params["alpha"]
                self.theta_500 = self.prefactor_M_500_to_theta*self.M_500**(1./3.)
                sigma_vec = self.sigma_matrix[:,patch_index]

                sigma = np.interp(self.theta_500,self.theta_500_vec,sigma_vec)
                x1 = np.log(Y_500/sigma)

                #x1 is log q_mean

            if layer == 1:

                #x0 is log q_true

                x1 = np.exp(x0)

                #x1 is q_true

        if observable == "m_lens":

            if layer == 0:

                lensing_bias = 0.9
                x1 = np.log(lensing_bias) + x0 - np.log(1e15)

            if layer == 1:

                x1 = np.exp(x0)

        return x1

    def eval_derivative_scaling_relation(self,x0,observable="q_mmf3",layer=0,patch_index=0):

        observable = self.observable

        if observable == "q_mmf3" or observable == "q_mmf3_mean":

            if layer == 0:

                #x0 is ln M_500, returns d ln q_mean / d ln M_500
                sigma_vec = self.sigma_matrix[:,patch_index]
                log_sigma_vec_derivative = np.interp(np.log(self.theta_500),np.log(self.theta_500_vec),np.gradient(np.log(sigma_vec),np.log(self.theta_500_vec)))
                dx1_dx0 = self.params["alpha"] - log_sigma_vec_derivative/3.

            if layer == 1:

                #x0 is log q_true, x1 is q_true, returns q_true

                dx1_dx0 = np.exp(x0)

        if observable == "m_lens":

            if layer == 0:

                dx1_dx0 = 1.

            elif layer == 1:

                dx1_dx0 = np.exp(x0)

        return dx1_dx0


class scaling_relation_params:

    def __init__(self,observable="q_mmf3"):

        print("observable",observable)

        if observable == "q_mmf3" or observable == "q_mmf3_mean":

            self.params = {"alpha":1.78,
            "beta":0.66,
            "Y_star":10.**(-0.186),
            "sigma_lnY":0.173,
            "b":0.2}

class scatter:

    def __init__(self):

        a = 1.

    def get_cov(self,observable1="q_mmf3",observable2="q_mmf3",patch1=0,patch2=0,layer=0):

        if layer == 0:

            if observable1 == "q_mmf3" and observable2 == "q_mmf3":

                cov = 0.173**2

            elif observable1 == "q_mmf3_mean" and observable2 == "q_mmf3_mean":

                cov = 0.173**2

            elif observable1 == "m_lens" and observable2 == "m_lens":

                cov = 0.2**2

            elif (observable1 == "q_mmf3_mean" and observable2 == "m_lens") or (observable1 == "m_lens" and observable2 == "q_mmf3_mean"):

                cov = 0.

            else:

                cov = 0.

        elif layer == 1:

            if observable1 == "q_mmf3" and observable2 == "q_mmf3":

                cov = 1.

            elif observable1 == "q_mmf3_mean" and observable2 == "q_mmf3_mean":

                cov = 1.

            elif observable1 == "m_lens" and observable2 == "m_lens":

                cov = 0.05**2

            else:

                cov = 0.

        return cov
