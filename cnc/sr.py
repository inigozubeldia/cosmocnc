import numpy as np
import pylab as pl
from .hmf import *
from .params import *
from .config import *

class scaling_relations:

    def __init__(self,observable="q_mmf3",cnc_params = None):

        self.observable = observable
        self.cnc_params = cnc_params
        self.preprecompute = False

    def get_n_layers(self):

        observable = self.observable

        if observable == "q_mmf3" or observable == "q_mmf3_mean" or observable == "p_zc19" or observable == 'xi':

            n_layers = 2

        elif observable == "m_lens":

            n_layers = 2

        return n_layers

    def initialise_scaling_relation(self):

        observable = self.observable
        self.const = constants()

        if observable == "q_mmf3" or observable == "q_mmf3_mean":

            f = open(root_path + "data/noise_planck.txt","r")
            sigma_matrix_flat = np.array(f.readlines()).astype(np.float)
            f.close()

            f = open(root_path + "data/thetas_planck_arcmin.txt","r")
            self.theta_500_vec = np.array(f.readlines()).astype(np.float)
            f.close()

            f = open(root_path + "data/skyfracs_planck.txt","r")
            self.skyfracs = np.array(f.readlines()).astype(np.float)
            f.close()

            self.sigma_matrix = sigma_matrix_flat.reshape((80,417))

        if observable == "q_mmf3_mean":

            sigma_matrix_0 = np.zeros((self.sigma_matrix.shape[0],1))
            sigma_matrix_0[:,0] = np.average(self.sigma_matrix,axis=1,weights=self.skyfracs)
            self.sigma_matrix = sigma_matrix_0
            self.skyfracs = np.array([np.sum(self.skyfracs)])

        if observable == "p_zc19":

            self.sigma_theta_lens_vec = np.load(root_path + "data/sigma_theta_lens_vec.npy") #first index is patch index, from 0 to 417
            self.sigma_theta_lens_vec[:,0,:] = self.sigma_theta_lens_vec[:,0,:]*180.*60./np.pi #in arcmin

        # SPT case:
        if observable == 'xi':
            # print('dealing with xi sr')
            # this is pasted from Bocquet's SPT_SZ_cluster_likelihood/SPTcluster_data.py
            SPTfieldSize = (82.8711, 100.241, 147.589, 222.647, 189.955, 155.547, 156.243,
                155.731, 145.888, 102.657, 83.2849, 166.812, 70.4952, 111.217, 108.625,
                83.6339, 204.453, 102.832, 68.6716)
            self.skyfracs = np.asarray(SPTfieldSize)/2500
            # print('self.skyfracs:',len(self.skyfracs))
            # print('self.skyfracs:',self.skyfracs)

            # this is pasted from Bocquet's SPT_SZ_cluster_likelihood/SPTcluster_data.py
            SPTfieldCorrection = (1.3267815, 1.3875717, 1.2923182, 1.2479916, 1.1095432,
                1.2668965, 1.1357954, 1.1901025, 1.1754438, 1.0798830, 1.1631297, 1.1999745,
                1.1762851, 1.1490106, 1.1916442, 1.1307591, 1.1864938, 1.1629247, 1.1823912)
            # the spt field correction is what multiplies the zeta[z,M] relation
            self.SPTfieldCorrection = np.asarray(SPTfieldCorrection)


    def preprecompute_scaling_relation(self,params=None,other_params=None):

        if params is None:

            params = scaling_relation_params_default

        self.params = params
        self.preprecompute = True

        if self.observable == "q_mmf3" or self.observable == "q_mmf3_mean":

            self.M_500 = np.exp(other_params["lnM"])
            self.M_500_alpha = self.M_500**self.params["alpha"]
            self.M_500_13 = self.M_500**(1./3.)


    def precompute_scaling_relation(self,params=None,other_params=None,layer=0,patch_index=0):

        if params is None:

            params = scaling_relation_params_default

        self.params = params

        observable = self.observable

        if observable == "q_mmf3" or observable == "q_mmf3_mean":

            H0 = other_params["H0"]
            E_z = other_params["E_z"]
            D_A = other_params["D_A"]

            self.prefactor_Y_500 = (self.params["bias_sz"]/6.)**self.params["alpha"]*(H0/70.)**(-2.+self.params["alpha"])*10.**self.params["log10_Y_star"]*E_z**self.params["beta"]*0.00472724*(D_A/500.)**(-2.)
            self.prefactor_M_500_to_theta = 6.997*(H0/70.)**(-2./3.)*(self.params["bias_sz"]/3.)**(1./3.)*E_z**(-2./3.)*(500./D_A)

        if observable == "p_zc19":

            H0 = other_params["H0"]
            E_z = other_params["E_z"]
            D_A = other_params["D_A"]
            D_CMB = other_params["D_CMB"]
            D_l_CMB = other_params["D_l_CMB"]
            rho_c = other_params["rho_c"] # cosmology.critical_density(z_obs).value*1000.*mpc**3/solar
            gamma = self.const.gamma

            c = 3.
            r_s = (3./4./rho_c/500./np.pi/c**3*10.**15)**(1./3.)
            rho_0 = rho_c*500./3.*c**3/(np.log(1.+c)-c/(1.+c))
            Sigma_c = 1./(4.*np.pi*D_A*D_l_CMB*gamma)*D_CMB
            R = 5.*c
            factor = r_s*rho_0/Sigma_c
            convergence = 2.*(2.-3.*R+R**3)/(3.*(-1.+R**2)**(3./2.))

            self.prefactor_lens = factor*convergence
            self.prefactor_M_500_to_theta_lensing = 6.997*(H0/70.)**(-2./3.)*(self.params["bias_cmblens"]/3.)**(1./3.)*E_z**(-2./3.)*(500./D_A)

        if observable == 'xi':
            # print('precomputing quantities for spt xi-m scaling relations')
            # SPT code snipet:
            # massterm = (mass/self.SZmPivot)**self.Bsz
            # zterm = (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.Csz
            # return self.Asz * massterm[None,:] * zterm[:,None]
            # end of SPT code snipet.

            E_z = other_params["E_z"]
            E_z0p6 = other_params["E_z0p6"]
            self.prefactor_xi =  self.params["A_sz"]*(E_z/E_z0p6)**self.params["C_sz"]
            # print(self.prefactor_xi,E_z,E_z0p6)
            # exit(0)

    def eval_scaling_relation(self,x0,layer=0,patch_index=0,other_params=None,direction="forward"):

        observable = self.observable

        if observable == "q_mmf3" or observable == "q_mmf3_mean":

            if layer == 0:

                #x0 is ln M_500

                if self.preprecompute == False:

                    self.M_500 = np.exp(x0)
                    Y_500 = self.prefactor_Y_500*self.M_500**self.params["alpha"]
                    self.theta_500 = self.prefactor_M_500_to_theta*self.M_500**(1./3.)

                elif self.preprecompute == True:

                    Y_500 = self.prefactor_Y_500*self.M_500_alpha
                    self.theta_500 = self.prefactor_M_500_to_theta*self.M_500_13

                sigma_vec = self.sigma_matrix[:,patch_index]

                sigma = np.interp(self.theta_500,self.theta_500_vec,sigma_vec)
                x1 = np.log(Y_500/sigma)

                #x1 is log q_mean

            if layer == 1:

                #x0 is log q_true

                x1 = np.sqrt(np.exp(x0)**2+self.params["dof"])

                #x1 is q_true

        if observable == "m_lens":

            if layer == 0:

                x1 = np.log(self.params["bias_cmblens"]) + x0

            if layer == 1:

                x1 = np.exp(x0)

        if observable == "p_zc19":

            if layer == 0:

                M_500 = np.exp(x0)
                self.theta_500_lensing = self.prefactor_M_500_to_theta_lensing*M_500**(1./3.)

                sigma = np.interp(self.theta_500_lensing,self.sigma_theta_lens_vec[patch_index,0,:],self.sigma_theta_lens_vec[patch_index,1,:])
                x1 = np.log((M_500*0.1*self.params["bias_cmblens"])**(1./3.)*self.prefactor_lens/sigma*self.params["a_lens"]) #with change from 1e14 to 1e15 units

            elif layer == 1:

                x1 = np.exp(x0)

        if observable == 'xi':
            if layer == 0:
                # print('evaluating xi-m relation spt case')
                #x0 is ln M_500
                M_500 = np.exp(x0) #### Msun
                # print('M_500',M_500)
                # exit(0)
                xi = self.prefactor_xi*(M_500*1e14/self.cnc_params['SZmPivot'])**self.params["B_sz"]
                sigma = 1./self.SPTfieldCorrection[patch_index]
                x1 = np.log(xi/sigma)
                # print('x1: ',np.exp(x1))
                # exit(0)
                # print('self.params["dof"] : ',self.params["dof"])
                # exit(0)
            elif layer == 1:
                # print('evaluating true sz snr spt case')
                #x0 is log q_true
                x1 = np.sqrt(np.exp(x0)**2+self.params["dof"])
                #x1 is q_true
                # print('self.params["dof"] : ',self.params["dof"])
                # exit(0)


        self.x1 = x1

        return x1

    def eval_derivative_scaling_relation(self,x0,layer=0,patch_index=0,scalrel_type_deriv="analytical"):

        observable = self.observable

        if scalrel_type_deriv == "analytical":

            if observable == "q_mmf3" or observable == "q_mmf3_mean":

                if layer == 0:

                    #x0 is ln M_500, returns d ln q_mean / d ln M_500
                    sigma_vec = self.sigma_matrix[:,patch_index]
                    log_sigma_vec_derivative = np.interp(np.log(self.theta_500),np.log(self.theta_500_vec),np.gradient(np.log(sigma_vec),np.log(self.theta_500_vec)))
                    dx1_dx0 = self.params["alpha"] - log_sigma_vec_derivative/3.

                if layer == 1:

                    #x0 is log q_true, x1 is q_true, returns q_true (including optimisation correction)
                    #dx1_dx0 = np.exp(x0)

                    dof = self.params["dof"]
                    exp = np.exp(2.*x0)
                    dx1_dx0 = exp/np.sqrt(exp+dof)

            if observable == "m_lens":

                if layer == 0:

                    dx1_dx0 = 1.

                elif layer == 1:

                    dx1_dx0 = np.exp(x0)

            if observable == "p_zc19":

                if layer == 0:

                    theta_vec = self.sigma_theta_lens_vec[patch_index,0,:]
                    sigma_vec = self.sigma_theta_lens_vec[patch_index,1,:]
                    log_sigma_vec_derivative = np.interp(np.log(self.theta_500_lensing),np.log(sigma_vec),np.gradient(np.log(sigma_vec),np.log(theta_vec)))
                    dx1_dx0 = 1./3. - log_sigma_vec_derivative/3.

                elif layer == 1:

                    dx1_dx0 = np.exp(x0)

        elif scalrel_type_deriv == "numerical": #must always be computed strictly after executing self.eval_scaling_relation()

            dx1_dx0 = np.gradient(self.x1,x0)

            ###3 check this and remove the case TBD
            if observable == "xi" and layer == 1:
                dof = self.params["dof"]
                exp = np.exp(2.*x0)
                dx1_dx0 = exp/np.sqrt(exp+dof)

        return dx1_dx0

class covariance_matrix:

    def __init__(self,scatter,observables,observable_patches,layer=[0,1]):#

        self.layer = layer
        self.cov = []
        self.inv_cov = []

        for k in range(0,len(self.layer)):

            cov_matrix = np.zeros((len(observables),len(observables)))

            for i in range(0,len(observables)):

                for j in range(0,len(observables)):

                    cov_matrix[i,j] = scatter.get_cov(observable1=observables[i],
                    observable2=observables[j],patch1=observable_patches[observables[i]],
                    patch2=observable_patches[observables[j]],layer=self.layer[k])

            self.cov.append(cov_matrix)
            self.inv_cov.append(np.linalg.inv(cov_matrix))

class scatter:

    def __init__(self,params=None):

        if params is None:

            params = scaling_relation_params_default

        self.params = params

    def get_cov(self,observable1="q_mmf3",observable2="q_mmf3",patch1=0,patch2=0,layer=0):

        if layer == 0:

            if observable1 == "q_mmf3" and observable2 == "q_mmf3":

                cov = self.params["sigma_lnq"]**2

            elif observable1 == "xi" and observable2 == "xi":

                cov = self.params["sigma_lnq"]**2

            elif observable1 == "q_mmf3_mean" and observable2 == "q_mmf3_mean":

                cov = self.params["sigma_lnq"]**2

            elif observable1 == "m_lens" and observable2 == "m_lens":

                cov =  self.params["sigma_lnmlens"]**2

            elif (observable1 == "q_mmf3_mean" and observable2 == "m_lens") or (observable1 == "m_lens" and observable2 == "q_mmf3_mean"):

                cov = 0.

            elif observable1 == "p_zc19" and observable2 == "p_zc19":

                cov = self.params["sigma_lnp"]**2

            elif (observable1 == "q_mmf3" and observable2 == "p_zc19") or (observable1 == "p_zc19" and observable2 == "q_mmf3"):

                cov = self.params["corr_lnq_lnp"]*self.params["sigma_lnp"]*self.params["sigma_lnq"]

            elif (observable1 == "q_mmf3_mean" and observable2 == "p_zc19") or (observable1 == "p_zc19" and observable2 == "q_mmf3_mean"):

                cov = self.params["corr_lnq_lnp"]*self.params["sigma_lnp"]*self.params["sigma_lnq"]

            else:

                cov = 0.

        elif layer == 1:

            if observable1 == "q_mmf3" and observable2 == "q_mmf3":

                cov = 1.

            elif observable1 == "xi" and observable2 == "xi":

                cov = 1.

            elif observable1 == "q_mmf3_mean" and observable2 == "q_mmf3_mean":

                cov = 1.

            elif observable1 == "m_lens" and observable2 == "m_lens":

                cov = self.params["sigma_mlens"]**2

            elif observable1 == "p_zc19" and observable2 == "p_zc19":

                cov = 1.

            else:

                cov = 0.

        return cov
