import numpy as np
import cosmocnc
import scipy.integrate as integrate

class scaling_relations:

    def __init__(self,observable="q_mmf3",cnc_params=None,catalogue=None):

        self.logger = cosmocnc.logging.getLogger(__name__)
        self.observable = observable
        self.cnc_params = cnc_params
        self.preprecompute = False
        self.catalogue = catalogue
        self.root_path = cosmocnc.root_path


    def get_n_layers(self):

        observable = self.observable

        if observable == "p_so_sim_stacked":

            n_layers = 1

        else:

            n_layers = 2

        return n_layers

    def initialise_scaling_relation(self,cosmology=None):

        observable = self.observable
        self.const = cosmocnc.constants()

        if observable == "p_so_sim_original" or observable == "p_so_sim_stacked":

            [theta_500_vec,sigma_lens_vec] = np.load(self.root_path + "data/so_sim_lensing_mf_noise.npy")
            theta_500_vec = theta_500_vec*180.*60./np.pi #in arcmin

            self.sigma_theta_lens_vec = np.zeros((1,2,len(theta_500_vec))) #first index is patch index, just 0
            self.sigma_theta_lens_vec[0,0,:] = theta_500_vec
            self.sigma_theta_lens_vec[0,1,:] = sigma_lens_vec

        if observable == "p_so_sim":

            [theta_500_vec,sigma_lens_vec] = np.load(self.root_path + "data/so_sim_lensing_mf_noise.npy")
            theta_500_vec = theta_500_vec*180.*60./np.pi #in arcmin

            x = np.log(theta_500_vec)
            y = np.log(sigma_lens_vec)
            self.sigma_lens_poly = np.polyfit(x,y,deg=3)

            sigma_sz_vec_eval = np.exp(np.polyval(self.sigma_lens_poly,x))

        if observable == "q_so_sim":

            theta_500_vec,sigma_sz_vec = np.load(self.root_path + "data/so_sim_sz_mf_noise.npy")

            self.theta_500_vec = theta_500_vec*180.*60./np.pi

            x = np.log(self.theta_500_vec)
            y = np.log(sigma_sz_vec)
            self.sigma_sz_poly = np.polyfit(x,y,deg=3)
            self.sigma_sz_polyder = np.polyder(self.sigma_sz_poly)

            self.skyfracs = [0.4] #from SO goals and forecasts paper

            #False detection pdf

            q_vec = np.linspace(5.,10.,self.cnc_params["n_points"])
            pdf_fd = np.exp(-(q_vec-3.)**2/1.5**2)
            pdf_fd = pdf_fd/integrate.simpson(pdf_fd,x=q_vec)
            self.pdf_false_detection = [q_vec,pdf_fd]

    def precompute_scaling_relation(self,params=None,other_params=None,patch_index=0):

        observable = self.observable
        self.params = params

        if observable == "p_so_sim" or observable == "p_so_sim_stacked":

            H0 = other_params["H0"]
            E_z = other_params["E_z"]
            D_A = other_params["D_A"]
            D_CMB = other_params["D_CMB"]
            D_l_CMB = other_params["D_l_CMB"]
            rho_c = other_params["rho_c"] # cosmology.critical_density(z_obs).value*1000.*mpc**3/solar
            gamma = self.const.gamma

            c = 3.
            r_s = (3./4./rho_c/500./np.pi/c**3*1e15)**(1./3.)
            rho_0 = rho_c*500./3.*c**3/(np.log(1.+c)-c/(1.+c))
            Sigma_c = 1./(4.*np.pi*D_A*D_l_CMB*gamma)*D_CMB
            R = 5.*c
            factor = r_s*rho_0/Sigma_c
            convergence = 2.*(2.-3.*R+R**3)/(3.*(-1.+R**2)**(3./2.))

            self.prefactor_lens = factor*convergence
            self.prefactor_M_500_to_theta_lensing = 6.997*(H0/70.)**(-2./3.)*(self.params["bias_cmblens"]/3.)**(1./3.)*E_z**(-2./3.)*(500./D_A)

        elif observable == "q_so_sim":

            E_z = other_params["E_z"]
            h70 = other_params["H0"]/70.
            H0 = other_params["H0"]
            D_A = other_params["D_A"]

            A_szifi = self.params["A_szifi"]
            self.prefactor_logy0 = np.log(10.**(A_szifi)*E_z**2*(self.params["bias_sz"]/3.*h70)**self.params["alpha_szifi"]/np.sqrt(h70))
            self.prefactor_M_500_to_theta = 6.997*(H0/70.)**(-2./3.)*(self.params["bias_sz"]/3.)**(1./3.)*E_z**(-2./3.)*(500./D_A)

    def eval_scaling_relation(self,x0,layer=0,patch_index=0,other_params=None):

        observable = self.observable

        if observable == "p_so_sim":

            if layer == 0:

                log_theta_500_lensing = np.log(self.prefactor_M_500_to_theta_lensing) + x0/3.
                log_sigma = np.polyval(self.sigma_lens_poly,log_theta_500_lensing)

                x1 = np.log(self.prefactor_lens*self.params["a_lens"]*(0.1*self.params["bias_cmblens"])**(1./3.)) + x0/3. - log_sigma

            elif layer == 1:

                x1 = np.exp(x0)

        if observable == "p_so_sim_stacked":

            if layer == 0:

                log_theta_500_lensing = np.log(self.prefactor_M_500_to_theta_lensing) + x0/3.
                log_sigma = np.polyval(self.sigma_lens_poly,log_theta_500_lensing)

                x1 = np.log(self.prefactor_lens*self.params["a_lens"]*(0.1*self.params["bias_cmblens"])**(1./3.)) + x0/3. - log_sigma

        if observable == "q_so_sim":

            if layer == 0:

                log_y0 = x0*self.params["alpha_szifi"] + self.prefactor_logy0
                log_theta_500 = np.log(self.prefactor_M_500_to_theta) + x0/3.
                self.log_theta_500 = log_theta_500
                log_sigma_sz = np.polyval(self.sigma_sz_poly,log_theta_500)
                x1 = log_y0 - log_sigma_sz

            if layer == 1:

                x1 = np.sqrt(np.exp(x0)**2+self.params["dof"])

        self.x1 = x1

        return x1

    def eval_derivative_scaling_relation(self,x0,layer=0,patch_index=0,scalrel_type_deriv="analytical"):

        observable = self.observable

        if scalrel_type_deriv == "analytical":

            if observable == "q_so_sim":

                if layer == 0:

                    dx1_dx0 = self.params["alpha_szifi"] - np.polyval(self.sigma_sz_polyder,self.log_theta_500)/3.

                if layer == 1:

                    dof = self.params["dof"]
                    exp = np.exp(2.*x0)
                    dx1_dx0 = exp/np.sqrt(exp+dof)

        elif scalrel_type_deriv == "numerical": #must always be computed strictly after executing self.eval_scaling_relation()

            dx1_dx0 = np.gradient(self.x1,x0)

        return dx1_dx0

    def eval_scaling_relation_no_precompute(self,x0,layer=0,patch_index=0,other_params=None,params=None):

        self.params = params
        self.other_params = other_params
        observable = self.observable

        if observable == "q_so_sim":

            if layer == 0:

                E_z = other_params["E_z"]
                H0 = other_params["H0"]
                h70 = H0/70.
                D_A = other_params["D_A"]
                A_szifi = self.params["A_szifi"]

                prefactor_logy0 = np.log(10.**(A_szifi)*E_z**2*(self.params["bias_sz"]/3.*h70)**self.params["alpha_szifi"]/np.sqrt(h70))
                log_y0 = prefactor_logy0 + x0*self.params["alpha_szifi"]

                prefactor_M_500_to_theta = 6.997*(H0/70.)**(-2./3.)*(self.params["bias_sz"]/3.)**(1./3.)*E_z**(-2./3.)*(500./D_A)

                log_theta_500 = np.log(prefactor_M_500_to_theta) + x0/3.
                log_sigma_sz = np.polyval(self.sigma_sz_poly,log_theta_500)
                x1 = log_y0 - log_sigma_sz

            elif layer == 1:

                x1 = np.sqrt(np.exp(x0)**2+self.params["dof"])

        if observable == "p_so_sim" or observable == "p_so_sim_stacked":

            if layer == 0:

                H0 = other_params["H0"]
                E_z = other_params["E_z"]
                D_A = other_params["D_A"]
                D_CMB = other_params["D_CMB"]
                D_l_CMB = other_params["D_l_CMB"]
                rho_c = other_params["rho_c"] # cosmology.critical_density(z_obs).value*1000.*mpc**3/solar
                gamma = self.const.gamma

                c = 3.
                r_s = (3./4./rho_c/500./np.pi/c**3*1e15)**(1./3.)
                rho_0 = rho_c*500./3.*c**3/(np.log(1.+c)-c/(1.+c))
                Sigma_c = 1./(4.*np.pi*D_A*D_l_CMB*gamma)*D_CMB
                R = 5.*c
                factor = r_s*rho_0/Sigma_c
                convergence = 2.*(2.-3.*R+R**3)/(3.*(-1.+R**2)**(3./2.))

                prefactor_lens = factor*convergence
                prefactor_M_500_to_theta_lensing = 6.997*(H0/70.)**(-2./3.)*(self.params["bias_cmblens"]/3.)**(1./3.)*E_z**(-2./3.)*(500./D_A)


                log_theta_500_lensing = np.log(prefactor_M_500_to_theta_lensing) + x0/3.
                log_sigma = np.polyval(self.sigma_lens_poly,log_theta_500_lensing)

                x1 = np.log(prefactor_lens*self.params["a_lens"]*(0.1*self.params["bias_cmblens"])**(1./3.)) + x0/3. - log_sigma

            elif layer == 1:

                x1 = np.exp(x0)

        return x1


    def get_mean(self,x0,patch_index=0,scatter=None,compute_var=False):

        if self.observable == "p_so_sim":

            log_theta_500_lensing = np.log(self.prefactor_M_500_to_theta_lensing) + x0/3.
            log_sigma = np.polyval(self.sigma_lens_poly,log_theta_500_lensing)

            lnp_mean = np.log(self.prefactor_lens*self.params["a_lens"]*(0.1*self.params["bias_cmblens"])**(1./3.)) + x0/3. - log_sigma
            sigma_intrinsic = np.sqrt(scatter.get_cov(observable1=self.observable,
                                                         observable2=self.observable,
                                                         layer=0,patch1=patch_index,patch2=patch_index))

            mean = np.exp(lnp_mean + sigma_intrinsic**2*0.5)
            ret = mean

            if compute_var == True:

                var_intrinsic = (np.exp(sigma_intrinsic**2)-1.)*np.exp(2.*lnp_mean+sigma_intrinsic**2)
                var_total = var_intrinsic + 1.
                ret = [mean,var_total]

        return ret

    def get_cutoff(self,layer=0):

        if self.observable == "q_so_sim":

            if layer == 0:

                cutoff = -np.inf

            elif layer == 1:

                cutoff = self.params["q_cutoff"]

        return cutoff

class scatter:

    def __init__(self,params=None,catalogue=None):

        self.params = params
        self.catalogue = catalogue

    def get_cov(self,observable1=None,observable2=None,patch1=0,patch2=0,layer=0,other_params=None):

        if layer == 0:

            if observable1 == "p_so_sim" and observable2 == "p_so_sim":

                cov = self.params["sigma_lnp"]**2

            elif observable1 == "p_so_sim" and observable2 == "q_so_sim":

                cov =  self.params["corr_lnq_lnp"]*self.params["sigma_lnq_szifi"]*self.params["sigma_lnp"]

            elif (observable1 == "q_so_sim" and observable2 == "q_so_sim"):

                cov = self.params["sigma_lnq_szifi"]**2

            else:

                cov = 0.

        elif layer == 1:

            if observable1 == "p_so_sim" and observable2 == "p_so_sim":

                cov = 1.

            elif (observable1 == "q_so_sim" and observable2 == "q_so_sim"):

                cov = 1

            else:

                cov = 0.

        return cov