import numpy as np
import pylab as pl
from .hmf import *
from .params import *
from .config import *
from numpy.lib import scimath as sm

class scaling_relations:

    def __init__(self,observable="q_mmf3",cnc_params = None):

        self.observable = observable
        self.cnc_params = cnc_params
        self.preprecompute = False

    # we should code this a bit more compactly (BB)
    def get_n_layers(self):

        observable = self.observable

        if observable == "q_mmf3" or observable == "q_mmf3_mean" or observable == "p_zc19" or observable == 'xi':

            n_layers = 2

        elif observable == "Yx":

            n_layers = 2

        elif observable == "WLMegacam":

            n_layers = 2

        elif observable == "WLHST":

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

            # this is pasted from Bocquet's SPT_SZ_cluster_likelihood/SPTcluster_data.py
            SPTfieldSize = (82.8711, 100.241, 147.589, 222.647, 189.955, 155.547, 156.243,
                            155.731, 145.888, 102.657, 83.2849, 166.812, 70.4952, 111.217, 108.625,
                            83.6339, 204.453, 102.832, 68.6716)
            self.skyfracs = np.asarray(SPTfieldSize)/2500


            # this is pasted from Bocquet's SPT_SZ_cluster_likelihood/SPTcluster_data.py
            SPTfieldCorrection = (1.3267815, 1.3875717, 1.2923182, 1.2479916, 1.1095432,
                1.2668965, 1.1357954, 1.1901025, 1.1754438, 1.0798830, 1.1631297, 1.1999745,
                1.1762851, 1.1490106, 1.1916442, 1.1307591, 1.1864938, 1.1629247, 1.1823912)
            # the spt field correction is what multiplies the zeta[z,M] relation
            self.SPTfieldCorrection = np.asarray(SPTfieldCorrection)


    def preprecompute_scaling_relation(self,
                                       params=None,
                                       catalog=None,
                                       other_params=None):

        if params is None:

            params = scaling_relation_params_default

        self.params = params
        self.preprecompute = True

        if self.observable == "q_mmf3" or self.observable == "q_mmf3_mean":

            self.M_500 = np.exp(other_params["lnM"])
            self.M_500_alpha = self.M_500**self.params["alpha"]
            self.M_500_13 = self.M_500**(1./3.)


    def precompute_scaling_relation(self,
                                    params=None,
                                    catalog=None,
                                    other_params=None,
                                    layer=0,
                                    patch_index=0):

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

            E_z = other_params["E_z"]
            E_z0p6 = other_params["E_z0p6"]
            self.prefactor_xi =  self.params["A_sz"]*(E_z/E_z0p6)**self.params["C_sz"]


        if observable == 'Yx':
            h = other_params["H0"]/100.
            E_z = other_params["E_z"]
            self.prefactor_Yx = 3 * (h/.7)**-2.5 \
                                * (1. /.7**(3/2) / self.params['A_x'] \
                                / E_z**self.params['C_x'])**(1/self.params['B_x'])

        if observable == 'WLMegacam':

            massModelErr = (catalog.WLcalib['MegacamSim'][1]**2 + catalog.WLcalib['MegacamMcErr']**2 + catalog.WLcalib['MegacamCenterErr']**2)**.5
            zDistShearErr = (catalog.WLcalib['MegacamzDistErr']**2 + catalog.WLcalib['MegacamShearErr']**2 + catalog.WLcalib['MegacamContamCorr']**2)**.5
            self.params['bWL_Megacam'] = catalog.WLcalib['MegacamSim'][0] + self.params['WLbias']*massModelErr + self.params['MegacamBias']*zDistShearErr
            self.params["sigma_lnWLMegacam"] = catalog.WLcalib['MegacamSim'][2]+self.params['WLscatter']*catalog.WLcalib['MegacamSim'][3] # 'DWL_Megacam' in Bocquet's code



        if observable == 'WLHST':

            massModelErr = catalog.catalogue['WLdata'][patch_index]['massModelErr']
            zDistShearErr = catalog.catalogue['WLdata'][patch_index]['zDistShearErr']
            name = catalog.catalogue['SPT_ID'][patch_index]
            self.params['bWL_HST'] = catalog.WLcalib['HSTsim'][name][0] + self.params['WLbias']*massModelErr + self.params['HSTbias']*zDistShearErr
            self.params["sigma_lnWLHST"] = catalog.WLcalib['HSTsim'][name][2]+self.params['WLscatter']*catalog.WLcalib['HSTsim'][name][3] # 'DWL_Megacam' in Bocquet's code


    def eval_scaling_relation(self,
                              x0,
                              catalog=None,
                              layer=0,
                              patch_index=0,
                              other_params=None,
                              direction="forward"):

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

                h = other_params["H0"]/100.

                M_500 = np.exp(x0)*h*1e14 #### Msun/h

                xi = self.prefactor_xi*(M_500/self.cnc_params['SZmPivot'])**self.params["B_sz"]
                sigma = 1./self.SPTfieldCorrection[patch_index]
                x1 = np.log(xi/sigma)

            elif layer == 1:

                #x0 is log q_true
                x1 = np.sqrt(np.exp(x0)**2+self.params["dof"]) ### PDF of OBS vs TRUE -- normal distribution, with mean = x1 and std-dev = 1


        if observable == 'Yx':
            if layer == 0:

                h = other_params["H0"]/100.
                #x0 is ln M_500
                M_500 = np.exp(x0)*h*1e14 #### Msun/h
                x1 = self.prefactor_Yx*(M_500/1e14)**(1/self.params['B_x'])

                # print('now adding radial dependence:')
                # here we implement aq. 19 of Bocquet et al.
                # # Angular diameter distances in current and reference cosmology [Mpc]
                # dA = cosmo.dA(self.catalog['redshift'][dataID], self.cosmology)/self.cosmology['h']
                dA = other_params['D_A'] # in Mpc
                # print('dA:',dA)
                # dAref = cosmo.dA(self.catalog['redshift'][dataID], cosmologyRef)/cosmologyRef['h']
                zcluster = other_params['zc']
                cosmoRef = other_params['cosmology'].spt_cosmoRef_masscal
                dAref = np.exp(np.interp(np.log(1.+zcluster),
                                         other_params['cosmology'].spt_ln1pzs_masscal,
                                         other_params['cosmology'].spt_lndas_hmpc_masscal))/cosmoRef['h']
                # print('dAref',dAref)
                # # R500 [kpc]
                # rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.catalog['redshift'][dataID], self.cosmology)**2
                rho_c_hunits = other_params['rho_c']/h**2
                # print('rho_c_hunits = %.5e'%rho_c_hunits)
                # print('M_obsArr',M_500)
                # exit(0)
                # r500 = 1000 * (3*M_obsArr/(4*np.pi*500*rho_c_z))**(1/3) / self.cosmology['h']
                r500 = 1000 * (3*M_500/(4*np.pi*500*rho_c_hunits))**(1/3) / h
                # # r500 in reference cosmology [kpc]
                r500ref = r500 * dAref/dA
                # print('r500,dAref,dA:',patch_index,r500,dAref,dA)
                # print('r500,r500ref',r500,r500ref)
                # # Xray observable at fiducial r500...
                # obsArr*= (self.catalog['r500'][dataID]/r500ref)**self.scaling['dlnMg_dlnr']
                rcorr = (catalog.catalogue['r500'][patch_index]/r500ref)**self.params['dlnMg_dlnr']
                # print('r500,r500ref',catalog.catalogue['r500'][patch_index],r500ref,self.params['dlnMg_dlnr'])
                # print('x1,rcorr:',x1,rcorr)
                # exit(0)
                x1 *= rcorr
                # # ... corrected to reference cosmology
                # print('adding radial dep for Xray obs: %s %.3e'%(obsname,self.scaling['dlnMg_dlnr']))
                # obsArr*= (dAref/dA)**2.5
                # exit(0)


                x1 = np.log(x1) ## scaling relation


            elif layer == 1:
                # x0 is lnYx(M)
                x1 = np.exp(x0) ### OBS vs TRUE -- normal distribution, with mean = x1 = Yx-obs and std-dev = Yx_std

        if observable == 'WLMegacam':
            if layer == 0:

                #x0 is ln M_500
                M_500 = np.exp(x0) #### Msun
                x1 = M_500*self.params['bWL_Megacam'] # eq. 3 in 1812.01679 (Bocquet et al)
                x1 = np.log(x1)


            elif layer == 1:
                # x0 is lnMwl

                x1 = np.exp(x0)*1e14
                h = other_params["H0"]/100.
                rho_c_hunits = other_params['rho_c']/h**2
                self.rho_c_z =  rho_c_hunits
                Dl = other_params['D_A']*h

                self.get_beta(catalog,patch_index,other_params)


                ##### M200 and scale radius, wrt critical density, everything in h units
                mArr = x1*h # mass in msun/h
                zcluster = other_params['zc']
                m200c = other_params['cosmology'].get_m500c_to_m200c_at_z_and_M(zcluster,mArr)
                r200c = (3.*m200c/4./np.pi/200./rho_c_hunits)**(1./3.)
                c200c = other_params['cosmology'].get_c200c_at_m_and_z(m200c,zcluster)

                self.delta_c = 200./3. * c200c**3. / (np.log(1.+c200c) - c200c/(1.+c200c))
                self.rs = r200c/c200c

                Sigma_c = 1.6624541593797974e+18/Dl/self.beta_avg


                ##### dimensionless radial distance [Radius][Mass]
                self.x_2d = catalog.catalogue['WLdata'][patch_index]['r_deg'][:,None] * Dl * np.pi/180. / self.rs[None,:]

                # gamma_t [Radius][Mass]
                gamma_2d = self.get_Delta_Sigma() / Sigma_c

                # kappa [Radius][Mass]
                kappa_2d = self.get_Sigma() / Sigma_c



                # Reduced shear g_t [Radius][Mass]
                g_2d = gamma_2d/(1-kappa_2d) * (1 + kappa_2d*(self.beta2_avg/self.beta_avg**2-1))

                # Keep all radial bins (make cut in data)
                rInclude = range(len(catalog.catalogue['WLdata'][patch_index]['r_deg']))
                self.rInclude = rInclude

                x1 = g_2d[rInclude,:]


        if observable == 'WLHST':
            if layer == 0:

                #x0 is ln M_500
                M_500 = np.exp(x0) #### Msun
                x1 = M_500*self.params['bWL_HST']
                x1 = np.log(x1)


            elif layer == 1:
                # x0 is lnMwl
                x1 = np.exp(x0)*1e14
                h = other_params["H0"]/100.
                rho_c_hunits = other_params['rho_c']/h**2
                self.rho_c_z =  rho_c_hunits
                Dl = other_params['D_A']*h

                self.get_beta(catalog,patch_index,other_params)


                ##### M200 and scale radius, wrt critical density, everything in h units
                mArr = x1*h # mass in msun/h

                zcluster = other_params['zc']

                m200c = other_params['cosmology'].get_m500c_to_m200c_at_z_and_M(zcluster,mArr)

                r200c = (3.*m200c/4./np.pi/200./rho_c_hunits)**(1./3.)

                c200c = other_params['cosmology'].get_c200c_at_m_and_z(m200c,zcluster)

                self.delta_c = 200./3. * c200c**3. / (np.log(1.+c200c) - c200c/(1.+c200c))
                self.rs = r200c/c200c


                ##### dimensionless radial distance [Radius][Mass]
                self.x_2d = catalog.catalogue['WLdata'][patch_index]['r_deg'][:,None] * Dl * np.pi/180. / self.rs[None,:]

                # Sigma_crit, with c^2/4piG [h Msun/Mpc^2] [Radius]
                rangeR = range(len(catalog.catalogue['WLdata'][patch_index]['r_deg']))
                betaR = np.array([self.beta_avg[catalog.catalogue['WLdata'][patch_index]['magbinids'][i]] for i in rangeR])
                beta2R = np.array([self.beta2_avg[catalog.catalogue['WLdata'][patch_index]['magbinids'][i]] for i in rangeR])
                Sigma_c = 1.6624541593797974e+18/Dl/betaR


                # gamma_t [Radius][Mass]
                gamma_2d = self.get_Delta_Sigma() / Sigma_c[:,None]

                # kappa [Radius][Mass]
                kappa_2d = self.get_Sigma() / Sigma_c[:,None]

                # [Radius][Mass]
                mu0_2d = 1./((1.-kappa_2d)**2 - gamma_2d**2)
                kappaFake = (mu0_2d-1)/2.

                # Magnification correction [Radius][Mass]
                mykappa = kappaFake * 0.3/betaR[:,None]

                magcorr = [np.interp(mykappa[i],
                                     catalog.catalogue['WLdata'][patch_index]['magcorr'][catalog.catalogue['WLdata'][patch_index]['magbinids'][i]][0],
                                     catalog.catalogue['WLdata'][patch_index]['magcorr'][catalog.catalogue['WLdata'][patch_index]['magbinids'][i]][1]) for i in rangeR]

                # Beta correction [Radius][Mass]
                betaratio = beta2R/betaR**2
                betaCorr = (1 + kappa_2d*(betaratio[:,None]-1))

                # Reduced shear g_t [Radius][Mass]
                g_2d = np.array(magcorr) * gamma_2d/(1-kappa_2d) * betaCorr


                # Only consider 500<r/kpc/1500 in reference cosmology
                cosmoRef = other_params['cosmology'].spt_cosmoRef

                DlRef = np.exp(np.interp(np.log(1.+zcluster),
                                         other_params['cosmology'].spt_ln1pzs,
                                         other_params['cosmology'].spt_lndas_hmpc))
                rPhysRef = catalog.catalogue['WLdata'][patch_index]['r_deg'] * DlRef * np.pi/180. /cosmoRef['h']
                rInclude = np.where((rPhysRef>.5)&(rPhysRef<1.5))[0]


                self.rInclude = rInclude
                x1 = g_2d[rInclude,:]



        self.x1 = x1

        return x1


    ########################################
    def get_beta(self,catalog,patch_index,other_params):
        """Compute <beta> and <beta^2> from distribution of redshift galaxies."""
        ##### Only consider redshift bins behind the cluster
        betaArr = np.zeros(len(catalog.catalogue['WLdata'][patch_index]['redshifts']))
        zcluster = other_params['zc']
        bgIdx = np.where(catalog.catalogue['WLdata'][patch_index]['redshifts']>zcluster)[0]

        cosmo = other_params['cosmology']
        h = other_params['cosmology'].background_cosmology.H0.value/100.
        betaArr[bgIdx] = np.array([cosmo.background_cosmology.angular_diameter_distance_z1z2(zcluster,z).value*h for z in catalog.catalogue['WLdata'][patch_index]['redshifts'][bgIdx]])

        zs = catalog.catalogue['WLdata'][patch_index]['redshifts'][bgIdx]
        DA_zs = cosmo.background_cosmology.angular_diameter_distance(zs).value*h


        betaArr[bgIdx]/=DA_zs

        # ##### Weight beta(z) with N(z) distribution to get <beta> and <beta^2>
        if catalog.catalogue['WLdata'][patch_index]['datatype']!='HST':
            self.beta_avg = np.sum(catalog.catalogue['WLdata'][patch_index]['Nz']*betaArr)/catalog.catalogue['WLdata'][patch_index]['Ntot']
            self.beta2_avg = np.sum(catalog.catalogue['WLdata'][patch_index]['Nz']*betaArr**2)/catalog.catalogue['WLdata'][patch_index]['Ntot']
        else:
            self.beta_avg, self.beta2_avg = {}, {}
            for i in catalog.catalogue['WLdata'][patch_index]['pzs'].keys():
                self.beta_avg[i] = np.sum(catalog.catalogue['WLdata'][patch_index]['pzs'][i]*betaArr)/catalog.catalogue['WLdata'][patch_index]['Ntot'][i]
                self.beta2_avg[i] = np.sum(catalog.catalogue['WLdata'][patch_index]['pzs'][i]*betaArr**2)/catalog.catalogue['WLdata'][patch_index]['Ntot'][i]




    ########################################
    ##### Delta Sigma[Radius][Mass]
    # originally by Joerg Dietrich
    # adapted from Bocquet's code
    def get_Delta_Sigma(self):
        fac = 2 * self.rs * self.rho_c_z * self.delta_c
        val1 = 1. / (1 - self.x_2d**2)
        num = ((3 * self.x_2d**2) - 2) * self.arcsec(self.x_2d)
        div = self.x_2d**2 * (sm.sqrt(self.x_2d**2 - 1))**3
        val2 = (num / div).real
        val3 = 2 * np.log(self.x_2d / 2) / self.x_2d**2
        result = fac * (val1+val2+val3)
        return result



    ########################################
    ##### Sigma_NFW[Radius][Mass]
    # by Joerg Dietrich
    #from Bocquet's code
    def get_Sigma(self):
        val1 = 1. / (self.x_2d**2 - 1)
        val2 = (self.arcsec(self.x_2d) / (sm.sqrt(self.x_2d**2 - 1))**3).real
        return 2 * self.rs * self.rho_c_z * self.delta_c * (val1-val2)




    ########################################
    ##### Compute the inverse sec of the complex number z.
    # by Joerg Dietrich
    # from the spt code (Bocquet et al)
    def arcsec(self, z):
        val1 = 1j / z
        val2 = sm.sqrt(1 - 1./z**2)
        val = 1j * np.log(val2 + val1)
        return .5 * np.pi + val


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
            covdim = len(observables)
            cov_matrix = np.zeros((covdim,covdim))

            for i in range(0,covdim):

                for j in range(0,covdim):

                    cov_matrix[i,j] = scatter.get_cov(observable1=observables[i],
                                                      observable2=observables[j],
                                                      patch1=observable_patches[observables[i]],
                                                      patch2=observable_patches[observables[j]],
                                                      layer=self.layer[k])

            self.cov.append(cov_matrix)
            self.inv_cov.append(np.linalg.inv(cov_matrix))

class scatter:

    def __init__(self,params=None,catalog=None):

        if params is None:

            params = scaling_relation_params_default

        self.params = params
        self.catalog = catalog

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

            elif (observable1 == "xi" and observable2 == "Yx") or (observable2 == "xi" and observable1 == "Yx"):

                cov = self.params["corr_xi_Yx"]*self.params["sigma_lnYx"]*self.params["sigma_lnq"] ### patch1 is the cluster index

            elif observable1 == "Yx" and observable2 == "Yx":

                cov = self.params["sigma_lnYx"]**2

            elif observable1 == "WLMegacam" and observable2 == "WLMegacam":

                cov = self.params["sigma_lnWLMegacam"]**2

            elif (observable1 == "xi" and observable2 == "WLMegacam") or (observable2 == "xi" and observable1 == "WLMegacam"):

                cov = self.params["corr_xi_Yx"]*self.params["sigma_lnq"]*self.params["sigma_lnWLMegacam"]

            elif (observable1 == "Yx" and observable2 == "WLMegacam") or (observable2 == "Yx" and observable1 == "WLMegacam"):

                cov = self.params["corr_Yx_WL"]*self.params["sigma_lnYx"]*self.params["sigma_lnWLMegacam"]


            elif observable1 == "WLHST" and observable2 == "WLHST":

                cov = self.params["sigma_lnWLHST"]**2

            elif (observable1 == "xi" and observable2 == "WLHST") or (observable2 == "xi" and observable1 == "WLHST"):

                cov = self.params["corr_xi_Yx"]*self.params["sigma_lnq"]*self.params["sigma_lnWLHST"]

            elif (observable1 == "Yx" and observable2 == "WLHST") or (observable2 == "Yx" and observable1 == "WLHST"):

                cov = self.params["corr_Yx_WL"]*self.params["sigma_lnYx"]*self.params["sigma_lnWLHST"]

            else:

                cov = 0.

        elif layer == 1:

            if observable1 == "q_mmf3" and observable2 == "q_mmf3":

                cov = 1.

            elif observable1 == "xi" and observable2 == "xi":

                cov = 1.

            elif observable1 == "Yx" and observable2 == "Yx":

                cov = self.catalog.catalogue["Yx_std"][patch1]**2 ### patch1 is the cluster index

            elif observable1 == "q_mmf3_mean" and observable2 == "q_mmf3_mean":

                cov = 1.

            elif observable1 == "m_lens" and observable2 == "m_lens":

                cov = self.params["sigma_mlens"]**2

            elif observable1 == "p_zc19" and observable2 == "p_zc19":

                cov = 1.

            elif observable1 == "WLMegacam" and observable2 == "WLMegacam":

                cov = 1

            elif observable1 == "WLHST" and observable2 == "WLHST":

                cov = 1

            else:

                cov = 0.

        return cov
