import numpy as np
import pylab as pl
from .cnc import *
from .sr import *

class catalogue_generator:

    def __init__(self,number_counts=None,n_catalogues=1,seed=None
    ,get_sky_coords=False,sky_frac=None,get_theta=False,std_vec_dict=None,
    patches_from_coord=False):

        self.n_catalogues = n_catalogues
        self.get_sky_coords = get_sky_coords
        self.sky_frac = sky_frac
        self.number_counts = number_counts
        self.params_cnc = self.number_counts.cnc_params
        self.get_theta = get_theta
        self.std_vec_dict = std_vec_dict
        self.patches_from_coord = patches_from_coord

        if seed is not None:

            np.random.seed(seed=seed)

        self.number_counts.get_hmf()

        self.scaling_relations = self.number_counts.scaling_relations
        self.scatter = self.number_counts.scatter
        self.skyfracs = self.scaling_relations[self.params_cnc["obs_select"]].skyfracs

        if self.sky_frac is None:

            self.sky_frac = np.sum(self.skyfracs)

        print("Sky frac",self.sky_frac)

        self.hmf_matrix = self.number_counts.hmf_matrix*4.*np.pi*self.sky_frac
        self.ln_M = self.number_counts.ln_M
        self.redshift_vec = self.number_counts.redshift_vec

    def get_total_number_clusters(self):

        self.dndz = integrate.simpson(self.hmf_matrix, x=self.ln_M,axis=1)
        self.dndln_M = integrate.simpson(self.hmf_matrix, x=self.redshift_vec,axis=0)
        self.n_tot = integrate.simpson(self.dndz, x=self.redshift_vec)

        print("Total mean number of clusters",self.n_tot)

    def sample_total_number_clusters(self):

        self.n_tot_obs = np.random.poisson(lam=self.n_tot,size=self.n_catalogues)

    def get_sky_patches_multinomial(self):

        self.sky_patches = {}
        p = self.skyfracs/np.sum(self.skyfracs)

        for i in range(0,self.n_catalogues):

            patches = np.random.multinomial(self.n_tot_obs[i],p)
            self.sky_patches[i] = np.repeat(np.arange(len(patches)),patches)


    def get_sky_patches_from_coord(self,observables,n_clusters):

        self.sky_patches = {}

        lon,lat = sample_lonlat(int(np.round(n_clusters/np.sum(self.skyfracs)*1.2)))

        patches = self.scaling_relations[self.params_cnc["obs_select"]].get_patch(lon,lat).astype(int)
        indices_select = np.where(patches > -0.5)[0][0:n_clusters]
        
        self.sky_patches[self.params_cnc["obs_select"]] = patches[indices_select]
        lon = lon[indices_select]
        lat = lat[indices_select]

        for observable in observables[1:]:

            self.sky_patches[observable] = self.scaling_relations[observable].get_patch(lon,lat).astype(int)

        return lon,lat

    def generate_catalogues_hmf(self):

        self.get_total_number_clusters()
        self.sample_total_number_clusters()

        self.catalogue_list = []

        for i in range(0,self.n_catalogues):

            n_clusters = int(self.n_tot_obs[i])

            z_samples,ln_M_samples = get_samples_pdf_2d(n_clusters,self.redshift_vec,self.ln_M,self.hmf_matrix)

            catalogue = {}
            catalogue["z"] = z_samples
            catalogue["M"] = np.exp(ln_M_samples)

            if self.get_sky_coords == True:

                lon,lat = sample_lonlat(n_clusters)
                catalogue["lon"] = lon
                catalogue["lat"] = lat

            if self.get_theta == True:

                catalogue["theta_so"] = self.get_theta_so(catalogue["M"],catalogue["z"])


            self.catalogue_list.append(catalogue)

    def get_theta_so(self,M,z):

        bias = self.number_counts.scal_rel_params["bias_sz"]
        H0 = self.number_counts.cosmo_params["h"]*100.
        D_A = self.number_counts.D_A
        E_z = self.number_counts.E_z

        D_A = np.interp(M,self.number_counts.redshift_vec,D_A)
        E_z = np.interp(M,self.number_counts.redshift_vec,E_z)

        prefactor_M_500_to_theta = 6.997*(H0/70.)**(-2./3.)*(bias/3.)**(1./3.)*E_z**(-2./3.)*(500./D_A)
        theta_so = prefactor_M_500_to_theta*M**(1./3.) #in arcmin

        return theta_so

    def generate_catalogues(self):

        self.get_total_number_clusters()
        self.sample_total_number_clusters()

        if self.patches_from_coord == False:

            self.get_sky_patches_multinomial()

        self.catalogue_list = []

        for i in range(0,self.n_catalogues):

            catalogue = {}
            n_clusters = int(self.n_tot_obs[i])

            if self.get_sky_coords == True and self.patches_from_coord == False:

                lon,lat = sample_lonlat(n_clusters)

            if self.patches_from_coord == True:

                lon,lat = self.get_sky_patches_from_coord(self.params_cnc["observables"][0],n_clusters)

            z_samples,ln_M_samples = get_samples_pdf_2d(n_clusters,self.redshift_vec,self.ln_M,self.hmf_matrix)

            n_observables = len(self.params_cnc["observables"][0])

            D_A = np.interp(z_samples,self.redshift_vec,self.number_counts.D_A)
            E_z = np.interp(z_samples,self.redshift_vec,self.number_counts.E_z)
            D_l_CMB = np.interp(z_samples,self.redshift_vec,self.number_counts.D_l_CMB)
            rho_c = np.interp(z_samples,self.redshift_vec,self.number_counts.rho_c)

            other_params = {"D_A": D_A,
                            "E_z": E_z,
                            "H0": self.number_counts.cosmology.background_cosmology.H0.value,
                            "D_l_CMB":D_l_CMB,
                            "rho_c":rho_c,
                            "D_CMB":self.number_counts.cosmology.D_CMB,
                            "zc":z_samples,
                            "cosmology":self.number_counts.cosmology}

            n_layers = self.scaling_relations[self.params_cnc["obs_select"]].get_n_layers()

            observable_patches = {}
            x0 = {}

            for observable in self.params_cnc["observables"][0]:

                x0[observable] = ln_M_samples

                if self.patches_from_coord == False:
               
                    observable_patches[observable] = np.zeros(n_clusters,dtype=np.int8)
                    observable_patches[self.params_cnc["obs_select"]] = self.sky_patches[i]

                elif self.patches_from_coord == True:

                    observable_patches[observable] = self.sky_patches[observable]

            for i in range(0,n_layers):

                x1 = {}

                for j in range(0,n_observables):

                    observable = self.params_cnc["observables"][0][j]
                    scal_rel = self.scaling_relations[observable]

                    vec = self.params_cnc["observable_vectorised"]

                    #if self.params_cnc["observable_vectorised"][observable] is True or self.params_cnc["observable_vectorised"] is True:

                    if (isinstance(vec, dict) and vec.get(observable, True)) or (isinstance(vec, bool) and vec):

                        x1[observable] = scal_rel.eval_scaling_relation_no_precompute(x0[observable],
                        layer=i,patch_index=observable_patches[observable],
                        params=self.number_counts.scal_rel_params,
                        other_params=other_params)

                    #elif self.params_cnc["observable_vectorised"][observable] is False:
                    elif (isinstance(vec, dict) and not vec.get(observable, True)) or (isinstance(vec, bool) and not vec):

                        x1[observable] = []

                        for k in range(0,n_clusters):

                            other_params_cluster = {}

                            for key in other_params.keys():
                                
                                if isinstance(other_params[key],float) or key == "cosmology":

                                    other_params_cluster[key] = other_params[key]                                

                                else:  

                                    other_params_cluster[key] = other_params[key][k]

                            a = scal_rel.eval_scaling_relation_no_precompute(np.array([x0[observable][k]]),
                            layer=i,patch_index=observable_patches[observable][k],
                            params=self.number_counts.scal_rel_params,
                            other_params=other_params_cluster)[0]
                            x1[observable].append(a)                        

                #Same covariance for all clusters 

                if self.params_cnc["cov_constant"][str(i)] is True:

                    covariance = covariance_matrix(self.scatter,self.params_cnc["observables"][0],
                    observable_patches=observable_patches,layer=np.arange(n_layers),other_params=other_params)
                    cov = covariance.cov[i]

                    noise = np.transpose(np.random.multivariate_normal(np.zeros(n_observables),cov,size=n_clusters))

                    for ll in range(0,len(self.params_cnc["observables"][0])):

                        x1[self.params_cnc["observables"][0][ll]] = x1[self.params_cnc["observables"][0][ll]] + noise[i,:]

                #Different covariance

                elif self.params_cnc["cov_constant"][str(i)] is False:

                    for k in range(0,n_clusters):

                        observable_patches_cluster = {}
                        other_params_cluster = {}

                        for key in observable_patches.keys():

                            observable_patches_cluster[key] = observable_patches[key][k]

                        for key in other_params.keys():

                            if isinstance(other_params[key],float) or key == "cosmology":

                                other_params_cluster[key] = other_params[key]

                            else:  

                                other_params_cluster[key] = other_params[key][k]

                        if i < n_layers-1:

                            covariance = covariance_matrix(self.scatter,self.params_cnc["observables"][0],
                            observable_patches=observable_patches_cluster,layer=np.arange(n_layers),other_params=other_params_cluster)
                            cov = covariance.cov[i]

                            noise = np.transpose(np.random.multivariate_normal(np.zeros(n_observables),cov,size=1))

                            kk = 0 

                            for observable in self.params_cnc["observables"][0]:

                                x1[observable][k] = x1[observable][k] + noise[kk,0]  
                                kk = kk + 1  

                        elif i == n_layers-1:

                            if  any(self.params_cnc["observable_vector"].values()) is True:

                                for observable in self.params_cnc["observables"][0]:

                                    if self.params_cnc["observable_vector"][observable] is False:

                                        covariance = covariance_matrix(self.scatter,[observable],
                                        observable_patches=observable_patches_cluster,layer=np.arange(n_layers),other_params=other_params_cluster)
                                        cov = covariance.cov[i]

                                        noise = np.transpose(np.random.multivariate_normal([0.],cov,size=1))
                                        x1[observable][k] = x1[observable][k] + noise[0][0]  

                                    elif self.params_cnc["observable_vector"][observable] is True:

                                        std = self.std_vec_dict[observable]
                                        noise = np.random.normal(loc=0.0,scale=1.0,size=len(std))*std
                                        x1[observable][k] = x1[observable][k] + noise

                x0 = x1

            print(x1[self.params_cnc["obs_select"]].shape)

            indices_select = np.where(np.array(x1[self.params_cnc["obs_select"]]) > self.params_cnc["obs_select_min"])[0]
            
            z_samples_select = z_samples[indices_select]
            ln_M_samples_select = ln_M_samples[indices_select]

            catalogue["z"] = z_samples_select
            catalogue["M"] = np.exp(ln_M_samples_select)

            if self.get_sky_coords == True:

                catalogue["lon"] = lon[indices_select]
                catalogue["lat"] = lat[indices_select]

            for k in range(0,len(self.params_cnc["observables"][0])):

                observable = self.params_cnc["observables"][0][k]

                if isinstance(vec, bool) and vec:

                    print(k,observable,"hereeee")

                    catalogue[self.params_cnc["observables"][0][k]] = x1[observable][indices_select]

                else:

                    catalogue[self.params_cnc["observables"][0][k]] = []

                    for kk in range(0,len(catalogue["M"])):

                        catalogue[self.params_cnc["observables"][0][k]].append(x1[observable][indices_select[kk]])
                                        

                catalogue[observable + "_patch"] = observable_patches[observable][indices_select]



            self.catalogue_list.append(catalogue)


def get_samples_pdf(n_samples,x,cpdf):

    cpdf = cpdf/np.max(cpdf)
    cpdf_samples = np.random.rand(n_samples)
    x_samples = np.interp(cpdf_samples,cpdf,x)

    return x_samples



def get_samples_pdf_2d(n_samples,x,y,pdf):

    cpdf_xgy = np.cumsum(pdf,axis=0)*(x[1]-x[0])

    for i in range(0,cpdf_xgy.shape[1]):

        cpdf_xgy[:,i] = cpdf_xgy[:,i]/np.max(cpdf_xgy[:,i])

    cpdf_y = np.cumsum(np.sum(pdf,axis=0))*(y[1]-y[0])*(x[1]-x[0])

    y_samples = get_samples_pdf(n_samples,y,cpdf_y)

    x_matrix = np.zeros(cpdf_xgy.shape)
    z = np.linspace(0.,1.,len(x))

    for i in range(0,len(y)):

        x_matrix[:,i] = np.interp(z,cpdf_xgy[:,i],x)

    cpdf_samples = np.random.rand(n_samples)

    interpolator = interpolate.RegularGridInterpolator((z,y),x_matrix,method='linear',bounds_error=True)
    x_samples = interpolator((cpdf_samples,y_samples))

    return (x_samples,y_samples)

#Longitude and latitude in radian

def sample_lonlat(n_clusters):

    lon = 2.*np.pi*np.random.rand(n_clusters)
    lat = np.arccos(2.*np.random.rand(n_clusters)-1.)

    return lon,lat

