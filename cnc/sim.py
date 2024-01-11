import numpy as np
import pylab as pl
from .cnc import *
from .sr import *
import numba
import time

class catalogue_generator:

    def __init__(self,n_catalogues=1,params_cnc=cnc_params_default,seed=None,
    scal_rel_params=scaling_relation_params_default,
    cosmo_params=cosmo_params_default,get_sky_coords=False,sky_frac=None):

        self.n_catalogues = n_catalogues
        self.params_cnc = params_cnc
        self.get_sky_coords = get_sky_coords
        self.sky_frac = sky_frac

        if seed is not None:

            np.random.seed(seed=seed)

        self.number_counts = cluster_number_counts(cnc_params=self.params_cnc)
        self.number_counts.scal_rel_params = scal_rel_params
        self.number_counts.cosmo_params = cosmo_params

        self.number_counts.initialise()
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

        self.dndz = integrate.simps(self.hmf_matrix,self.ln_M,axis=1)
        self.dndln_M = integrate.simps(self.hmf_matrix,self.redshift_vec,axis=0)

        self.n_tot = integrate.simps(self.dndz,self.redshift_vec)

    def sample_total_number_clusters(self):

        self.n_tot_obs = np.random.poisson(lam=self.n_tot,size=self.n_catalogues)

    def get_sky_patches(self):

        self.sky_patches = {}
        p = self.skyfracs/np.sum(self.skyfracs)

        for i in range(0,self.n_catalogues):

            patches = np.random.multinomial(self.n_tot_obs[i],p)
            self.sky_patches[i] = np.repeat(np.arange(len(patches)),patches)

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

            self.catalogue_list.append(catalogue)

    def generate_catalogues(self):

        self.get_total_number_clusters()
        self.sample_total_number_clusters()
        self.get_sky_patches()

        self.catalogue_list = []

        for i in range(0,self.n_catalogues):

            n_clusters = int(self.n_tot_obs[i])

            z_samples,ln_M_samples = get_samples_pdf_2d(n_clusters,self.redshift_vec,self.ln_M,self.hmf_matrix)

            n_observables = len(self.params_cnc["observables"][0])
            x0 = np.empty((n_observables,n_clusters))

            for j in range(0,n_observables):

                x0[j,:] = ln_M_samples

            D_A = np.interp(z_samples,self.redshift_vec,self.number_counts.D_A)
            E_z = np.interp(z_samples,self.redshift_vec,self.number_counts.E_z)
            D_l_CMB = np.interp(z_samples,self.redshift_vec,self.number_counts.D_l_CMB)
            rho_c = np.interp(z_samples,self.redshift_vec,self.number_counts.rho_c)

            other_params = {"D_A": D_A,"E_z": E_z,
            "H0": self.number_counts.cosmology.background_cosmology.H0.value,
            "D_l_CMB":D_l_CMB,"rho_c":rho_c,"D_CMB":self.number_counts.cosmology.D_CMB}

            n_layers = self.scaling_relations[self.params_cnc["obs_select"]].get_n_layers()

            patch_indices = self.sky_patches[i]

            observable_patches = {}

            for observable in self.params_cnc["observables"][0]:

                observable_patches[observable] = np.zeros(n_clusters,dtype=np.int8)

            observable_patches[self.params_cnc["obs_select"]] = patch_indices

            for i in range(0,n_layers):

                x1 = np.zeros((n_observables,n_clusters))

                for j in range(0,n_observables):

                    observable = self.params_cnc["observables"][0][j]

                    scal_rel = self.scaling_relations[observable]

                    x1[j,:] = scal_rel.eval_scaling_relation_no_precompute(x0[j,:],
                    layer=i,patch_index=observable_patches[observable],
                    params=self.number_counts.scal_rel_params,
                    other_params=other_params)

                covariance = covariance_matrix(self.scatter,self.params_cnc["observables"][0],
                observable_patches=observable_patches,layer=np.arange(n_layers))
                cov = covariance.cov[i]

                noise = np.transpose(np.random.multivariate_normal(np.zeros(n_observables),cov,size=n_clusters))
                x1 = x1 + noise
                x0 = x1

            catalogue = {}

            indices_select = np.where(x1[0,:] > self.params_cnc["obs_select_min"])[0]
            z_samples_select = z_samples[indices_select]
            ln_M_samples_select = ln_M_samples[indices_select]
            x1_select = x1[:,indices_select]

            catalogue["z"] = z_samples_select
            catalogue["M"] = np.exp(ln_M_samples_select)

            for k in range(0,len(self.params_cnc["observables"][0])):

                catalogue[self.params_cnc["observables"][0][k]] = x1_select[k,:]
                catalogue[self.params_cnc["observables"][0][k] + "_patch"] = observable_patches[observable][indices_select]

            if self.get_sky_coords == True:

                lon,lat = sample_lonlat(n_clusters)
                catalogue["lon"] = lon
                catalogue["lat"] = lat

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
