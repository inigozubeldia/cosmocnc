import numpy as np
import pylab as pl
from .cnc import *
from .sr import *
from numpy import trapz
import numba
import time

class catalogue_generator:

    def __init__(self,number_counts=None,n_catalogues=1,seed=None
    ,get_sky_coords=False,sky_frac=None,get_theta=False):

        self.n_catalogues = n_catalogues
        self.get_sky_coords = get_sky_coords
        self.sky_frac = sky_frac
        self.number_counts = number_counts
        self.params_cnc = self.number_counts.cnc_params
        self.get_theta = get_theta

        if seed is not None:

            np.random.seed(seed=seed)

        self.number_counts.get_hmf()
        # print(self.number_counts.get_hmf())

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
        # print(self.hmf_matrix)
        # print(self.ln_M)
        # print(self.redshift_vec)
        
        self.dndz = integrate.simpson(self.hmf_matrix, x=self.ln_M,axis=1)
        # print(self.dndz)
        self.dndln_M = integrate.simpson(self.hmf_matrix, x=self.redshift_vec,axis=0)
        self.n_tot = integrate.simpson(self.dndz, x=self.redshift_vec)
        # print("Total number of clusters",self.n_tot)

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
        # print(self.get_total_number_clusters())

        self.catalogue_list = []

        for i in range(0,self.n_catalogues):

            n_clusters = int(self.n_tot_obs[i])

            z_samples,ln_M_samples = get_samples_pdf_2d(n_clusters,self.redshift_vec,self.ln_M,self.hmf_matrix)
            # print(n_clusters)

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
    # z = np.linspace(0.,1.,len(x))
    # eps = 1e-15
    # z = np.exp(np.linspace(np.log(eps), 0.0, len(x)))  # in (eps, 1]
    # --- replace the whole z-construction block with: ---
    eps = 1e-12
    logit = lambda u: np.log(u) - np.log1p(-u)       # stable log(u/(1-u))

    # we'll build a common target grid in "w-space" (logit-CDF), normalized to [0,1]
    wmin, wmax = logit(eps), logit(1.0 - eps)
    w = np.linspace(wmin, wmax, len(x))
    w_norm = (w - wmin) / (wmax - wmin)              # in [0,1]


    # for i in range(0,len(y)):

    #     x_matrix[:,i] = np.interp(z,cpdf_xgy[:,i],x)
    # put before the loop (once):
    # BEFORE (your code, just above the loop)
    xmin, xmax = np.min(x), np.max(x)
    x_unit = (x - xmin) / (xmax - xmin)
    x_unit = np.clip(x_unit, eps, 1.0 - eps)
    invlogit = lambda z: 1.0 / (1.0 + np.exp(-z))

    # ADD this line (precompute once, outside the loop)
    x_logit = logit(x_unit)


    # loop body replacement:
    # INSIDE the loop (replace body)
    for i in range(len(y)):
        u_col = np.clip(cpdf_xgy[:, i], eps, 1.0 - eps)
        w_col = logit(u_col)                      # CDF axis in logit
        x_matrix[:, i] = np.interp(w, w_col, x_logit)   # logit–logit

    # --- replace your cpdf_samples clip + interpolator bits with: ---
    cpdf_samples = np.random.rand(n_samples)
    u_s = np.clip(cpdf_samples, eps, 1.0 - eps)
    w_s = logit(u_s)

    # normalize samples to the [0,1] w-axis used by the interpolator
    w_s_norm = (w_s - wmin) / (wmax - wmin)

    # build interpolator on normalized w-axis
    # AFTER interpolation (sampling back)
    interpolator = interpolate.RegularGridInterpolator(
        (w_norm, y), x_matrix, method='linear',
        bounds_error=False, fill_value=x_logit[0]
    )
    x_logit_samp = interpolator((w_s_norm, y_samples))

    # NEW: clip to safe range before invlogit
    lo, hi = logit(eps), logit(1.0 - eps)
    x_logit_samp = np.clip(x_logit_samp, lo, hi)

    x_unit_samp = invlogit(x_logit_samp)
    x_samples   = xmin + x_unit_samp * (xmax - xmin)

    return (x_samples,y_samples)

# def get_samples_pdf_2d(n_samples, x, y, pdf):
#     # CDFs along x|y and along y
#     cpdf_xgy = np.cumsum(pdf, axis=0) * (x[1] - x[0])
#     for i in range(cpdf_xgy.shape[1]):
#         m = np.max(cpdf_xgy[:, i])
#         cpdf_xgy[:, i] = cpdf_xgy[:, i] / (m if m > 0 else 1.0)

#     cpdf_y = np.cumsum(np.sum(pdf, axis=0)) * (y[1] - y[0]) * (x[1] - x[0])

#     # 1D y sampling (your existing routine)
#     y_samples = get_samples_pdf(n_samples, y, cpdf_y)

#     # --- sigmoid–sigmoid setup ---
#     eps = 1e-12           # keep away from exact 0/1
#     k_u = 20.0            # CDF-axis sigmoid steepness (tune)
#     k_x = 20.0            # x-axis sigmoid steepness (tune)

#     def sig01(v, k):
#         # expects v in [0,1]; returns (0,1)
#         return 1.0 / (1.0 + np.exp(-k * (v - 0.5)))

#     def invsig01(s, k):
#         # inverse of sig01; returns in [0,1]
#         return 0.5 + (np.log(s) - np.log1p(1.0 - s)) / k

#     # Common "w" grid for the CDF axis (sigmoid of uniform u)
#     u_grid = np.linspace(eps, 1.0 - eps, len(x))
#     w = sig01(u_grid, k_u)
#     w_norm = (w - w.min()) / (w.max() - w.min())

#     # Sigmoid-warp x to [0,1] then to (0,1)
#     xmin, xmax = np.min(x), np.max(x)
#     xrng = xmax - xmin if xmax > xmin else 1.0
#     x_unit = (x - xmin) / xrng
#     x_unit = np.clip(x_unit, eps, 1.0 - eps)
#     x_sig = sig01(x_unit, k_x)  # (0,1)

#     # Build x_matrix by interpolating in sigmoid(CDF) -> sigmoid(x)
#     x_matrix = np.zeros_like(cpdf_xgy)
#     for i in range(len(y)):
#         u_col = np.clip(cpdf_xgy[:, i], eps, 1.0 - eps)
#         w_col = sig01(u_col, k_u)  # (0,1)
#         x_matrix[:, i] = np.interp(w, w_col, x_sig)

#     # Sample: map uniform CDF samples through same sigmoid axis
#     cpdf_samples = np.random.rand(n_samples)
#     u_s = np.clip(cpdf_samples, eps, 1.0 - eps)
#     w_s = sig01(u_s, k_u)
#     w_s_norm = (w_s - w.min()) / (w.max() - w.min())

#     # Interpolate in (w_norm, y)
#     interpolator = interpolate.RegularGridInterpolator(
#         (w_norm, y), x_matrix, method='linear',
#         bounds_error=False, fill_value=x_sig[0]
#     )
#     x_sig_samp = interpolator((w_s_norm, y_samples))
#     x_sig_samp = np.clip(x_sig_samp, eps, 1.0 - eps)  # numeric safety

#     # Invert x sigmoid and de-normalize back to x-space
#     x_unit_samp = invsig01(x_sig_samp, k_x)
#     x_samples = xmin + x_unit_samp * xrng

#     return (x_samples, y_samples)




#Longitude and latitude in radian

def sample_lonlat(n_clusters):

    lon = 2.*np.pi*np.random.rand(n_clusters)
    lat = np.arccos(2.*np.random.rand(n_clusters)-1.)

    return lon,lat
