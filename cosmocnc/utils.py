import numpy as np
import multiprocess as mp
import scipy.signal as signal
import scipy.stats as stats
import scipy.ndimage as ndimage
import functools
import math
import time
import sys
import pylab as pl
from scipy import ndimage



import logging

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        format='%(levelname)s - %(message)s',
        level=level
    )

def set_verbosity(verbosity):
    levels = {
        'none': logging.CRITICAL,
        'minimal': logging.INFO,
        'extensive': logging.DEBUG
    }

    level = levels.get(verbosity, logging.INFO)
    configure_logging(level)

def convolve_1d(x,dn_dx,sigma=None,type="fft",kernel=None,sigma_min=0):

    if sigma > sigma_min:

        if kernel is None:

            kernel = gaussian_1d(x-np.mean(x)+(x[1]-x[0])*0.5,sigma)

        dn_dx = signal.convolve(dn_dx,kernel,mode="same",method=type)/np.sum(kernel)

    return dn_dx

def convolve_nd(distribution,kernel):

    convolved = signal.convolve(distribution,kernel,mode="same")/np.sum(kernel)

    #convolved = ndimage.convolve(distribution,kernel,mode="nearest")/np.sum(kernel)

    return convolved

def eval_gaussian_nd(x_mesh,cov=None):

    shape = x_mesh.shape

    if shape[0] > 1:

        x_mesh = x_mesh.reshape(*x_mesh.shape[:-2],-1)

        #Old code:

        #pdf = stats.multivariate_normal.pdf(np.transpose(x_mesh),cov=cov)
        #pdf = np.transpose(pdf.reshape(shape[1:]))

        #New code: slightly faster

        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_factor = 1.0 / np.sqrt((2 * np.pi)**shape[0] * det_cov)
        mahalanobis = np.sum(np.dot(inv_cov, x_mesh) * x_mesh, axis=0)
        pdf = norm_factor * np.exp(-0.5 * mahalanobis)
        pdf = np.transpose(pdf.reshape(shape[1:]))


    else:

        pdf = gaussian_1d(x_mesh,np.sqrt(cov))[0,:]

    return pdf

def get_mesh(x):

    if x.shape[0] == 1:

        x_mesh = x

    elif x.shape[0] == 2:

        x_mesh = np.array(np.meshgrid(x[0,:],x[1,:]))

    elif x.shape[0] == 3:

        x_mesh = np.array(np.meshgrid(x[0,:],x[1,:],x[2,:]))

    return x_mesh

def gaussian_1d(x,sigma):

    return np.exp(-x**2/(2.*sigma**2))/(np.sqrt(2.*np.pi)*sigma)



def apodise(x_map):

    window_1d = signal.windows.tukey(x_map.shape[0],alpha=0.1)
    window = [window_1d for i in range(0,len(x_map.shape))]
    window = functools.reduce(np.multiply, np.ix_(*window))

    return x_map*window

def extract_diagonal(tensor):

    if len(tensor.shape) == 2:

        diag = np.diag(tensor)

    elif len(tensor.shape) == 3:

        diag = np.zeros(tensor.shape[0])

        for i in range(0,tensor.shape[0]):

            diag[i] = tensor[i,i,i]

    return diag


def get_cash_statistic(n_obs_vec,n_mean_vec):

    C = eval_cash_statistic(n_obs_vec,n_mean_vec)

    C_mean = np.zeros(len(C))
    C_var = np.zeros(len(C))

    for i in range(0,len(n_mean_vec)):

        C_mean[i],C_var[i] = eval_cash_statistic_expected(n_mean_vec[i])

    indices_nonnan = ~np.isnan(C)

    C = C[indices_nonnan]
    C_var = C_var[indices_nonnan]
    C_mean = C_mean[indices_nonnan]

    C = np.sum(C)
    C_var = np.sum(C_var)
    C_mean = np.sum(C_mean)
    C_std = np.sqrt(C_var)

    return (C,C_mean,C_std)

def eval_cash_statistic(n_obs,n_mean):

    return 2.*(n_mean - n_obs + n_obs*np.log(n_obs/n_mean))

def eval_cash_statistic_expected(n_mean):

    #Mean

    if 0. <=  n_mean <= 0.5:

        C_mean = -0.25*n_mean**3+1.38*n_mean**2-2*n_mean*np.log(n_mean)

    elif 0.5 < n_mean <= 2.:

        C_mean = -0.00335*n_mean**5 + 0.04259*n_mean**4 - 0.27331*n_mean**3 + 1.381*n_mean**2 - 2.*n_mean*np.log(n_mean)

    elif 2. < n_mean <= 5.:

        C_mean = 1.019275 + 0.1345*n_mean**(0.461-0.9*np.log(n_mean))

    elif 5. < n_mean <= 10.:

        C_mean = 1.00624 + 0.604/n_mean**1.68

    elif n_mean > 10.:

        C_mean = 1. + 0.1649/n_mean + 0.226/n_mean**2

    #Variance

    if 0. <= n_mean <= 0.1:

        C_var = 0.

        for j in range(0,5):

            C_var = C_var + np.exp(-n_mean)*n_mean**j/math.factorial(j)*(n_mean-j+j*np.log(j/n_mean))**2

        C_var = 4.*C_var - C_mean**2

    elif 0.1 < n_mean <= 0.2:

        C_var = -262.*n_mean**4 + 195.*n_mean**3 - 51.24*n_mean**2 + 4.34*n_mean + 0.77005

    elif 0.2 < n_mean <= 0.3:

        C_var = 4.23*n_mean**2 - 2.8254*n_mean + 1.12522

    elif 0.3 < n_mean <= 0.5:

        C_var = -3.7*n_mean**3 + 7.328*n_mean**2 - 3.6926*n_mean + 1.20641

    elif 0.5 < n_mean <= 1.:

        C_var = 1.28*n_mean**4 - 5.191*n_mean**3 + 7.666*n_mean**2 - 3.5446*n_mean + 1.15431

    elif 1. < n_mean <= 2.:

        C_var = 0.1125*n_mean**4 - 0.641*n_mean**3 + 0.859*n_mean**2 + 1.0914*n_mean - 0.05748

    elif 2. < n_mean <= 3.:

        C_var = 0.089*n_mean**3 - 0.872*n_mean**2 + 2.8422*n_mean - 0.67539

    elif 3. < n_mean <= 5.:

        C_var = 2.12336 + 0.012202*n_mean**(5.717 - 2.6*np.log(n_mean))

    elif 5 < n_mean <= 10.:

        C_var = 2.05159 + 0.331*n_mean**(1.343-np.log(n_mean))

    elif n_mean > 10.:

        C_var = 12./n_mean**3 + 0.79/n_mean**2 + 0.6747/n_mean + 2.

    return (C_mean,C_var)

def launch_multiprocessing(function,n_cores):

    if n_cores > 1:

        processes = []
        out_q = mp.Queue()

        for rank in range(n_cores):

            p = mp.Process(target=function, args=(rank,out_q,))
            processes.append(p)

        return_dict = {}

        [x.start() for x in processes]

        for p in processes:

            return_dict.update(out_q.get())

        [x.join() for x in processes]

    elif n_cores == 1:

        return_dict = function(0,{})

    return return_dict

def rejection_sample_1d(x,pdf,n_samples):

    samples = np.zeros(n_samples)

    for i in range(0,n_samples):

        pdf_eval = 0.
        pdf_sample = 1.

        while pdf_sample > pdf_eval:

            x_sample = np.random.rand()*(x[-1]-x[0])+x[0]
            pdf_sample = np.random.rand()*(pdf[-1]-pdf[0])+pdf[0]
            pdf_eval = np.interp(x_sample,x,pdf)

        samples[i] = x_sample

    return samples

def tile_1d_array(a,n_dim_output):

    grid = np.meshgrid(*([a] * n_dim_output), indexing='ij')
    custom_array = grid[0]

    return custom_array

def tile_1d_array_different_dim(original_array,n_dim_output,n_additional_dim):

    m = n_dim_output
    l = n_additional_dim

    # Add the additional dimensions to the original array
# Create coordinate grids for the additional dimensions
# Create the shape for the result array
    result_shape = (l,) * m + (original_array.size,)

    # Create the result array by broadcasting the original array
    result_array = original_array.reshape((1,) * m + original_array.shape)

    # Repeat the result array along the new dimensions
    for i in range(m):
        result_array = np.repeat(result_array, l, axis=i)

    return result_array

def interpolate_nd(r_interp,r_in,f_in,method="linear"):

    if r_interp.shape[-1] == 2:

        x_in = r_in[:,0]
        y_in = r_in[:,0]
        x_interp = r_interp[:,0]
        y_interp = r_interp[:,1]

        dx = x_in[1]-x_in[0]
        dy = y_in[1]-y_in[0]

        cpdf = interp2d([x_in[0],y_in[0]],[x_in[-1],y_in[-1]],[dx,dy],f_in,k=1,p=[False,False],e=[0,0])(x_interp,y_interp)

    return cpdf

#@numba.njit
def interpolate_deep(x_interp,x,f):

    ret = np.empty(len(x_interp))

    for i in range(0,len(x_interp)):

        ret[i] = np.interp(x_interp[i],x,f[i,:])

    return ret

def sample_from_uniform(x_min,x_max,n=1):

    sample = np.random.uniform(low=x_min,high=x_max,size=n)

    if len(sample) == 1:

        sample = sample[0]

    return sample

def sample_from_gaussian(mu,sigma,n=1):

    sample = np.random.normal(loc=mu, scale=sigma,size=n)

    if len(sample) == 1:

        sample = sample[0]

    return sample
