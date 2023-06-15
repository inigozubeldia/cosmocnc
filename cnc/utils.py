import numpy as np
import multiprocess as mp
import scipy.signal as signal
import scipy.stats as stats
import functools
import math

def convolve_1d(x,dn_dx,sigma_scatter): #alternative convolution function, slower

    if sigma_scatter > 0.:

        kernel = gaussian_1d(x-np.mean(x),sigma_scatter)
        dn_dx = signal.convolve(dn_dx,kernel,mode="same",method="fft")/np.sum(kernel)

    return dn_dx

def convolve_1dk(x,dn_dx,sigma_scatter):

    if sigma_scatter > 0.:

        kernel = gaussian_1d(x-np.mean(x),sigma_scatter)

        padding_fraction = 0.5
        array = dn_dx

        pad_size = int(len(kernel)*padding_fraction)

        padded_array = np.pad(array,pad_size,mode='constant')
        kernel = np.pad(kernel,pad_size,mode='constant')

        array_fft = np.fft.rfft(padded_array)
        kernel_fft = np.fft.rfft(kernel)

        dn_dx = np.fft.fftshift(np.fft.irfft(array_fft*kernel_fft))[pad_size:pad_size+len(array)]/np.sum(kernel)

    return dn_dx

def convolve_nd(distribution,kernel):

#    convolved = np.fft.fftshift(np.fft.ifftn(np.fft.fftn(distribution)* np.fft.fftn(kernel))).real/np.sum(kernel) #FASTER BUT DOES WEIRD THINGS

    convolved = signal.convolve(distribution,kernel,mode="same",method="fft")/np.sum(kernel)

    return convolved

def eval_gaussian_nd(x_mesh,cov=None):

    shape = x_mesh.shape
    x_mesh = x_mesh.reshape(*x_mesh.shape[:-2],-1)
    pdf = stats.multivariate_normal.pdf(np.transpose(x_mesh),cov=cov)
    pdf = pdf.reshape(shape[1:])

    return pdf

def get_mesh(x):

    if x.shape[0] == 1:

        x_mesh = np.array(np.meshgrid(x[0,:]))

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

        diag = np.diag(cpdf)

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
