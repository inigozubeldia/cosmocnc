import numpy as np
import pylab as pl
import cnc
import time

data_lik_from_abundance = [False]

labels = ["Data lik from abundance","Data lik from scratch"]

for j in range(0,len(data_lik_from_abundance)):

    number_counts = cnc.cluster_number_counts()

    cnc_params = number_counts.cnc_params

    cnc_params["priors"] = False


    #cnc_params["data_lik_from_abundance"] = data_lik_from_abundance[j]
    #cnc_params["cluster_catalogue"] = "zc19_simulated_0"
    #number_counts.cnc_params = cnc_params

    number_counts.initialise()

    t0 = time.time()

    n = 10

    bias_vec = np.linspace(0.1,1.5,n)

    sigma_8_vec = np.linspace(0.79,0.82,n)
    Om0_vec = np.linspace(0.1,0.5,n)
    bias_vec = np.linspace(0.55,0.7,n)
    #bias_vec = [0.8]
    bias_cmb_lens_vec = np.linspace(0.1,1.5,n)
    sigma_cmblens_vec = np.linspace(0.01,0.3,n)
    corr_vec = np.linspace(-0.9,0.9,n)
    a_lens_vec = np.linspace(9,11.,n)
    lik_vec = np.zeros(n)
    sigma_sz = np.linspace(0.05,1.2,n)
    h_vec = np.linspace(0.6,0.75,n)

    a = h_vec

    for i in range(0,n):

        t0 = time.time()

        cosmo_params = number_counts.cosmo_params
        scal_rel_params = number_counts.scal_rel_params

    #    cosmo_params["sigma_8"] = a[i]
    #    cosmo_params["Om0"] = Om0_vec[i]


    #    scal_rel_params["bias_sz"] = bias_vec[i]
    #    scal_rel_params["sigma_lnp"] = sigma_cmblens_vec[i]
    #    scal_rel_params["bias_cmblens"] = bias_cmb_lens_vec[i]
    #    scal_rel_params["a_lens"] = a_lens_vec[i]
    #    scal_rel_params["corr_lnq_lnp"] = corr_vec[i]
    #    scal_rel_params["sigma_lnq"] = sigma_sz[i]

        cosmo_params["h"] = h_vec[i]

        number_counts.update_params(cosmo_params,scal_rel_params)

        lik_vec[i] = number_counts.get_log_lik()

        print(i,lik_vec[i])

        print("time",time.time()-t0)

    print(lik_vec)

    lik_vec = np.exp(lik_vec-np.max(lik_vec))

    pl.plot(a,lik_vec,label=labels[j])

pl.legend()

pl.show()
