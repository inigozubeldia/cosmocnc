import numpy as np
import pylab as pl
import cosmocnc as cnc
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
import time
pl.rc('text', usetex=True)
pl.rc('font', family='serif')
cm = 1./2.54
fig = pl.figure()
gs = fig.add_gridspec(2,1)#,hspace=0)
axs = gs.subplots()#
labels = ["SPT baseline"]
k = 0
binned_likelihood_types = ["obs_select","z"]
n_catalogues = 1
n_bins = [20,20]
for i in range(0,len(binned_likelihood_types)):
    binned_obs_all = np.zeros((n_catalogues,n_bins[i]-1))
    n_tot_obs_vec = np.zeros(n_catalogues)
    for j in range(0,n_catalogues):
        number_counts = cnc.cluster_number_counts()
        number_counts.cnc_params["compute_abundance_matrix"] = True
        number_counts.cnc_params["cluster_catalogue"] = "SPT2500d"
        number_counts.cnc_params["observables"] = [["xi"]]
        number_counts.cnc_params["obs_select"] = "xi"
        number_counts.cnc_params["data_lik_from_abundance"] = False
        number_counts.cnc_params["number_cores_hmf"] = 1
        number_counts.cnc_params["number_cores_abundance"] = 1
        number_counts.cnc_params["number_cores_data"] = 8
        number_counts.cnc_params["obs_select_min"] = 5.
        number_counts.cnc_params["obs_select_max"] = 200.
        number_counts.cnc_params["n_points"] = 1024*16 #64*4#2**13, ##number of points in which the mass function at each redshift (and all the convolutions) is evaluated
        number_counts.cnc_params["n_obs_select"] = number_counts.cnc_params["n_points"]
        number_counts.cnc_params["parallelise_type"] = "redshift"
        number_counts.cnc_params["path_to_cosmopower_organization"] = "/Users/boris/Work/CLASS-SZ/SO-SZ/cosmopower-organization/"
        #number_counts.cnc_params["cosmology_tool"] = "classy_sz"
        number_counts.cnc_params["scalrel_type_deriv"] = "numerical"
        #number_counts.cnc_params["scalrel_type_deriv"] = "numerical"
        number_counts.cnc_params["z_max"] = 3.
        number_counts.cnc_params["z_min"] = 0.01
        number_counts.cnc_params["n_z"] = 100
        number_counts.scal_rel_params["dof"] = 0.
        number_counts.cnc_params["M_min"] = 1e13
        number_counts.cnc_params["M_max"] = 1e16
        number_counts.cnc_params["apply_obs_cutoff"] = False
        number_counts.scal_rel_params["dof"] = 0.
        # number_counts.scal_rel_params["a_lens"] = 1.
        # number_counts.scal_rel_params["corr_lnq_lnp"] = 0.
        # number_counts.scal_rel_params["bias_sz"] = 0.8
        number_counts.scal_rel_params["q_cutoff"] = 0.
        number_counts.scal_rel_params['sigma_lnq'] = 0.01
        #number_counts.cosmo_params["k_cutoff"] = 0.67
        #number_counts.cosmo_params["ps_cutoff"] = 0.75
        number_counts.cnc_params["abundance_integral_type"] = "fft"
        number_counts.cnc_params["likelihood_type"] = "unbinned"
        # number_counts.cnc_params["binned_lik_type"] = binned_likelihood_types[i] #can be "obs_select", "z", or "z_and_obs_select"
        # number_counts.cnc_params["bins_edges_z"] = np.linspace(number_counts.cnc_params["z_min"],number_counts.cnc_params["z_max"],n_bins[1])
        # number_counts.cnc_params["bins_edges_obs_select"] = np.exp(np.linspace(np.log(number_counts.cnc_params["obs_select_min"]),np.log(number_counts.cnc_params["obs_select_max"]),n_bins[0]))
        number_counts.initialise()
        if j == 0:
            t0 = time.time()
            number_counts.get_log_lik()
            t1 = time.time()
            print("Time abundance",t1-t0)
            print("Time likelihood",t1-t0)
            # n_binned_theory = number_counts.n_binned
            # x = number_counts.bins_centres
            # n_tot_theory = np.sum(n_binned_theory) #number_counts.n_tot
            n_tot_obs = number_counts.catalogue.n_tot
            n_tot_theo = number_counts.n_tot
            print("Total n clusters",n_tot_obs,n_tot_theo)
        number_counts.catalogue.bin_number_counts()
        n_obs = number_counts.catalogue.number_counts
        binned_obs_all[j,:] = n_obs
        n_tot_obs = number_counts.catalogue.n_tot
        n_tot_obs_vec[j] = n_tot_obs
    #pl.tight_layout()
    n_tot_vec = np.sum(binned_obs_all,axis=1)
    n_tot_mean = np.mean(n_tot_vec)
    n_tot_std = np.std(n_tot_vec)/np.sqrt(float(n_catalogues))
    print("n total integrated mean ",np.mean(n_tot_obs_vec))
    print("")
    print("")
    print("n tot theory",n_tot_theory,np.sqrt(n_tot_theory))
    print("n tot mean",n_tot_mean,n_tot_std)
    print("n diff",n_tot_mean-n_tot_theory)
    color_lines = "k"
    aspect = 0.7
    n_binned_obs_mean = np.mean(binned_obs_all,axis=0)
    if i == 0:
        np.save("/Users/boris/Work/CLASS-SZ/SO-SZ/cosmocnc/tutorials/test_files/number_counts_so_snr.npy",[x,n_binned_theory,n_binned_obs_mean])
        #axs[0].semilogx(q,n_obs,label=labels[k])
        axs[0].errorbar(x,n_binned_theory,yerr=np.sqrt(n_binned_theory),label="Theoretical prediction",linestyle="none",fmt="")
        axs[0].semilogx()
        axs[0].errorbar(x,n_binned_obs_mean,color="tab:orange",yerr=np.sqrt(n_binned_theory/n_catalogues),label="Synthetic catalogue",linestyle="none",fmt="")
        axs[0].set_xlabel("$q_{\mathrm{obs}}$")
        axs[0].set_ylabel("$N$")
        axs[0].axvline(x=number_counts.cnc_params["obs_select_min"],color=color_lines)
        axs[0].axhline(y=0.,color=color_lines)
        axs[0].legend()
        axs[0].set_title("SO baseline")
        axs[0].set_box_aspect(aspect)
        axs[0].xaxis.set_minor_formatter(mticker.ScalarFormatter())
        axs[0].xaxis.set_major_formatter(ScalarFormatter())
        axs[0].xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    elif i == 1:
        np.save("/Users/boris/Work/CLASS-SZ/SO-SZ/cosmocnc/tutorials/test_files/number_counts_so_z.npy",[x,n_binned_theory,n_binned_obs_mean])
        axs[1].errorbar(x,n_binned_theory,yerr=np.sqrt(n_binned_theory),label=labels[k],linestyle="none",fmt="")
        axs[1].errorbar(x,n_binned_obs_mean,color="tab:orange",yerr=np.sqrt(n_binned_theory/n_catalogues),linestyle="none",fmt="")
        axs[1].set_xlabel("$z$")
        axs[1].set_ylabel("$N$")
        axs[1].axvline(x=0.,color=color_lines)
        axs[1].axhline(y=0.,color=color_lines)
        axs[1].axhline(y=1.,color=color_lines)
        #axs[1].set_yscale("log")
        axs[1].set_box_aspect(aspect)
fig.tight_layout()
pl.savefig("/Users/boris/Work/CLASS-SZ/SO-SZ/cosmocnc/tutorials/test_files/abundance_1d_SO_binned_mean.pdf",bbox_inches='tight')
pl.show()