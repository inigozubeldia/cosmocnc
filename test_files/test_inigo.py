import numpy as np
import pylab as pl
import cnc
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
z_mins = [1e-2,0.25]
z_maxs = [1.,2.]
obs_selects = ["q_mmf3_mean","xi"]
observabless = [[["q_mmf3_mean"]],[["xi","Yx","WLMegacam","WLHST"]]]
catalogues = ["Planck_MMF3_cosmo","SPT2500d"]
labels = [r"\textit{Planck}",r"SPT2500d"]
colors = ["tab:blue","tab:orange"]
linestyles = ["solid","dashed"]
pl.rc('text', usetex=True)
pl.rc('font', family='serif')
cm = 1./2.54
fig = pl.figure()
gs = fig.add_gridspec(2,1)#,hspace=0)
axs = gs.subplots()#
#pl.tight_layout()
for i in range(0,len(obs_selects)):
    print(catalogues[i])
    number_counts = cnc.cluster_number_counts()
    number_counts.cnc_params["cluster_catalogue"] = catalogues[i]
    number_counts.cnc_params["observables"] = observabless[i]
    number_counts.cnc_params["obs_select"] = obs_selects[i]
    number_counts.cnc_params["n_z"] = 200
    number_counts.cnc_params["obs_select_min"] = 5.
    number_counts.cnc_params["z_min"] = z_mins[i]
    number_counts.cnc_params["compute_abundance_matrix"] = True
    number_counts.cnc_params["z_max"] = z_maxs[i]
    number_counts.cnc_params["M_max"] = 1e16
    # number_counts.cnc_params["M_max"] = 5e15
    number_counts.cnc_params["n_points"] = 10000 ##number of points in which the mass function at each redshift (and all the convolutions) is evaluated
    number_counts.cnc_params["n_obs_select"] = 10000
    number_counts.cnc_params["cosmology_tool"] = "classy_sz"
    number_counts.cnc_params["scalrel_type_deriv"] = "numerical"
    number_counts.initialise()
    number_counts.get_number_counts()
    print("n tot",number_counts.n_tot)
    n_z = number_counts.n_z
    n_obs = number_counts.n_obs
    z = number_counts.redshift_vec
    q = number_counts.obs_select_vec#[0:n//2]
    n = len(q)
    q = q[0:n//2]
    n_obs = n_obs[0:n//2]
    color_lines = "k"
    aspect = 0.7
    #axs[0].semilogx(q,n_obs,label=r"\texttt{cncfast}")
    axs[0].semilogx(q,n_obs,label=labels[i],color=colors[i],linestyle=linestyles[i])
    axs[0].set_xlabel("$q_{\mathrm{obs}}$")
    axs[0].set_ylabel("$dN / dq_{\mathrm{obs}}$")
    axs[0].axvline(x=5.,color=color_lines)
    axs[0].axhline(y=0.,color=color_lines)
    axs[0].legend()
    axs[0].set_box_aspect(aspect)
    axs[0].xaxis.set_minor_formatter(mticker.ScalarFormatter())
    axs[0].xaxis.set_major_formatter(ScalarFormatter())
    axs[0].xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axs[1].plot(z,n_z,color=colors[i],linestyle=linestyles[i])
    axs[1].set_xlabel("$z$")
    axs[1].set_ylabel("$dN / dz$")
    axs[1].axvline(x=0.,color=color_lines)
    axs[1].axhline(y=0.,color=color_lines)
axs[1].set_box_aspect(aspect)
fig.tight_layout()
# pl.savefig("figures/abundance_1d_comparison.pdf")
pl.show()
