import yaml
import numpy as np

# Define paths for the output YAML files
output_path_forward = "/Users/boris/Work/CLASS-SZ/SO-SZ/cosmocnc/mcmcs/cobaya/input_files/cnc_unbinned_cluster_counts_spt_unbinned_evaluate_forward.yaml"
output_path_backward = "/Users/boris/Work/CLASS-SZ/SO-SZ/cosmocnc/mcmcs/cobaya/input_files/cnc_unbinned_cluster_counts_spt_unbinned_evaluate_backward.yaml"

# Common configuration template
config_template = {
    "output": "/Users/boris/Work/CLASS-SZ/SO-SZ/cosmocnc/mcmcs/cobaya/chains/cnc_unbinned_cluster_counts_spt_unbinned_evaluate",
    "likelihood": {
        "cosmocnc.cnc_cobaya_likelihood.cnc_likelihood": None
    },
    "theory": {
        "cosmocnc.cnc_cobaya_theory.cnc": {
            "cosmology_tool": "classy_sz",
            "number_cores_abundance": 10,
            "number_cores_data": 10,
            "number_cores_hmf": 10,
            "cluster_catalogue": 'SPT2500d',
            "obs_select": 'xi',
            "observables": [["xi"]],
            "z_min": 0.25,
            "z_max": 2.0,
            "bins_edges_z": np.linspace(0.25, 2.0, 21).tolist(),
            "z_errors": True,
            "z_error_min": 1e-5,
            "obs_select_min": 5.0,
            "obs_select_max": 47.0,
            "bins_edges_obs_select": np.exp(np.linspace(np.log(5.0), np.log(47.0), 6)).tolist(),
            "obs_select_min": 5.0,
            "parallelise_type": "patch",
            "SZmPivot": 3e14,
            "dof": 3.0,
            "q_cutoff": 2.0,
            "scalrel_type_deriv": "numerical",
            "n_points": 1024*64,
            "cosmology_tool": "classy_sz",
            "class_sz_ndim_redshifts": 500,
            "class_sz_ndim_masses": 100,
            "class_sz_concentration_parameter": "B13",
            "class_sz_hmf": "T08M500c",
            "class_sz_cosmo_model": "ede-v2",
            "hmf_calc": "classy_sz",
            "cosmocnc_verbose": "extensive" 
        }
    },
    "params": {
        "H0": {
            "prior": {"min": 40.0, "max": 100.0},
            "ref": {"dist": "norm", "loc": 67.4, "scale": 1.0},
            "proposal": 1.0,
            "latex": "H_0"
        },
        "tau_reio": {"value": 0.06},
        "sigma_8": {
            "prior": {"min": 0.1, "max": 2.0},
            "ref": {"dist": "norm", "loc": 0.81, "scale": 0.01},
            "proposal": 0.01,
            "latex": "\\sigma_8"
        },
        "n_s": {
            "prior": {"min": 0.8812, "max": 1.0492},
            "ref": {"dist": "norm", "loc": 0.96, "scale": 0.004},
            "proposal": 0.004,
            "latex": "n_\\mathrm{s}"
        },
        "Om0": {
            "prior": {"min": 0.08, "max": 0.5},
            "ref": {"dist": "norm", "loc": 0.31, "scale": 0.05},
            "proposal": 0.05,
            "latex": "\\Omega_\\mathrm{m}"
        },
        "Ob0": {
            "prior": {"min": 0.01, "max": 0.2},
            "ref": {"dist": "norm", "loc": 0.048, "scale": 0.005},
            "proposal": 0.005,
            "latex": "\\Omega_\\mathrm{b}"
        },
        "A_sz": {
            "prior": {"min": 1, "max": 10},
            "ref": {"dist": "norm", "loc": 2.0, "scale": 0.05},
            "proposal": 0.05,
            "latex": "A_{\\mathrm{SZ}}"
        },
        "B_sz": {
            "prior": {"min": 1.0, "max": 2.5},
            "ref": {"dist": "norm", "loc": 1.5, "scale": 0.05},
            "proposal": 0.05,
            "latex": "B_{\\mathrm{SZ}}"
        },
        "C_sz": {
            "prior": {"min": -1.0, "max": 2.0},
            "ref": {"dist": "norm", "loc": 0.5, "scale": 0.05},
            "proposal": 0.05,
            "latex": "C_{\\mathrm{SZ}}"
        },
        "A_x": {
            "prior": {"min": 3.0, "max": 10},
            "ref": {"dist": "norm", "loc": 5.0, "scale": 0.05},
            "proposal": 0.05,
            "latex": "A_{\\mathrm{X}}"
        },
        "B_x": {
            "prior": {"min": 0.3, "max": 0.9},
            "ref": {"dist": "norm", "loc": 0.6, "scale": 0.05},
            "proposal": 0.05,
            "latex": "B_{\\mathrm{X}}"
        },
        "C_x": {
            "prior": {"min": -1.0, "max": 0.5},
            "ref": {"dist": "norm", "loc": 0.25, "scale": 0.05},
            "proposal": 0.05,
            "latex": "C_{\\mathrm{X}}"
        },
        "sigma_lnq": {
            "prior": {"min": 0.05, "max": 0.3},
            "ref": {"dist": "norm", "loc": 0.179, "scale": 0.01},
            "proposal": 0.01,
            "latex": "\\sigma_{\\mathrm{SZ}}"
        },
        "sigma_lnYx": {
            "prior": {"min": 0.05, "max": 0.3},
            "ref": {"dist": "norm", "loc": 0.179, "scale": 0.01},
            "proposal": 0.01,
            "latex": "\\sigma_{\\mathrm{X}}"
        },
        "corr_xi_Yx": {
            "prior": {"min": -1, "max": 1},
            "ref": {"dist": "norm", "loc": 0.0, "scale": 0.01},
            "proposal": 0.01,
            "latex": "\\rho_{\\mathrm{SZ},\\mathrm{X}}"
        }
    },
    "prior": {
        "n_s_prior": "lambda n_s: stats.norm.logpdf(n_s, loc=0.96, scale=0.0042)",
        "sigma_lnq_prior": "lambda sigma_lnq: stats.norm.logpdf(sigma_lnq, loc=0.173, scale=0.023)",
        "Ob0h2_prior": "lambda Ob0,H0: stats.norm.logpdf(Ob0*H0**2/10000, loc=0.022245895, scale=0.00015)"
    },
    "sampler": {
        "evaluate": {
            "override": {
                "H0": 67.4416,
                "sigma_8": 0.813132,
                "n_s": 0.955518,
                "Om0": 0.318898,
                "Ob0": 0.0471481,
                "A_sz": 5.5,
                "B_sz": 1.75,
                "C_sz": 0.5,
                "A_x": 6.5,
                "B_x": 0.6,
                "C_x": -0.25,
                "sigma_lnq": 0.255,
                "sigma_lnYx": 0.255,
                "corr_xi_Yx": 0.1
            }
        }
    },
    "timing": True
}

# Adjustments for forward and backward configurations
forward_config = config_template.copy()
forward_config["theory"]["cosmocnc.cnc_cobaya_theory.cnc"]["data_lik_from_abundance"] = True

backward_config = config_template.copy()
backward_config["theory"]["cosmocnc.cnc_cobaya_theory.cnc"]["data_lik_from_abundance"] = False

# Write configurations to YAML files
with open(output_path_forward, 'w') as file:
    yaml.dump(forward_config, file, default_flow_style=False)

with open(output_path_backward, 'w') as file:
    yaml.dump(backward_config, file, default_flow_style=False)

print("YAML files have been created.")