import numpy as np
import cosmocnc

class cluster_catalogue_survey:

    def __init__(self,catalogue_name=None,observables=None,obs_select=None,cnc_params=None):

        self.catalogue_name = catalogue_name
        self.observables = observables
        self.obs_select =  obs_select
        self.cnc_params = cnc_params
        root_path = cosmocnc.root_path

        if self.catalogue_name[0:7] == "SO_sim_":

            catalogue = np.load(root_path + "data/catalogues_sim/catalogue_so_simulated_" + str(self.catalogue_name[7:]) + ".npy",allow_pickle=True)[0]

            self.catalogue = {}
            self.catalogue["q_so_sim"] = catalogue["q_so_sim"]
            self.catalogue["z"] = catalogue["z"]
            self.catalogue["z_std"] = np.zeros(len(self.catalogue["z"]))*1e-2
            self.catalogue["p_so_sim"] = catalogue["p_so_sim"]

            self.catalogue_patch = {}
            self.catalogue_patch["q_so_sim"] = catalogue["q_so_sim_patch"]
            self.catalogue_patch["p_so_sim"] = catalogue["p_so_sim_patch"]

            self.M = catalogue["M"]

            #Stacked CMB lensing

            self.stacked_data_labels = ["p_so_sim_stacked"]

            self.catalogue_patch["p_so_sim_stacked"] = np.zeros(len(self.catalogue["p_so_sim"])) #if one wants to use p with just one layer
            self.stacked_data = {"p_so_sim_stacked":{}}

            self.stacked_data["p_so_sim_stacked"]["data_vec"] = np.mean(self.catalogue["p_so_sim"])
            self.stacked_data["p_so_sim_stacked"]["inv_cov"] = float(len(self.catalogue["p_so_sim"]))
            self.stacked_data["p_so_sim_stacked"]["cluster_index"] = np.arange(len(self.catalogue["z"]))
            self.stacked_data["p_so_sim_stacked"]["observable"] = "p_so_sim"

            if "non_val" in self.cnc_params["catalogue_params"]:

                self.catalogue["validated"] = np.ones(len(self.catalogue["p_so_sim"]))

                if self.cnc_params["catalogue_params"]["non_val"] == True:

                    #Add some non-validated true detections

                    N_td_nonval = self.cnc_params["catalogue_params"]["N_td_nonval"]
                    N_fd = self.cnc_params["catalogue_params"]["N_fd"]

                    np.random.seed(seed=1)

                    indices = np.arange(len(self.catalogue["q_so_sim"]))
                    indices_nonval = np.random.choice(indices,N_td_nonval,replace=False)

                    self.catalogue["validated"][indices_nonval] = np.zeros(N_td_nonval)
                    self.catalogue["z"][indices_nonval] = np.array([float('nan')]*N_td_nonval)

                    f_v = (len(self.catalogue["z"])-N_td_nonval)/len(self.catalogue["z"])

                    #Add false detections

                    q_vec = np.linspace(5.,10.,self.cnc_params["n_points"])
                    pdf_fd = np.exp(-(q_vec-3.)**2/1.5**2)
                    pdf_fd = pdf_fd/integrate.simps(pdf_fd,q_vec)
                    self.pdf_false_detection = [q_vec,pdf_fd]

                    q_fd = rejection_sample_1d(q_vec,pdf_fd,N_fd)

                    self.catalogue["q_so_sim"] = np.concatenate((self.catalogue["q_so_sim"],q_fd))
                    self.catalogue_patch["q_so_sim"] = np.concatenate((self.catalogue_patch["q_so_sim"],np.zeros(len(q_fd))))
                    self.catalogue["z"] = np.concatenate((self.catalogue["z"],np.array([float('nan')]*N_fd)))
                    self.catalogue["p_so_sim"] = np.concatenate((self.catalogue["p_so_sim"],np.array([float('nan')]*N_fd)))
                    self.catalogue_patch["p_so_sim"] = np.concatenate((self.catalogue_patch["p_so_sim"],np.array([float('nan')]*N_fd)))

                    self.catalogue["validated"] = np.concatenate((self.catalogue["validated"],np.zeros(N_fd)))

                    if self.cnc_params["catalogue_params"]["none_validated"] == True:

                        self.catalogue["validated"] = np.zeros(len(self.catalogue["validated"]))
                        self.catalogue["z"] = np.array([float('nan')]*len(self.catalogue["z"]))
                        f_v = 0.

                    self.cnc_params["f_true_validated"] = f_v