import numpy as np
import pylab as pl
import torch
from torch.distributions import Normal, Uniform, Distribution
import os
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import SNPE
from sbi.inference import NPE
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.utils.get_nn_models import (
    posterior_nn,
)  # For SNLE: likelihood_nn(). For SNRE: classifier_nn()


params_sbi_default = {
"data_train": "so_szcmblens_train",
"params_gaussian": None,
"params_uniform": None,
"params_info": None,
}

class sbi_cnc:

    def __init__(self,params_sbi=None):

        self.params_sbi = params_sbi

        params_gaussian = self.params_sbi["params_gaussian"]
        params_uniform = self.params_sbi["params_uniform"]
        params_info = self.params_sbi["params_info"]


        params_gaussian_info = [params_info[param] for param in params_gaussian]
        params_uniform_info = [params_info[param] for param in params_uniform]

        prior = mixed_distribution(params_gaussian_info,params_uniform_info)
        prior, num_parameters, prior_returns_numpy = process_prior(prior)
        self.prior = prior

    def load_data(self):

        self.data_train = data_sbi(params_sbi=self.params_sbi)

    def train(self,save=False,n_sim=-1):


        # Create inference object. Here, NPE is used

        density_estimator_build_fun = posterior_nn(model="maf",hidden_features=50,num_transforms=5)
        #density_estimator_build_fun = posterior_nn(model="maf")

        #self.inference = SNPE(prior=self.prior,density_estimator=density_estimator_build_fun)
        self.inference = NPE(prior=self.prior,density_estimator=density_estimator_build_fun)

        self.posterior = None

        data = self.data_train.data[0:n_sim,:]
        params = self.data_train.params[0:n_sim,:]

        if n_sim < 0:

            n_sim = self.data_train.n_sim

        self.n_sim = n_sim

        data = torch.tensor(data)
        params = torch.tensor(params)

        #Train the density estimator and build the posterior

        inference = self.inference.append_simulations(params,data)
        #density_estimator = self.inference.train(learning_rate=5e-6)#stop_after_epochs=300,max_num_epochs=300)
        density_estimator = self.inference.train(stop_after_epochs=1000,learning_rate=5e-5)#,max_num_epochs=2000)

        self.posterior = self.inference.build_posterior(density_estimator)

        if save == True:

            np.save("posterior_" + self.params_sbi["data_train"] + ".npy",self.posterior)

    def load(self):

        self.posterior = np.load("posterior_" + self.params_sbi["data_train"] + ".npy",allow_pickle=True)[()]

    def sample_posterior(self,data_test=None,n_samples=100000):

        self.params_sbi["data_test"] = data_test

        data_test_for_sbi = data_test_sbi(self.params_sbi)

        data_test = data_test_for_sbi.data
        params_true = data_test_for_sbi.params

        data_test = torch.tensor(data_test)
        params_true = torch.tensor(params_true)

        self.posterior_samples = self.posterior.sample((n_samples,),x=data_test).numpy()


class mixed_distribution(Distribution):

    # normal_params: List of tuples (mean, std) for Gaussian distributions
    # uniform_params: List of tuples (low, high) for Uniform distributions

    def __init__(self,normal_params,uniform_params):

        self.normals = [Normal(mean, std) for mean, std in normal_params]
        self.uniforms = [Uniform(low, high) for low, high in uniform_params]
        self.num_normals = len(self.normals)
        self.num_uniforms = len(self.uniforms)
        self.total_dims = self.num_normals + self.num_uniforms

       # Set batch and event shapes
        batch_shape = torch.Size()
        event_shape = torch.Size([self.total_dims])
        super().__init__(batch_shape, event_shape)

    def sample(self, sample_shape=torch.Size()):
        # Sample from each normal and uniform distribution
        normal_samples = [dist.sample(sample_shape).unsqueeze(-1) for dist in self.normals]
        uniform_samples = [dist.sample(sample_shape).unsqueeze(-1) for dist in self.uniforms]

        # Concatenate samples along the last dimension
        samples = torch.cat(normal_samples + uniform_samples, dim=-1)
        return samples

    def log_prob(self, value):

        normal_values = value[..., :self.num_normals]
        uniform_values = value[..., self.num_normals:]

        log_probs = []
        for i, dist in enumerate(self.normals):
            log_probs.append(dist.log_prob(normal_values[..., i]))
        for i, dist in enumerate(self.uniforms):
            log_probs.append(dist.log_prob(uniform_values[..., i]))

        # Sum log probabilities for the full distribution
        return torch.sum(torch.stack(log_probs), dim=0)


class data_sbi:

    def __init__(self,params_sbi=params_sbi_default):

        if params_sbi["data_train"] == "so_szcmblens_train":

            params_gaussian = ["n_s","Ob0h2","sigma_lnq_szifi","bias_cmblens","sigma_lnp"]
            params_uniform = ["h","sigma_8","Oc0h2","A_szifi","alpha_szifi"]
            params_total = params_gaussian + params_uniform

            num_parameters = len(params_total)

            n_sim = 0
            sim_ids = []

            n_sim_original = 5000

            for i in range(0,n_sim_original):

                if os.path.isfile("/rds-d4/user/iz221/hpc-work/catalogues_so_sbi/scores/catalogue_scores_train_paper_" + str(i) + ".npy") == True:
                #if os.path.isfile("/rds-d4/user/iz221/hpc-work/catalogues_so_sbi/scores/catalogue_so_simulated_sbi_paper_" + str(i) + ".npy") == True:

                    n_sim = n_sim + 1

                    sim_ids.append(i)

            print("N sim total",n_sim)

            # data = np.zeros((n_sim,num_parameters))
            # params = np.zeros((n_sim,num_parameters))
            #
            # for i in range(0,len(sim_ids)):
            #
            #     prename = "/rds-d4/user/iz221/hpc-work/catalogues_so_sbi/scores/catalogue_scores_train_paper_"
            #     #prename = "/rds-d4/user/iz221/hpc-work/catalogues_so_sbi/scores/catalogue_so_simulated_sbi_paper_"
            #     scores = np.load(prename + str(sim_ids[i]) + ".npy",allow_pickle=True)[()]
            #
            #     params_sim = np.load("/rds-d4/user/iz221/hpc-work/catalogues_so_sbi/catalogue_params_sbi_paper_" + str(sim_ids[i]) + ".npy",allow_pickle=True)[()]
            #
            #     score_sorted = []
            #     params_sorted = []
            #
            #     for param in params_total:
            #
            #         score_sorted.append(scores[param])
            #         params_sorted.append(params_sim[param])
            #
            #     data[i,:] = score_sorted
            #     params[i,:] = params_sorted

            #n_sim = 5000#10#0

            # data = data[0:n_sim,:]
            # params = params[0:n_sim,:]
            #
            # np.save("data_sbi.npy",(data,params))

            (data,params) = np.load("data_sbi.npy",allow_pickle=True)[()]

            indices = np.random.permutation(np.arange(n_sim))

            #params = params[indices,:]

            self.data = data.astype(np.float32)
            self.params = params.astype(np.float32)

            self.params_total = params_total
            self.params_gaussian = params_gaussian
            self.params_uniform = params_uniform

            n_sim = data.shape[0]

            self.n_sim = n_sim

            print("")
            print("")
            print("Data loaded")

class data_test_sbi:

    def __init__(self,params_sbi=params_sbi_default):

        if params_sbi["data_test"][0:17] == "so_szcmblens_test":

            params_gaussian = ["n_s","Ob0h2","sigma_lnq_szifi","bias_cmblens","sigma_lnp"]
            params_uniform = ["h","sigma_8","Oc0h2","A_szifi","alpha_szifi"]
            params_total = params_gaussian + params_uniform

            i_test = params_sbi["data_test"][18:]

            prename = "/rds-d4/user/iz221/hpc-work/catalogues_so_sbi/scores/catalogue_scores_test_"
            scores_test = np.load(prename + i_test + ".npy",allow_pickle=True)[()]
            params_test = np.load("/rds-d4/user/iz221/hpc-work/catalogues_so_sbi/catalogue_params_sbi_test_" + i_test + ".npy",allow_pickle=True)[()]

            data_test = []
            params_true = []

            for param in params_total:

                data_test.append(scores_test[param])
                params_true.append(params_test[param])

            data_test = np.array(data_test)
            params_true = np.array(params_true)

            self.data = data_test
            self.params = params_true
