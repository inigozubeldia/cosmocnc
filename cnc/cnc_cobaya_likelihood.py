from cobaya.likelihood import Likelihood
from typing import Optional, Sequence
import numpy as np

class theta_mc_prior(Likelihood):
    # variables from yaml
    theta_mc_mean: float
    theta_mc_std: float

    def initialize(self):

        self.minus_half_invvar = - 0.5 / self.theta_mc_std ** 2

    def get_requirements(self):

        return {'theta_mc': None}

    def logp(self, **params_values):

        theta_mc_theory = self.provider.get_param("theta_mc")
        return self.minus_half_invvar * (theta_mc_theory - self.theta_mc_mean) ** 2

class cnc_likelihood(Likelihood):

    def initialize(self):

        super().initialize()

    def get_requirements(self):

        return {"sz_unbinned_cluster_counts": {}}

    def _get_theory(self, **params_values):

        theory = self.theory.get_sz_unbinned_cluster_counts()

        return theory

    def logp(self, **params_values):

        _derived = params_values.pop("_derived", None)
        theory = self._get_theory(**params_values)
        loglkl = theory

        return loglkl
