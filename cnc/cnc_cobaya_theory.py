from cobaya.theories.classy import classy
from copy import deepcopy
from typing import NamedTuple, Sequence, Union, Optional
from cobaya.tools import load_module
import logging
import os
import numpy as np

class cnc(classy):

    def initialize(self):
        """Importing cnc from the correct path, if given, and if not, globally."""
        self.classy_module = self.is_installed()
        from cnc import cluster_number_counts

        # global CosmoComputationError, CosmoSevereError
        self.cnc = cluster_number_counts()
        super(classy,self).initialize()
        self.extra_args["output"] = self.extra_args.get("output", "")

        self.cnc.initialise()
        self.derived_extra = []
        self.log.info("Initialized!")


    def must_provide(self, **requirements):
        if "sz_unbinned_cluster_counts" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("sz_unbinned_cluster_counts")
            # specify the method to collect the new observable
            self.collectors["sz_unbinned_cluster_counts"] = Collector(
                    method="log_lik_unbinned", # name of the method in classy.pyx
                    args_names=[],
                    args=[])

    # get the required new observable
    def get_sz_unbinned_cluster_counts(self):
        # thats the function passed to soliket
        cls = deepcopy(self._current_state["sz_unbinned_cluster_counts"])
        # print(cls)
        return cls


    # IMPORTANT: this method is imported from cobaya and modified to accomodate the emulators
    def calculate(self, state, want_derived=True, **params_values_dict):
        params_values = params_values_dict.copy()
        cosmo_params = {
        'Om0': params_values['omega_cdm']/(params_values['H0']/100.)**2+params_values['omega_b']/(params_values['H0']/100.)**2,
        'Ob0': params_values['omega_b']/(params_values['H0']/100.)**2,
        'h': params_values['H0']/100.,
        'A_s': np.exp(params_values['logA'])*1e-10,
        'n_s': params_values['n_s'],
        'm_nu': 0.0
        }
        scal_rel_params = self.cnc.scal_rel_params

        self.cnc.update_params(cosmo_params,scal_rel_params)

        self.cnc.get_number_counts()

        # Gather products
        for product, collector in self.collectors.items():
            method = getattr(self.cnc, collector.method)
            state[product] = method()

    def _get_derived_all(self, derived_requested=True):
        return [],[]

    def close(self):
        return 1
    #
    @classmethod
    def is_installed(cls, **kwargs):
        return load_module('cnc')


# this just need to be there as it's used to fill-in self.collectors in must_provide:
class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence] = None
    post: Optional[callable] = None
