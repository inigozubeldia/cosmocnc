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

        self.cnc = cluster_number_counts()
        super(classy,self).initialize()
        self.extra_args["output"] = self.extra_args.get("output","")

        self.cnc.initialise()
        self.derived_extra = []
        self.log.info("Initialized")

    def must_provide(self, **requirements):

        if "sz_unbinned_cluster_counts" in requirements:

            # make sure cobaya still runs as it does for standard classy

            requirements.pop("sz_unbinned_cluster_counts")

            # specify the method to collect the new observable

            self.collectors["sz_unbinned_cluster_counts"] = Collector(
                    method="get_log_lik", # name of the method in classy.pyx
                    args_names=[],
                    args=[])

    # get the required new observable

    def get_sz_unbinned_cluster_counts(self):

        # thats the function passed to soliket

        cls = deepcopy(self._current_state["sz_unbinned_cluster_counts"])

        return cls

    def calculate(self, state, want_derived=True, **params_values_dict):

        params_values = params_values_dict.copy()

        cosmo_params = self.cnc.cosmo_params

        assign_parameter_value(cosmo_params,params_values,"Om0")
        assign_parameter_value(cosmo_params,params_values,"Ob0")
        assign_parameter_value(cosmo_params,params_values,"h")
        assign_parameter_value(cosmo_params,params_values,"sigma_8")
        assign_parameter_value(cosmo_params,params_values,"n_s")

        scal_rel_params = self.cnc.scal_rel_params

        assign_parameter_value(scal_rel_params,params_values,"alpha")
        assign_parameter_value(scal_rel_params,params_values,"log10_Y_star")
        assign_parameter_value(scal_rel_params,params_values,"bias_sz")
        assign_parameter_value(scal_rel_params,params_values,"bias_cmblens")
        assign_parameter_value(scal_rel_params,params_values,"sigma_lnq")
        assign_parameter_value(scal_rel_params,params_values,"sigma_lnp")
        assign_parameter_value(scal_rel_params,params_values,"corr_lnq_lnp")

        self.cnc.update_params(cosmo_params,scal_rel_params)

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

def assign_parameter_value(lik_dict,cobaya_dict,parameter):

    if parameter in cobaya_dict:

        lik_dict[parameter] = cobaya_dict[parameter]

# this just need to be there as it's used to fill-in self.collectors in must_provide:
class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence] = None
    post: Optional[callable] = None
