import numpy as np
import pylab as pl
from .config import *
from .sr import *
from .utils import *
from .params import *
import importlib.util

class cluster_catalogue:

    def __init__(self,catalogue_name=None,
                 precompute_cnc_quantities=True,
                 bins_obs_select_edges=np.linspace(0.01,1.01,11),
                 bins_z_edges=np.exp(np.linspace(np.log(6.),np.log(100),6)),
                 observables=None, # "q_mmf3_mean","p_zc19", etc
                 obs_select=None,
                 cnc_params=None,
                 scal_rel_params=None):

        self.logger = logging.getLogger(__name__)

        if scal_rel_params is None:

            scal_rel_params = scaling_relation_params_default

        self.catalogue_name = catalogue_name
        self.catalogue = {}
        self.catalogue_patch = {}
        self.precompute_cnc_quantities = precompute_cnc_quantities
        self.cnc_params = cnc_params
        self.scal_rel_params = scal_rel_params


        if isinstance(bins_obs_select_edges,str):

            bins_obs_select_edges = eval(bins_obs_select_edges)

        if isinstance(bins_z_edges,str):

            bins_z_edges = eval(bins_z_edges)

        if isinstance(observables,str):

            observables = eval(observables)

        self.bins_obs_select_edges = bins_obs_select_edges
        self.bins_z_edges = bins_z_edges
        self.observables = observables
        self.obs_select =  obs_select

        path_to_survey = self.cnc_params["survey_cat"]
        spec_cat = importlib.util.spec_from_file_location("cat_module",path_to_survey)
        self.cat_module = importlib.util.module_from_spec(spec_cat)
        spec_cat.loader.exec_module(self.cat_module)
        cluster_catalogue_survey = self.cat_module.cluster_catalogue_survey

        catalogue_survey = cluster_catalogue_survey(catalogue_name=self.catalogue_name,
        observables=self.observables,obs_select=self.obs_select,cnc_params=self.cnc_params)

        self.catalogue = catalogue_survey.catalogue
        self.catalogue_patch = catalogue_survey.catalogue_patch

        self.validate_per_cluster_arrays()

        if hasattr(catalogue_survey,'stacked_data_labels'):

            self.stacked_data_labels = catalogue_survey.stacked_data_labels

        if hasattr(catalogue_survey,'stacked_data'):

            self.stacked_data = catalogue_survey.stacked_data

        if hasattr(catalogue_survey,'pdf_false_detection'):

            self.pdf_false_detection = catalogue_survey.pdf_false_detection

        if hasattr(catalogue_survey,'M'):

            self.M = catalogue_survey.M


        if self.precompute_cnc_quantities == True:

            self.get_precompute_cnc_quantities()

        self.n_clusters = len(self.catalogue[self.obs_select])

    def bin_number_counts(self):

            #Compute binned number counts

            if self.cnc_params["binned_lik_type"] == "obs_select":

                self.number_counts = np.histogram(self.catalogue[self.obs_select],bins=self.bins_obs_select_edges)[0]

            elif self.cnc_params["binned_lik_type"] == "z":

                self.number_counts = np.histogram(self.catalogue["z"],bins=self.bins_z_edges)[0]

            elif self.cnc_params["binned_lik_type"] == "z_and_obs_select":

                self.number_counts = np.zeros((len(self.bins_z_edges)-1,len(self.bins_obs_select_edges)-1))

                for i in range(0,len(self.bins_z_edges)-1):

                    for j in range(0,len(self.bins_obs_select_edges)-1):

                        indices = np.where((self.catalogue[self.obs_select] > self.bins_obs_select_edges[j]) & (self.catalogue[self.obs_select] < self.bins_obs_select_edges[j+1])
                        & (self.catalogue["z"] > self.bins_z_edges[i]) & (self.catalogue["z"] < self.bins_z_edges[i+1]))[0]
                        self.number_counts[i,j] = len(indices)




    def validate_per_cluster_arrays(self):

        """Check that every per-cluster array the survey catalogue exposes has exactly one
        entry per cluster, in the same order.

        The observable columns, "z" and "z_std", and the per-observable patch indices in
        catalogue_patch are all consumed positionally: cluster i of one array is paired with
        cluster i of every other. A survey catalogue file that trims its cluster list after
        building one of these arrays would silently pair clusters with the wrong data (e.g.
        the wrong noise curve); that must fail loudly here instead. Ragged auxiliary entries
        are checked for length only. Patch indices must be finite and non-negative wherever
        the corresponding observable is present."""

        n = len(self.catalogue[self.obs_select])

        obs_names = [self.obs_select]

        for obs_set in self.observables:

            for obs in (obs_set if isinstance(obs_set,(list,tuple)) else [obs_set]):

                if obs not in obs_names:

                    obs_names.append(obs)

        for key in ["z","z_std"] + obs_names:

            if key in self.catalogue and len(self.catalogue[key]) != n:

                raise ValueError("catalogue['%s'] has %d entries but the catalogue has %d clusters; per-cluster arrays are consumed positionally and must be cut together" % (key,len(self.catalogue[key]),n))

        for key in obs_names:

            if key not in self.catalogue_patch:

                continue

            patch = np.asarray(self.catalogue_patch[key],dtype=float)

            if patch.ndim != 1 or len(patch) != n:

                raise ValueError("catalogue_patch['%s'] has shape %s but the catalogue has %d clusters; patch indices are consumed positionally and must be cut together with the observables" % (key,str(patch.shape),n))

            try:

                present = np.isfinite(np.asarray(self.catalogue[key],dtype=float))

            except (TypeError,ValueError):

                present = np.zeros(n,dtype=bool)

            if np.any(~np.isfinite(patch[present])) or np.any(patch[present] < 0.):

                raise ValueError("catalogue_patch['%s'] holds a non-finite or negative patch index for a cluster that has that observable" % key)

    def get_precompute_cnc_quantities(self):

        self.indices_no_z = np.argwhere(np.isnan(self.catalogue["z"]))[:,0]
        self.indices_with_z = np.argwhere(~np.isnan(self.catalogue["z"]))[:,0]

        if self.cnc_params["non_validated_clusters"] == True:

            self.n_val = len(np.where(self.catalogue["validated"] > 0.5)[0])

        self.n_tot = len(self.catalogue["z"])

        #Bin number counts

        self.bin_number_counts()

        #Precompute other quantities

        self.obs_select_max = np.max(self.catalogue[self.obs_select])

        self.observable_dict = {}
        self.indices_obs_select = []
        self.indices_other_obs = []

        for i in self.indices_with_z:

            observables_cluster = []

            for observable_set in self.observables:

                observable_set_cluster = []

                for observable in observable_set:

                    if np.any(np.isnan(self.catalogue[observable][i])) == False:
                    #if np.isnan(self.catalogue[observable][i]) == False:

                        observable_set_cluster.append(observable)

                if len(observable_set_cluster) > 0:

                    observables_cluster.append(observable_set_cluster)

            self.observable_dict[i] = observables_cluster

            if observables_cluster == [[self.obs_select]]:

                self.indices_obs_select.append(i)

            else:

                self.indices_other_obs.append(i)

        self.indices_obs_select = np.array(self.indices_obs_select)
        self.indices_other_obs = np.array(self.indices_other_obs)

        #For clusters with only selection observable, sort them by patch

        self.indices_unique = []
        self.indices_unique_dict = {}

        if len(self.indices_obs_select) > 0:

            self.indices_unique = np.unique(self.catalogue_patch[self.obs_select][self.indices_obs_select])

            for i in range(0,len(self.indices_unique)):

                patch_index = self.indices_unique[i]
                indices = np.where(self.catalogue_patch[self.obs_select][self.indices_obs_select] == patch_index)[0]

                self.indices_unique_dict[str(int(patch_index))] = indices
