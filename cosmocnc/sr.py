import numpy as np
import pylab as pl
from .hmf import *
from .params import *
from .config import *
from .utils import *

class covariance_matrix:

    def __init__(self,scatter,observables,observable_patches,layer=[0,1],other_params=None):#

        self.layer = layer
        self.cov = []
        self.inv_cov = []

        for k in range(0,len(self.layer)):

            cov_matrix = np.zeros((len(observables),len(observables)))

            for i in range(0,len(observables)):

                for j in range(0,len(observables)):

                    cov_matrix[i,j] = scatter.get_cov(observable1=observables[i],
                    observable2=observables[j],patch1=observable_patches[observables[i]],
                    patch2=observable_patches[observables[j]],layer=self.layer[k],other_params=other_params)

            self.cov.append(cov_matrix)
