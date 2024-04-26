import numpy as np
import pylab as pl
from cobaya.yaml import yaml_load_file
from cobaya.run import run
from mpi4py import MPI
from cobaya.log import LoggedError

rank = MPI.COMM_WORLD.Get_rank()

info = yaml_load_file("/home/iz221/cnc/mcmcs/cobaya/input_files/cobaya_cnc_planck_sim_unbinned_masscal.yaml")
#info = yaml_load_file("/home/iz221/cnc/mcmcs/cobaya/input_files/cobaya_cnc_planck_sim_binned.yaml")
#updated_info,sampler = run(info)

success = False

try:

    updated_info,sampler = run(info)
    success = True

except LoggedError as err:

    pass

success = all(comm.allgather(success))

if not success and rank == 0:

    print("Sampling failed!")
