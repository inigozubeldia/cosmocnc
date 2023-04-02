A very fast cluster number counts likelihood package

To install do:

$ pip install -e .

To test do:

$ python /path/to/cnc/test_files/test_module.py

$ python /path/to/cnc/test_files/test.py


To run a chain:

$ mpirun -np 4 cobaya-run /path/to/cnc/mcmcs/cobaya/input_files/cobaya_soliket_cnc_unbinned_cluster_counts_planck_template.yaml -f

Make sure you adapt the path in the yaml file !

This works with SOLIKET, on the branch dev-ymap_power_spectrum (boris's main soliket branch)
