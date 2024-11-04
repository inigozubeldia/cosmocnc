import subprocess
import os


# set some paths, e.g.:
# This path needs to be adjsuted (TBD: set this automatically)
#path_to_sd_projects = "/scratch/nas_chluba/specdist/bolliet/specdist_ml/"


#the path to save the output from cosmotherm
#path_to_recfast_results =  path_to_sd_projects + "specdist/specdist/recfast_outputs"

# make a directory
#subprocess.call(['mkdir','-p',path_to_recfast_results])

root_path = os.path.dirname(os.path.abspath(__file__)) + '/../'

# root_path = os.path.abspath("")
# assuming cosmopower organization codes are one level up:
# path_to_cosmopower_organization = root_path + '/../cosmopower-organization/'
path_to_cosmopower_organization = '/rds-d4/user/iz221/hpc-work/cosmopower/'
# print(path_to_cosmopower_organization)
# Check if the environment variable is already set
env_var = 'PATH_TO_COSMOPOWER_ORGANIZATION'

if os.getenv(env_var):
    path_to_cosmopower_organization = os.getenv(env_var)
    print(f"{env_var} is already set to: {path_to_cosmopower_organization}")
else:
    print(f"Warning: {env_var} is not set.")
    print("defaulting to ", path_to_cosmopower_organization)


env_var = 'PATH_TO_COSMOCNC'

if os.getenv(env_var):
    path_to_cosmocnc = os.getenv(env_var)
    print(f"{env_var} is already set to: {path_to_cosmocnc}")
else:
    print(f"Warning: {env_var} is not set.")
    path_to_cosmocnc = root_path
    print("defaulting to ", path_to_cosmocnc)
