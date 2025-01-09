
if os.getenv(env_var):
    path_to_cosmopower_organization = os.getenv(env_var)
    # print(f"{env_var} is already set to: {path_to_cosmopower_organization}")
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
