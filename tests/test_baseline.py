#Code to benchmark cosmocnc. It runs the tutorial on GitHub.
import numpy as np
import cosmocnc
import time
import pytest



cnc_params = cosmocnc.cnc_params_default
scal_rel_params = cosmocnc.scaling_relation_params_default
cosmo_params = cosmocnc.cosmo_params_default
#Catalogue and observables
cnc_params["cluster_catalogue"] = "SO_sim_0"
cnc_params["observables"] = [["q_so_sim"],["p_so_sim"]]
cnc_params["obs_select"] = "q_so_sim"
#Mass and redshift range
cnc_params["M_min"] = 1e13
cnc_params["M_max"] = 1e16
cnc_params["z_min"] = 0.01
cnc_params["z_max"] = 3.
#Selection observable range
cnc_params["obs_select_min"] = 5. #selection threshold
cnc_params["obs_select_max"] = 200.
#Precision parameters
cnc_params["n_points"] = 16384 #number of points in which the mass function at each redshift (and all the convolutions) is evaluated
cnc_params["n_points_data_lik"] = 2048 #number of points in which the mass function at each redshift (and all the convolutions) is evaluated
cnc_params["n_z"] = 100
cnc_params["sigma_mass_prior"] = 10
cnc_params["delta_m_with_ref"] = True
cnc_params["scalrel_type_deriv"] = "numerical"
cnc_params["downsample_hmf_bc"] = 2
cnc_params["compute_abundance_matrix"] = True
#Parallelisation
cnc_params["number_cores_hmf"] = 1
cnc_params["number_cores_abundance"] = 1
cnc_params["number_cores_data"] = 8
cnc_params["parallelise_type"] = "redshift"
#Cosmology parameters
cnc_params["cosmology_tool"] = "classy_sz"
cnc_params["cosmo_param_density"] = "critical"
cnc_params["cosmo_model"] = "lcdm"
cnc_params["hmf_calc"] = "cnc"
cnc_params["interp_tinker"] = "linear" #"linear" or "log" # only used if internal hmf is used
#Parameters for the binned likelihood
cnc_params["binned_lik_type"] = "z_and_obs_select"
cnc_params["bins_edges_z"] = np.linspace(cnc_params["z_min"],cnc_params["z_max"],9)
cnc_params["bins_edges_obs_select"] = np.exp(np.linspace(np.log(cnc_params["obs_select_min"]),np.log(cnc_params["obs_select_max"]),7))
#Stacked data, set to False for now
cnc_params["stacked_likelihood"] = False
cnc_params["stacked_data"] = ["p_so_sim_stacked"] #list of stacked data
cnc_params["compute_stacked_cov"] = True
#Initialisation
cnc_params["data_lik_from_abundance"] = False
cnc_params["likelihood_type"] = "unbinned"
scal_rel_params["corr_lnq_lnp"] = 0.
scal_rel_params["bias_sz"] = 0.8
number_counts = cosmocnc.cluster_number_counts()
number_counts.cnc_params = cnc_params
number_counts.scal_rel_params = scal_rel_params
number_counts.cosmo_params = cosmo_params


def test_ntot_classy_sz_cosmo_cosmocnc_hmf():
   start_time = time.time()
   number_counts.initialise()
   end_time = time.time()
   elapsed_time = end_time - start_time
   print(f"Took {elapsed_time:.2f} seconds to initialize")
   start_time = time.time()
   #Computation of the number counts
   number_counts.get_number_counts()
   dn_dz = number_counts.n_z
   dn_dSNR = number_counts.n_obs
   z = number_counts.redshift_vec
   SNR = number_counts.obs_select_vec
   
   n_tot_theory = number_counts.n_tot
   n_tot_obs = number_counts.catalogue.n_tot
   print("Predicted total number of clusters in the catalogue:",n_tot_theory,"+-",np.sqrt(n_tot_theory))
   print("Observed total number of clusters in the catalogue:",n_tot_obs)
   #Binned abundance
   cnc_params["likelihood_type"] = "binned"
   cnc_params["obs_select_min"] = 5.000
   number_counts.cnc_params = cnc_params


   log_lik = number_counts.get_log_lik() #Evaluate the binned likelihood
   bins_centres_z = number_counts.bins_centres_z #z bins centres
   bins_centres_snr = number_counts.bins_centres_obs #SNR bins centres
   n_binned_theory = number_counts.n_binned #Theoretical prediction for the number counts in the SNR-z bins
   n_binned_obs = number_counts.n_binned_obs #Actual number counts in the catalogue
   n_tot_bins_theory = np.sum(n_binned_theory) #Theoretical prediction for the total number of clusters in the bins
   n_tot_bins_obs = np.sum(n_binned_obs) #Actual number in the catalogue
   n_binned_snr_theory = np.sum(n_binned_theory,axis=0) #Theoretical prediction for the number counts in the SNR bins
   n_binned_snr_obs = np.sum(n_binned_obs,axis=0) #Actual number in the catalogue
   n_binned_z_theory = np.sum(n_binned_theory,axis=1) #Theoretical prediction for the number counts in the z bins
   n_binned_z_obs = np.sum(n_binned_obs,axis=1) #Actual number in the catalogue

   end_time = time.time()
   elapsed_time = end_time - start_time
   # Check the result is close to the expected value (15017)
   Ntot = n_tot_theory 
   assert Ntot == pytest.approx(15633.942224663508, rel=0.0002), f"Ntot is {Ntot}, expected close to  15633.942224663508"
   # Check the result is close to the expected value (15017)
   Ntot = n_tot_bins_theory
   assert Ntot == pytest.approx(15621.3, rel=0.0002), f"Ntot is {Ntot}, expected close to  15621.3"

   print("")
   print("Predicted total number of clusters in the bins = ",n_tot_bins_theory,"+-",np.sqrt(n_tot_bins_theory))
   print("Observed total number of clusters in the bins = ",n_tot_bins_obs)
   print(f"Test executed in {elapsed_time:.2f} seconds")
   

   
   # Calculate the percentage difference
   expected_array = np.array([1.1557058624e+04, 3.1940625320e+03, 7.3416807325e+02,
      1.2239569071e+02, 1.2896377151e+01, 7.2030685906e-01])
   percent_diff = np.abs((n_binned_snr_theory - expected_array) / expected_array) * 100
   # Print the percentage difference in all cases
   print(f"Percentage differences of n_binned_snr_theory: {percent_diff}")
   # Assert that the percentage difference is within 0.02%
   assert np.all(percent_diff < 0.02), f"Array values differ by more than 0.02%. Differences: {percent_diff}"


   # Calculate the percentage difference
   expected_array = np.array([5.3512308790e+03, 6.7567790779e+03, 2.6994097787e+03,
      6.6506120106e+02, 1.2534610320e+02, 1.9688601640e+01,
      3.1057628949e+00, 6.8019959041e-01])
   percent_diff = np.abs((n_binned_z_theory - expected_array) / expected_array) * 100
   # Print the percentage difference in all cases
   print(f"Percentage differences of n_binned_z_theory: {percent_diff}")
   # Assert that the percentage difference is within 0.02%
   assert np.all(percent_diff < 0.02), f"Array values differ by more than 0.02%. Differences: {percent_diff}"

   
   
   #Cluster catalogue
   catalogue = number_counts.catalogue
   q_obs = catalogue.catalogue["q_so_sim"] #tSZ signal-to-noise
   p_obs = catalogue.catalogue["p_so_sim"] #CMB lensing signal-to-noise
   z = catalogue.catalogue["z"] #Redshift
   #Stacked observable
   cnc_params["likelihood_type"] = "unbinned"
   cnc_params["observables"] = [["q_so_sim"]] #There is only one cluster-by-cluster mass observable.
   cnc_params["data_lik_from_abundance"] = False #So that the backward convolutional approach is followed (necessary for the stacked likelihood)
   cnc_params["stacked_likelihood"] = True
   cnc_params["stacked_data"] = ["p_so_sim_stacked"] #List of stacked data
   cnc_params["compute_stacked_cov"] = True
   number_counts.cnc_params = cnc_params
   number_counts.initialise()
   log_lik = number_counts.get_log_lik()

   # Calculate the percentage difference
   expected_log_lik = 98410.58588693838
   percent_diff = abs((log_lik - expected_log_lik) / expected_log_lik) * 100
   
   # Print the actual and expected values, and the percentage difference
   print(f"Actual log_lik: {log_lik}")
   print(f"Expected log_lik: {expected_log_lik}")
   print(f"Percentage difference: {percent_diff}%")
   
   # Assert that the percentage difference is within 0.02%
   assert percent_diff < 0.02, f"log_lik differs by more than 0.02%. Actual: {log_lik}, Expected: {expected_log_lik}"


   p_stacked_obs = number_counts.catalogue.stacked_data["p_so_sim_stacked"]["data_vec"]
   p_stacked_theory = number_counts.stacked_model["p_so_sim_stacked"]
   p_stacked_std = np.sqrt(number_counts.stacked_variance["p_so_sim_stacked"])
   print("Predicted stacked observable =",p_stacked_theory,"+-",p_stacked_std)
   print("Observed stacked observable =",p_stacked_obs)
   p_stacked_obs_2 = np.mean(p_obs)
   print("Observed stacked observable =",p_stacked_obs_2)

   # Assert that the predicted stacked observable and its standard deviation are close to expected values
   expected_p_stacked_theory = 0.30825248167029823
   expected_p_stacked_std = 0.00801369702374388

   assert np.isclose(p_stacked_theory, expected_p_stacked_theory, rtol=1e-5), \
      f"Predicted stacked observable {p_stacked_theory} is not close to expected value {expected_p_stacked_theory}"
   
   assert np.isclose(p_stacked_std, expected_p_stacked_std, rtol=1e-5), \
      f"Predicted stacked observable std {p_stacked_std} is not close to expected value {expected_p_stacked_std}"

   print(f"Assertion passed: Predicted stacked observable {p_stacked_theory} +- {p_stacked_std} is close to expected {expected_p_stacked_theory} +- {expected_p_stacked_std}")

   #Likelihood evaluation
   cnc_params["likelihood_type"] = "unbinned"
   cnc_params["observables"] = [["q_so_sim"]] #There is only one cluster-by-cluster mass observable.
   cnc_params["data_lik_from_abundance"] = True #So that the forward convolutional approach is followed (faster).
   cnc_params["stacked_likelihood"] = False




   number_counts.cnc_params = cnc_params
   number_counts.initialise()

   n = 20
   sigma_8_vec = np.linspace(0.808,0.815,n)
   log_lik = np.zeros(n)
   for i in range(0,n):
      print(i)
      cosmo_params["sigma_8"] = sigma_8_vec[i]
      number_counts.update_params(cosmo_params,scal_rel_params)
      log_lik[i] = number_counts.get_log_lik()
   
   lik_vec = np.exp(log_lik-np.max(log_lik))
   cnc_params["sigma_8"] = 0.811

   # Define the expected lik_vec
   expected_lik_vec = np.array([2.7304209931e-05, 2.2605144301e-04, 1.4774962771e-03,
         7.6235892989e-03, 3.1051244657e-02, 9.9829128872e-02,
         2.5331825348e-01, 5.0731618169e-01, 8.0179978578e-01,
         1.0000000000e+00, 9.8413037102e-01, 7.6417963606e-01,
         4.6816613124e-01, 2.2627554297e-01, 8.6273936072e-02,
         2.5947597104e-02, 6.1554708354e-03, 1.1517094754e-03,
         1.6994679560e-04, 1.9776166638e-05])

   # Test that all entries in lik_vec are close to the expected values
   np.testing.assert_allclose(lik_vec, expected_lik_vec, rtol=1e-8, atol=1e-10,
                              err_msg="lik_vec values do not match expected values")

   print("All lik_vec entries are close to expected values.")


   #Goodness of fit
   number_counts.cnc_params = cnc_params
   number_counts.initialise()
   C,C_mean,C_std = number_counts.get_c_statistic()
   print("Predicted C =",C_mean,"+-",C_std)
   print("Observed C =",C)

   # Test that predicted C is close to expected value
   expected_C_mean = 27.60197044996162
   expected_C_std = 7.118550901545521
   
   np.testing.assert_allclose(C_mean, expected_C_mean, rtol=1e-8, atol=1e-10,
                              err_msg=f"Predicted C mean {C_mean} is not close to expected {expected_C_mean}")
   np.testing.assert_allclose(C_std, expected_C_std, rtol=1e-8, atol=1e-10,
                              err_msg=f"Predicted C std {C_std} is not close to expected {expected_C_std}")
   
   print(f"Assertion passed: Predicted C {C_mean} +- {C_std} is close to expected {expected_C_mean} +- {expected_C_std}")



   #Mass estimation

   cnc_params["likelihood_type"] = "unbinned"
   cnc_params["observables"] = [["q_so_sim"]]
   cnc_params["data_lik_from_abundance"] = False #So that the backward convolutional approach is followed (needed).
   cnc_params["get_masses"] = True
   number_counts.cnc_params = cnc_params
   number_counts.initialise()
   number_counts.get_log_lik()
   number_counts.get_masses()
   ln_mass_est = number_counts.cluster_lnM #ln mass estimates
   ln_mass_std = number_counts.cluster_lnM_std #ln mass standard deviation
   mass_true = number_counts.catalogue.M #true mass
   m_x = mass_true
   m_y = np.exp(ln_mass_est)

   # Test that the sum of estimated masses is close to the expected value
   expected_sum = 53689.13350638108
   actual_sum = m_y.sum()
   
   np.testing.assert_allclose(actual_sum, expected_sum, rtol=1e-8, atol=1e-8,
                              err_msg=f"Sum of estimated masses {actual_sum} is not close to expected {expected_sum}")
   
   print(f"Assertion passed: Sum of estimated masses {actual_sum} is close to expected {expected_sum}")


   #Most extreme cluster
   number_counts.initialise()
   number_counts.get_log_lik_extreme_value()
   number_counts.eval_extreme_value_quantities()
   snr_max_mean = number_counts.obs_select_max_mean
   snr_max_std = number_counts.obs_select_max_std
   snr_max_obs = np.max(q_obs)
   print("Predicted maximum SNR",snr_max_mean,"+-",snr_max_std)
   print("Observed maximum SNR",snr_max_obs)

   # Test that predicted maximum SNR is close to expected value
   expected_snr_max_mean = 115.57641663774933
   expected_snr_max_std = 24.606812765830487
   
   np.testing.assert_allclose(snr_max_mean, expected_snr_max_mean, rtol=1e-8, atol=1e-10,
                              err_msg=f"Predicted maximum SNR mean {snr_max_mean} is not close to expected {expected_snr_max_mean}")
   np.testing.assert_allclose(snr_max_std, expected_snr_max_std, rtol=1e-8, atol=1e-10,
                              err_msg=f"Predicted maximum SNR std {snr_max_std} is not close to expected {expected_snr_max_std}")
   
   print(f"Assertion passed: Predicted maximum SNR {snr_max_mean} +- {snr_max_std} is close to expected {expected_snr_max_mean} +- {expected_snr_max_std}")