from cobaya.likelihood import Likelihood
from typing import Optional, Sequence
import numpy as np
import scipy.stats as stats

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

class cnc_bias_likelihood(Likelihood):

    filename_bias_vec: str = ""
    filename_bias_cov: str = ""

    def initialize(self):

        super().initialize()

        self._read_data()

    def _read_data(self):
        # Read the data from the file
        bias_vec = np.loadtxt(self.filename_bias_vec)
        cov_matrix = np.loadtxt(self.filename_bias_cov)

        # Extract the bias values and covariance matrix
        self.bias_vec = bias_vec
        # Check if the covariance matrix is positive definite   
        if not np.all(np.linalg.eigvals(cov_matrix) > 0):
            raise ValueError("Covariance matrix is not positive definite")
        # Check if the covariance matrix is symmetric       
        if not np.allclose(cov_matrix, cov_matrix.T):
            raise ValueError("Covariance matrix is not symmetric")
        # Check if the covariance matrix is square          
        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError("Covariance matrix is not square")
        # Check if the covariance matrix has the same dimensions as the bias vector
        if cov_matrix.shape[0] != self.bias_vec.shape[0]:
            raise ValueError("Covariance matrix and bias vector have different dimensions")
        # Check if the covariance matrix is invertible
        if np.linalg.cond(cov_matrix) > 1 / np.finfo(cov_matrix.dtype).eps:
            raise ValueError("Covariance matrix is not invertible")
        
        self.inv_cov = np.linalg.inv(cov_matrix)

    def get_requirements(self):

        return {"biases": {}}

    def _get_theory(self, **params_values):

        theory = self.provider.get_biases()

        return theory

    def logp(self, **params_values):

        theory = self._get_theory(**params_values)
        res = theory - self.bias_vec
        chi2 = np.dot(res, self.inv_cov.dot(res))
        loglkl = -0.5*chi2

        return loglkl


class so_cmb_prior_wcdm(Likelihood): #warning: h as input instead of H0

    def initialize(self):

        fisher = np.array([[ 9.00114848e+02,  3.44938030e+04, -1.01268886e+06, 3.29630918e+05,
          -8.21634769e+12, -1.99241898e+04, -5.18764284e+03],
         [ 3.44938030e+04,  8.69979088e+06, -6.33192100e+07,  1.58665297e+07,
          -2.01779166e+15, -3.59432063e+06, -1.98798554e+05],
         [-1.01268886e+06, -6.33192100e+07,  1.88009030e+09, -3.46188758e+08,
           1.49005171e+16,  4.40025363e+07,  5.83644203e+06],
         [ 3.29630918e+05,  1.58665297e+07, -3.46188758e+08, 1.31768098e+08,
          -3.59709579e+15, -7.04475964e+06, -1.89976587e+06],
         [-8.21634769e+12, -2.01779166e+15,  1.49005171e+16, -3.59709579e+15,
           4.73428704e+23,  8.41055456e+14,  4.73533765e+13],
         [-1.99241898e+04, -3.59432063e+06,  4.40025363e+07, -7.04475964e+06,
           8.41055456e+14,  2.08085978e+06,  1.14829325e+05],
         [-5.18764284e+03, -1.98798554e+05,  5.83644203e+06, -1.89976587e+06,
           4.73533765e+13,  1.14829325e+05,  2.98980050e+04]])

        self.minus_half_invcov = - 0.5*fisher
        H0_true = 67.4
        h = H0_true/100.
        tau_reio_true = 0.06
        Onu0h2_true = 0.00064412
        w0_true = -1.
        Ob0h2_true = 0.022245895
        Oc0h2_true = 0.315*h**2-Ob0h2_true
        A_s_true = 2.08467e-09
        n_s_true = 0.96

        self.param_vec_true = np.array([H0_true,tau_reio_true,Ob0h2_true,Oc0h2_true,A_s_true,n_s_true,w0_true])

    def get_requirements(self):

        return {"H0": None,"tau_reio":None,"w0":None,"Ob0h2":None,"Oc0h2":None,"A_s":None,"n_s":None}

    def logp(self, **params_values):

        H0 = self.provider.get_param("H0")
        tau_reio = self.provider.get_param("tau_reio")
        w0 = self.provider.get_param("w0")
        Ob0h2 = self.provider.get_param("Ob0h2")
        Oc0h2 = self.provider.get_param("Oc0h2")
        A_s = self.provider.get_param("A_s")
        n_s = self.provider.get_param("n_s")

        param_vec = np.array([H0,tau_reio,Ob0h2,Oc0h2,A_s,n_s,w0])
        res = param_vec - self.param_vec_true

        print("dot",self.minus_half_invcov.dot(res))

        log_lik = np.transpose(res).dot(self.minus_half_invcov.dot(res))

        print("log lik",log_lik)

        return log_lik



class so_cmb_prior_nulcdm(Likelihood):

    def initialize(self):

        cov_matrix = np.array([[ 1.06502e+00, -5.16621e-03, -8.96376e-04,  2.05383e-05, -6.99822e-04,
                      -2.62669e-11,  1.42877e-03],
                     [-5.16621e-03,  9.36274e-05,  6.38655e-06, -3.43076e-08, -1.17204e-06,
                       3.98447e-13,  2.53291e-06],
                     [-8.96376e-04,  6.38655e-06,  8.26514e-07, -1.36349e-08,  4.32241e-07,
                       3.01123e-14, -9.09089e-07],
                     [ 2.05383e-05, -3.43076e-08, -1.36349e-08,  2.94467e-09, -1.63808e-08,
                      -1.82160e-16, -2.31964e-08],
                     [-6.99822e-04, -1.17204e-06,  4.32241e-07, -1.63808e-08,  8.21531e-07,
                      -4.88633e-16, -1.70722e-06],
                     [-2.62669e-11,  3.98447e-13,  3.01123e-14, -1.82160e-16, -4.88633e-16,
                       1.72965e-21, -2.89565e-15],
                     [ 1.42877e-03,  2.53291e-06, -9.09089e-07, -2.31964e-08, -1.70722e-06,
                      -2.89565e-15,  6.65044e-06]])

        self.minus_half_invcov = - 0.5*np.linalg.inv(cov_matrix)

        H0_true = 67.4
        h = H0_true/100.
        tau_reio_true = 0.06
        Onu0h2_true = 0.00064412
        Ob0h2_true = 0.022245895
        Oc0h2_true = 0.315*h**2-Ob0h2_true
        A_s_true = 2.08467e-09
        n_s_true = 0.96

        self.param_vec_true = np.array([H0_true,tau_reio_true,Onu0h2_true,Ob0h2_true,Oc0h2_true,A_s_true,n_s_true])

    def get_requirements(self):

        return {"H0": None,"tau_reio":None,"Onu0h2":None,"Ob0h2":None,"Oc0h2":None,"A_s":None,"n_s":None}

    def logp(self, **params_values):

        H0 = self.provider.get_param("H0")
        tau_reio = self.provider.get_param("tau_reio")
        Onu0h2 = self.provider.get_param("Onu0h2")
        Ob0h2 = self.provider.get_param("Ob0h2")
        Oc0h2 = self.provider.get_param("Oc0h2")
        A_s = self.provider.get_param("A_s")
        n_s = self.provider.get_param("n_s")

        param_vec = np.array([H0,tau_reio,Onu0h2,Ob0h2,Oc0h2,A_s,n_s])
        res = param_vec - self.param_vec_true

        log_lik = np.transpose(res).dot(self.minus_half_invcov.dot(res))

        return log_lik

class so_cmb_prior_nulcdm_mnu(Likelihood):

    def initialize(self):

        cov_matrix = np.array([[ 1.06502e+00, -5.16621e-03, -8.96376e-04,  2.05383e-05, -6.99822e-04,
                      -2.62669e-11,  1.42877e-03],
                     [-5.16621e-03,  9.36274e-05,  6.38655e-06, -3.43076e-08, -1.17204e-06,
                       3.98447e-13,  2.53291e-06],
                     [-8.96376e-04,  6.38655e-06,  8.26514e-07, -1.36349e-08,  4.32241e-07,
                       3.01123e-14, -9.09089e-07],
                     [ 2.05383e-05, -3.43076e-08, -1.36349e-08,  2.94467e-09, -1.63808e-08,
                      -1.82160e-16, -2.31964e-08],
                     [-6.99822e-04, -1.17204e-06,  4.32241e-07, -1.63808e-08,  8.21531e-07,
                      -4.88633e-16, -1.70722e-06],
                     [-2.62669e-11,  3.98447e-13,  3.01123e-14, -1.82160e-16, -4.88633e-16,
                       1.72965e-21, -2.89565e-15],
                     [ 1.42877e-03,  2.53291e-06, -9.09089e-07, -2.31964e-08, -1.70722e-06,
                      -2.89565e-15,  6.65044e-06]])

        Onu02_to_mnu = 93.14

        cov_matrix[2,:] = cov_matrix[2,:]*Onu02_to_mnu
        cov_matrix[:,2] = cov_matrix[:,2]*Onu02_to_mnu

        self.minus_half_invcov = - 0.5*np.linalg.inv(cov_matrix)

        H0_true = 67.4
        h = H0_true/100.
        tau_reio_true = 0.06
        Onu0h2_true = 0.00064412
        Ob0h2_true = 0.022245895
        Oc0h2_true = 0.315*h**2-Ob0h2_true
        A_s_true = 2.08467e-09
        n_s_true = 0.96

        mnu_true = Onu0h2_true*Onu02_to_mnu

        self.param_vec_true = np.array([H0_true,tau_reio_true,mnu_true,Ob0h2_true,Oc0h2_true,A_s_true,n_s_true])

    def get_requirements(self):

        return {"H0": None,"tau_reio":None,"m_nu":None,"Ob0h2":None,"Oc0h2":None,"A_s":None,"n_s":None}

    def logp(self, **params_values):

        H0 = self.provider.get_param("H0")
        tau_reio = self.provider.get_param("tau_reio")
        m_nu = self.provider.get_param("m_nu")
        Ob0h2 = self.provider.get_param("Ob0h2")
        Oc0h2 = self.provider.get_param("Oc0h2")
        A_s = self.provider.get_param("A_s")
        n_s = self.provider.get_param("n_s")

        param_vec = np.array([H0,tau_reio,m_nu,Ob0h2,Oc0h2,A_s,n_s])
        res = param_vec - self.param_vec_true

        log_lik = np.transpose(res).dot(self.minus_half_invcov.dot(res))

        return log_lik


class desi_prior_nulcdm(Likelihood):

    def initialize(self):

        fisher = np.array([[1.369654863353185803e+01, -9.855313873854645863e+03, -4.484472083574374892e+02, 2.365607125643557751e+01],
        [-9.855313873854645863e+03, 7.225270719267868437e+06, 2.665953771213945001e+05, -1.819894418709868478e+04],
        [-4.484472083574374892e+02, 2.665953771213945001e+05, 3.817212923439533915e+04, -2.814617165869603355e+02],
        [2.365607125643557751e+01, -1.819894418709868478e+04 ,-2.814617165869603355e+02 ,5.120817278151554319e+01]])

        self.minus_half_invcov = - 0.5*fisher

        H0_true = 67.4
        h = H0_true/100.
        mnu_true = 0.06
        Ob0h2_true = 0.022245895
        Oc0h2_true = 0.315*h**2-Ob0h2_true

        self.param_vec_true = np.array([H0_true,Ob0h2_true,Oc0h2_true,mnu_true])

    def get_requirements(self):

        return {"H0": None,"Ob0h2":None,"Oc0h2":None,"m_nu":None}

    def logp(self, **params_values):

        H0 = self.provider.get_param("H0")
        Ob0h2 = self.provider.get_param("Ob0h2")
        Oc0h2 = self.provider.get_param("Oc0h2")
        mnu = self.provider.get_param("m_nu")

        param_vec = np.array([H0,Ob0h2,Oc0h2,mnu])
        res = param_vec - self.param_vec_true

        log_lik = np.transpose(res).dot(self.minus_half_invcov.dot(res))

        return log_lik


class desi_prior_wcdm(Likelihood):

    def initialize(self):

        fisher = np.array([[1.369654863353185803e+01,  -9.855313873854645863e+03, -4.484472083574374892e+02, 3.103177771032207488e+02],
        [-9.855313873854645863e+03, 7.225270719267868437e+06, 2.665953771213945001e+05,-2.312677425005600380e+05],
        [-4.484472083574374892e+02, 2.665953771213945001e+05, 3.817212923439533915e+04,-6.818149725908029723e+03],
        [3.103177771032207488e+02, -2.312677425005600380e+05, -6.818149725908029723e+03,7.563758947320854531e+03]])

        self.minus_half_invcov = - 0.5*fisher

        H0_true = 67.4
        h = H0_true/100.
        w0_true = -1.
        Ob0h2_true = 0.022245895
        Oc0h2_true = 0.315*h**2-Ob0h2_true

        self.param_vec_true = np.array([H0_true,Ob0h2_true,Oc0h2_true,w0_true])

    def get_requirements(self):

        return {"H0": None,"Ob0h2":None,"Oc0h2":None,"w0":None}

    def logp(self, **params_values):

        H0 = self.provider.get_param("H0")
        Ob0h2 = self.provider.get_param("Ob0h2")
        Oc0h2 = self.provider.get_param("Oc0h2")
        w0 = self.provider.get_param("w0")

        param_vec = np.array([H0,Ob0h2,Oc0h2,w0])
        res = param_vec - self.param_vec_true

        log_lik = np.transpose(res).dot(self.minus_half_invcov.dot(res))

        return log_lik

class desi_prior_lcdm(Likelihood):

    def initialize(self):

        fisher = np.array([[1.369654863353185803e+01, -9.855313873854645863e+03, -4.484472083574374892e+02, 2.365607125643557751e+01],
        [-9.855313873854645863e+03, 7.225270719267868437e+06, 2.665953771213945001e+05, -1.819894418709868478e+04],
        [-4.484472083574374892e+02, 2.665953771213945001e+05, 3.817212923439533915e+04, -2.814617165869603355e+02],
        [2.365607125643557751e+01, -1.819894418709868478e+04 ,-2.814617165869603355e+02 ,5.120817278151554319e+01]])

        indices = [0,1,2]

        fisher = fisher[indices,:]
        fisher = fisher[:,indices]

        self.minus_half_invcov = - 0.5*fisher

        H0_true = 67.4
        h = H0_true/100.
        Ob0h2_true = 0.022245895
        Oc0h2_true = 0.315*h**2-Ob0h2_true

        self.param_vec_true = np.array([H0_true,Ob0h2_true,Oc0h2_true])

    def get_requirements(self):

        return {"H0": None,"Ob0h2":None,"Oc0h2":None}

    def logp(self, **params_values):

        H0 = self.provider.get_param("H0")
        Ob0h2 = self.provider.get_param("Ob0h2")
        Oc0h2 = self.provider.get_param("Oc0h2")

        param_vec = np.array([H0,Ob0h2,Oc0h2])
        res = param_vec - self.param_vec_true

        log_lik = np.transpose(res).dot(self.minus_half_invcov.dot(res))

        return log_lik
