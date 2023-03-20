if self.cnc_params["other_obs_uncorrelated"] == False:

    covariance_matrix.diagonalise()
    covariance_matrix.repeat(self.cnc_params["n_points"])

    A = covariance_matrix.A_repeat
    A_inv = covariance_matrix.A_inv_repeat
    eigenvalues = covariance_matrix.eigenvalues

                dn_dx0_vec = np.repeat(halo_mass_function_z[np.newaxis,:],len(observables_select),axis=0)
                x0_vec = np.repeat(self.ln_M[np.newaxis,:],len(observables_select),axis=0)

                for j in range(0,self.scal_rel_selection.get_n_layers(observable=self.cnc_params["obs_select"])):

                    x1_vec = np.zeros((len(observables_select,len(halo_mass_function_z))))
                    dn_dx1_vec = np.zeros((len(observables_select,len(halo_mass_function_z))))

                    for k in range(0,len(observables_select)):

                        self.scal_rel_selection.precompute_scaling_relation(observable=observables_select[k],
                        other_params=other_params,layer=k,patch_index=observable_patches[k])

                        x1 = self.scal_rel_selection.eval_scaling_relation(x0_vec[j],observable=observables_select[k],
                        layer=k,patch_index=observable_patches[k])

                        dx1_dx0 = self.scal_rel_selection.eval_derivative_scaling_relation(x0_vec[j],
                        observable=observables_select[k],layer=k,patch_index=observable_patches[k])

                        dn_dx1 = dn_dx0_vec[j]/dx1_dx0
                        x1_interp = np.linspace(np.min(x1),np.max(x1),self.cnc_params["n_points"])
                        dn_dx1 = np.interp(x1_interp,x1,dn_dx1)

                        x1_vec[k,:] = x1_interp
                        dn_dx1_vec[k,:] = dn_dx1

                    if self.cnc_params["other_obs_uncorrelated"] == False:

                        dn_dx1_vec = np.einsum("ik,ijk->jk",dn_dx1_vec,A_inv[j])
                        x1_vec = np.einsum("ijk,jk->ik",A[j],x1_vec)

                    for k in range(0,len(observables_select)):

                        if self.cnc_params["other_obs_uncorrelated"] == False:

                            sigma_scatter = np.sqrt(eigenvalues[j][k])

                        else:

                            sigma_scatter = np.sqrt(covariance_matrix.cov[j][k,k])

                        dn_dx1_vec = convolve_dn(x1_vec,dn_dx1_vec,sigma_scatter)

                    if self.cnc_params["other_obs_uncorrelated"] == False:

                        dn_dx1_vec = np.einsum("ik,ijk->jk",dn_dx1_vec,A[j])
                        x1_vec = np.einsum("ijk,jk->ik",A_inv[j],x1_vec)

                    dn_dx0_vec = dn_dx1_vec
                    x0_vec = x1_vec




                if optimise_mass == True:

                    obs_select_cluster = self.catalogue.catalogue[self.cnc_params["obs_select"]][cluster_index]

                    for k in range(0,self.scal_rel_selection.get_n_layers(observable=self.cnc_params["obs_select"])):

                        patch_index_obs_select = self.catalogue.catalogue_patch[self.cnc_params["obs_select"]][cluster_index]

                        sigma_scatter = np.sqrt(self.scatter.get_cov(observable1=self.cnc_params["obs_select"],observable2=self.cnc_params["obs_select"],
                        layer=k,patch1=patch_index_obs_select,
                        patch2=patch_index_obs_select))

                        obs_min = obs_select_cluster - self.cnc_params["sigma_obs_hmf"]
                        obs_max = obs_select_cluster + self.cnc_params["sigma_obs_hmf"]

                        self.scal_rel_selection.precompute_scaling_relation(observable=self.cnc_params["obs_select"],
                        other_params=other_params,layer=k,patch_index=patch_index_obs_select)

                        x1 = self.scal_rel_selection.eval_scaling_relation(x0,observable=self.cnc_params["obs_select"],
                        layer=k,patch_index=indices_split[rank][i])







                    if self.cnc_params["other_obs_uncorrelated"] == False:

                        dn_dx1_vec = np.einsum("ik,ijk->jk",dn_dx1_vec,A_inv[j])
                        x1_vec = np.einsum("ijk,jk->ik",A[j],x1_vec)

                    for k in range(0,len(observables_select)):

                        if self.cnc_params["other_obs_uncorrelated"] == False:

                            sigma_scatter = np.sqrt(eigenvalues[j][k])

                        else:

                            sigma_scatter = np.sqrt(covariance_matrix.cov[j][k,k])

                        dn_dx1_vec = convolve_dn(x1_vec,dn_dx1_vec,sigma_scatter)

                    if self.cnc_params["other_obs_uncorrelated"] == False:

                        dn_dx1_vec = np.einsum("ik,ijk->jk",dn_dx1_vec,A[j])
                        x1_vec = np.einsum("ijk,jk->ik",A_inv[j],x1_vec)

                    dn_dx0_vec = dn_dx1_vec
                    x0_vec = x1_vec
