

import warnings
from contextlib import contextmanager
import logging


# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
dtype = tf.float32


class Restore_NN(tf.keras.Model):

    def __init__(self, 
                 parameters=None, 
                 modes=None, 
                 parameters_mean=None, 
                 parameters_std=None, 
                 features_mean=None, 
                 features_std=None, 
                 n_hidden=[512,512,512], 
                 restore=False, 
                 restore_filename=None, 
                 trainable=True,
                 optimizer=None,
                 verbose=False, 
                 ):
        
        # super
        super(Restore_NN, self).__init__()

        # restore
        
        self.restore(restore_filename)


        # input parameters mean and std
        self.parameters_mean = tf.constant(self.parameters_mean_, dtype=dtype, name='parameters_mean')
        self.parameters_std = tf.constant(self.parameters_std_, dtype=dtype, name='parameters_std')

        # (log)-spectra mean and std
        self.features_mean = tf.constant(self.features_mean_, dtype=dtype, name='features_mean')
        self.features_std = tf.constant(self.features_std_, dtype=dtype, name='features_std')

        # weights, biases and activation function parameters for each layer of the network
        self.W = []
        self.b = []
        self.alphas = []
        self.betas = [] 
        for i in range(self.n_layers):
            self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., 1e-3), name="W_" + str(i), trainable=trainable))
            self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name = "b_" + str(i), trainable=trainable))
        for i in range(self.n_layers-1):
            self.alphas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "alphas_" + str(i), trainable=trainable))
            self.betas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "betas_" + str(i), trainable=trainable))

        # restore weights if restore = True

        for i in range(self.n_layers):
            self.W[i].assign(self.W_[i])
            self.b[i].assign(self.b_[i])
        for i in range(self.n_layers-1):
            self.alphas[i].assign(self.alphas_[i])
            self.betas[i].assign(self.betas_[i])

        # optimizer
        self.optimizer = optimizer or tf.keras.optimizers.Adam()
        self.verbose= verbose

        # print initialization info, if verbose
        if self.verbose:
            multiline_str = "\nInitialized cosmopower_NN model, \n" \
                            f"mapping {self.n_parameters} input parameters to {self.n_modes} output modes, \n" \
                            f"using {len(self.n_hidden)} hidden layers, \n" \
                            f"with {list(self.n_hidden)} nodes, respectively. \n"
            print(multiline_str)



    # from https://github.com/HTJense/cosmopower/blob/packaging-paper/cosmopower/cosmopower_NN.py
    def restore(self, filename: str, allow_pickle: bool = False) -> None:
        r"""
        Load pre-trained model.
        The default file format is compressed numpy files (.npz). The
        Module will attempt to use this as a file extension and restore
        from there (i.e. look for `filename.npz`). If this file does
        not exist, and `allow_pickle` is set to True, then the file
        `filename.pkl` will be attempted to be read by `restore_pickle`.

        The function will trim the file extension from `filename`, so
        `restore("filename")` and `restore("filename.npz")` are identical.

        Parameters:
        :param filename: filename (without suffix) where model was saved.
        :param allow_pickle: whether or not to permit passing this filename
                             to the `restore_pickle` function.
        """
        # Check if npz file exists.
        filename_npz = filename + ".npz"
        if not os.path.exists(filename_npz):
            # Can we load this file as a pickle file?
            filename_pkl = filename + ".pkl"
            if allow_pickle and os.path.exists(filename_pkl):
                self.restore_pickle(filename_pkl)
                return

            raise IOError(f"Failed to restore network from {filename}: "
                          + (" is a pickle file, try setting 'allow_pickle = \
                              True'" if os.path.exists(filename_pkl) else
                             " does not exist."))

        with open(filename_npz, "rb") as fp:
            fpz = np.load(fp, allow_pickle=True)["arr_0"].flatten()[0]

            # print('filename_npz:', filename_npz)
            # print("Keys in the .npz file:", list(fpz.keys()))

            self.architecture = fpz["architecture"]
            self.n_layers = fpz["n_layers"]
            self.n_hidden = fpz["n_hidden"]
            self.n_parameters = fpz["n_parameters"]
            self.n_modes = fpz["n_modes"]

            self.parameters = list(fpz["parameters"])
            self.modes = fpz["modes"]

            # self.parameters_mean_ = fpz["parameters_mean"]
            # self.parameters_std_ = fpz["parameters_std"]
            # self.features_mean_ = fpz["features_mean"]
            # self.features_std_ = fpz["features_std"]

            # Attempt to load 'parameters_mean' or fall back to 'param_train_mean'
            self.parameters_mean_ = fpz.get("parameters_mean", fpz.get("param_train_mean"))
            self.parameters_std_ = fpz.get("parameters_std", fpz.get("param_train_std"))
            self.features_mean_ = fpz.get("features_mean", fpz.get("feature_train_mean"))
            self.features_std_ = fpz.get("features_std", fpz.get("feature_train_std"))


            # Fallback to 'weights_' if individual 'W_i' are not found
            if "weights_" in fpz:
                # Assign the list of weight arrays from 'weights_' directly
                self.W_ = fpz["weights_"]
            else:
                # Use individual weight arrays if available
                self.W_ = [fpz[f"W_{i}"] for i in range(self.n_layers)]

            # Fallback to 'biases_' if individual 'b_i' are not found
            if "biases_" in fpz:
                self.b_ = fpz["biases_"]
            else:
                self.b_ = [fpz[f"b_{i}"] for i in range(self.n_layers)]

            self.alphas_ = fpz[f"alphas_"]
            self.betas_ = fpz[f"betas_"]


    # auxiliary function to sort input parameters
    def dict_to_ordered_arr_np(self, 
                               input_dict, 
                               ):
        r"""
        Sort input parameters

        Parameters:
            input_dict (dict [numpy.ndarray]):
                input dict of (arrays of) parameters to be sorted

        Returns:
            numpy.ndarray:
                parameters sorted according to desired order
        """
        if self.parameters is not None:
            return np.stack([input_dict[k] for k in self.parameters], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)


    # forward prediction given input parameters implemented in Numpy
    def forward_pass_np(self, 
                        parameters_arr
                        ):
        r"""
        Forward pass through the network to predict the output, 
        fully implemented in Numpy

        Parameters:
            parameters_arr (numpy.ndarray):
                array of input parameters

        Returns:
          numpy.ndarray:
            output predictions
        """
        # forward pass through the network
        act = []
        layers = [(parameters_arr - self.parameters_mean_)/self.parameters_std_]
        for i in range(self.n_layers-1):

            # linear network operation
            act.append(np.dot(layers[-1], self.W_[i]) + self.b_[i])

            # pass through activation function
            layers.append((self.betas_[i] + (1.-self.betas_[i])*1./(1.+np.exp(-self.alphas_[i]*act[-1])))*act[-1])

        # final (linear) layer -> (standardised) predictions
        layers.append(np.dot(layers[-1], self.W_[-1]) + self.b_[-1])

        # rescale and output
        return layers[-1]*self.features_std_ + self.features_mean_


    # Numpy array predictions
    def predictions_np(self, 
                       parameters_dict
                       ):
        r"""
        Predictions given input parameters collected in a dict.
        Fully implemented in Numpy. Calls ``forward_pass_np``
        after ordering the input parameter dict

        Parameters:
            parameters_dict (dict [numpy.ndarray]):
                dictionary of (arrays of) parameters

        Returns:
            numpy.ndarray:
                output predictions
        """
        parameters_arr = self.dict_to_ordered_arr_np(parameters_dict)
        return self.forward_pass_np(parameters_arr)


    # Numpy array 10.**predictions
    def ten_to_predictions_np(self,
                            parameters_dict
                            ):
        r"""
        10^predictions given input parameters collected in a dict.
        Fully implemented in Numpy. It raises 10 to the output
        from ``forward_pass_np``

        Parameters:
            parameters_dict (dict [numpy.ndarray]):
                dictionary of (arrays of) parameters

        Returns:
            numpy.ndarray:
                10^output predictions
        """
        return 10.**self.predictions_np(parameters_dict)


class Restore_PCAplusNN(tf.keras.Model):

    def __init__(self, 
                 cp_pca=None,
                 n_hidden=[512,512,512], 
                 restore=False, 
                 restore_filename=None, 
                 trainable=True, 
                 optimizer=None,
                 verbose=False,
                 ):
        r"""
        Constructor.
        """
        # super
        super(Restore_PCAplusNN, self).__init__()


        self.restore(restore_filename)

        # input parameters mean and std
        self.parameters_mean = tf.constant(self.parameters_mean_, dtype=dtype, name='parameters_mean')
        self.parameters_std = tf.constant(self.parameters_std_, dtype=dtype, name='parameters_std')

        # PCA mean and std
        self.pca_mean = tf.constant(self.pca_mean_, dtype=dtype, name='pca_mean')
        self.pca_std = tf.constant(self.pca_std_, dtype=dtype, name='pca_std')

        # (log)-spectra mean and std
        self.features_mean = tf.constant(self.features_mean_, dtype=dtype, name='features_mean')
        self.features_std = tf.constant(self.features_std_, dtype=dtype, name='features_std')

        # pca transform matrix
        self.pca_transform_matrix = tf.constant(self.pca_transform_matrix_, dtype=dtype, name='pca_transform_matrix')

        # weights, biases and activation function parameters for each layer of the network
        self.W = []
        self.b = []
        self.alphas = []
        self.betas = [] 
        for i in range(self.n_layers):
            self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., np.sqrt(2./self.n_parameters)), name="W_" + str(i), trainable=trainable))
            self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name = "b_" + str(i), trainable=trainable))
        for i in range(self.n_layers-1):
            self.alphas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "alphas_" + str(i), trainable=trainable))
            self.betas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "betas_" + str(i), trainable=trainable))

        # restore weights if restore = True
        for i in range(self.n_layers):
            self.W[i].assign(self.W_[i])
            self.b[i].assign(self.b_[i])
        for i in range(self.n_layers-1):
            self.alphas[i].assign(self.alphas_[i])
            self.betas[i].assign(self.betas_[i])

        self.optimizer = optimizer or tf.keras.optimizers.Adam()
        self.verbose= verbose

        # print initialization info, if verbose
        if self.verbose:
            multiline_str = "\nInitialized cosmopower_PCAplusNN model, \n" \
                            f"mapping {self.n_parameters} input parameters to {self.n_pcas} PCA components \n" \
                            f"and then inverting the PCA compression to obtain {self.n_modes} modes \n" \
                            f"The model uses {len(self.n_hidden)} hidden layers, \n" \
                            f"with {list(self.n_hidden)} nodes, respectively. \n"
            print(multiline_str)



    # from https://github.com/HTJense/cosmopower/blob/packaging-paper/cosmopower/cosmopower_PCAplusNN.py
    def restore(self, filename: str, allow_pickle: bool = False) -> None:
        r"""
        Load pre-trained model.
        The default file format is compressed numpy files (.npz). The
        Module will attempt to use this as a file extension and restore
        from there (i.e. look for `filename.npz`). If this file does
        not exist, and `allow_pickle` is set to True, then the file
        `filename.pkl` will be attempted to be read by `restore_pickle`.

        The function will trim the file extension from `filename`, so
        `restore("filename")` and `restore("filename.npz")` are identical.

        Parameters:
        :param filename: filename (without suffix) where model was saved.
        :param allow_pickle: whether or not to permit passing this filename to
                             the `restore_pickle` function.
        """
        # Check if npz file exists.
        filename_npz = filename + ".npz"
        if not os.path.exists(filename_npz):
            # Can we load this file as a pickle file?
            filename_pkl = filename + ".pkl"
            if allow_pickle and os.path.exists(filename_pkl):
                self.restore_pickle(filename_pkl)
                return

            raise IOError(f"Failed to restore network from {filename}: "
                          + (" is a pickle file, try setting 'allow_pickle = \
                              True'" if os.path.exists(filename_pkl) else
                             " does not exist."))

        with open(filename_npz, "rb") as fp:
            fpz = np.load(fp, allow_pickle=True)["arr_0"].flatten()[0]

            # print('filename_npz:', filename_npz)
            # print("Keys in the .npz file:", list(fpz.keys()))


            self.architecture = fpz["architecture"]
            self.n_layers = fpz["n_layers"]
            self.n_hidden = fpz["n_hidden"]
            self.n_parameters = fpz["n_parameters"]
            self.n_modes = fpz["n_modes"]

            self.parameters = fpz["parameters"]
            self.modes = fpz["modes"]

            # Attempt to load 'parameters_mean' or fall back to 'param_train_mean'
            self.parameters_mean_ = fpz.get("parameters_mean", fpz.get("param_train_mean"))
            self.parameters_std_ = fpz.get("parameters_std", fpz.get("param_train_std"))
            self.features_mean_ = fpz.get("features_mean", fpz.get("feature_train_mean"))
            self.features_std_ = fpz.get("features_std", fpz.get("feature_train_std"))

            # Handle PCA-related keys
            # self.pca_mean_ = fpz["pca_mean"]
            # self.pca_std_ = fpz["pca_std"]
            # self.n_pcas = fpz["n_pcas"]

            self.pca_mean_ = fpz["pca_mean"]
            self.pca_std_ = fpz["pca_std"]
            self.n_pcas = fpz["n_pcas"]
            self.pca_transform_matrix_ = fpz["pca_transform_matrix"]

            # print('n_pcas:', self.n_pcas)

            # filename  = "/Users/boris/Work/CLASS-SZ/SO-SZ/cosmopower-organization/lcdm/TTTEEE/TE_v1.pkl"
            # f = open(filename, 'rb')
            # self.W_, self.b_, self.alphas_, self.betas_, \
            # self.parameters_mean_, self.parameters_std_, \
            # self.pca_mean_, self.pca_std_, \
            # self.features_mean_, self.features_std_, \
            # self.parameters, self.n_parameters, \
            # self.modes, self.n_modes, \
            # self.n_pcas, self.pca_transform_matrix_, \
            # self.n_hidden, self.n_layers, self.architecture = pickle.load(f)

            # print('self.n_pcas:', self.n_pcas)
            # print('self.pca_transform_matrix_:', self.pca_transform_matrix_)
            # print('PCA mean:', self.pca_mean_)
            # print('PCA std:', self.pca_std_)
            # f.close()
            # import sys
            # sys.exit(0)

            # self.pca_transform_matrix_ = fpz["pca_transform_matrix"]        

            # Fallback to 'weights_' if individual 'W_i' are not found
            if "weights_" in fpz:
                # Assign the list of weight arrays from 'weights_' directly
                self.W_ = fpz["weights_"]
            else:
                # Use individual weight arrays if available
                self.W_ = [fpz[f"W_{i}"] for i in range(self.n_layers)]

            # Fallback to 'biases_' if individual 'b_i' are not found
            if "biases_" in fpz:
                self.b_ = fpz["biases_"]
            else:
                self.b_ = [fpz[f"b_{i}"] for i in range(self.n_layers)]

            # Handle alphas and betas
            self.alphas_ = fpz.get("alphas_", [fpz.get(f"alphas_{i}") for i in range(self.n_layers - 1)])
            self.betas_ = fpz.get("betas_", [fpz.get(f"betas_{i}") for i in range(self.n_layers - 1)])



    # auxiliary function to sort input parameters
    def dict_to_ordered_arr_np(self, 
                               input_dict, 
                               ):
        r"""
        Sort input parameters

        Parameters:
            input_dict (dict [numpy.ndarray]):
                input dict of (arrays of) parameters to be sorted

        Returns:
            numpy.ndarray:
                parameters sorted according to desired order
        """
        if self.parameters is not None:
            return np.stack([input_dict[k] for k in self.parameters], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)


    # forward prediction given input parameters implemented in Numpy
    def forward_pass_np(self, 
                        parameters_arr,
                        ):
        r"""
        Forward pass through the network to predict the output, 
        fully implemented in Numpy

        Parameters:
            parameters_arr (numpy.ndarray):
                array of input parameters

        Returns:
          numpy.ndarray:
            output predictions
        """
        # forward pass through the network
        act = []
        layers = [(parameters_arr - self.parameters_mean_)/self.parameters_std_]
        for i in range(self.n_layers-1):

            # linear network operation
            act.append(np.dot(layers[-1], self.W_[i]) + self.b_[i])

            # pass through activation function
            layers.append((self.betas_[i] + (1.-self.betas_[i])*1./(1.+np.exp(-self.alphas_[i]*act[-1])))*act[-1])

        # final (linear) layer -> (normalized) PCA coefficients
        layers.append(np.dot(layers[-1], self.W_[-1]) + self.b_[-1])

        # rescale PCA coefficients, multiply out PCA basis -> normalised (log)-spectrum, shift and re-scale (log)-spectrum -> output (log)-spectrum
        return np.dot(layers[-1]*self.pca_std_ + self.pca_mean_, self.pca_transform_matrix_)*self.features_std_ + self.features_mean_


    def predictions_np(self, 
                       parameters_dict,
                       ):
        r"""
        Predictions given input parameters collected in a dict.
        Fully implemented in Numpy. Calls ``forward_pass_np``
        after ordering the input parameter dict

        Parameters:
            parameters_dict (dict [numpy.ndarray]):
                dictionary of (arrays of) parameters

        Returns:
            numpy.ndarray:
                output predictions
        """
        parameters_arr = self.dict_to_ordered_arr_np(parameters_dict)
        return self.forward_pass_np(parameters_arr)


    # 10.**predictions
    def ten_to_predictions_np(self,
                              parameters_dict,
                              ):
        r"""
        10^predictions given input parameters collected in a dict.
        Fully implemented in Numpy. It raises 10 to the output
        from ``forward_pass_np``

        Parameters:
            parameters_dict (dict [numpy.ndarray]):
                dictionary of (arrays of) parameters

        Returns:
            numpy.ndarray:
                10^output predictions
        """
        return 10.**self.predictions_np(parameters_dict)
