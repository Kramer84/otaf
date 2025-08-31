from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "KrigingWrapper",
]


import openturns as ot
import numpy as np
import math


class KrigingWrapper:
    """
    A wrapper for Kriging models using the OpenTURNS library.

    Attributes
    ----------
    library : str
        Name of the library used.
    model : ot.KrigingResult or None
        The trained Kriging model.
    mean_function : ot.Basis
        The mean function used in the Kriging model.
    kernel_function : ot.CovarianceModel
        The kernel function used in the Kriging model.
    nugget : float
        The noise term added to the diagonal of the covariance matrix.
    input_dim : int
        Dimensionality of the input data.
    train_dataframe : None
        Placeholder for training data.
    x_train : ot.Sample
        Training input data.
    z_train : ot.Sample
        Training output data.
    test_dataframe : None
        Placeholder for test data.
    x_test : ot.Sample
        Test input data.
    z_postmean : np.ndarray
        Predicted mean values for test data.
    z_postvar : np.ndarray
        Predicted variance values for test data.
    """

    def __init__(self):
        self.library = "openturns"
        self.model = None
        self.mean_function = None
        self.kernel_function = None
        self.nugget = None
        self.input_dim = 1
        self.train_dataframe = None
        self.x_train = None
        self.z_train = None
        self.test_dataframe = None
        self.x_test = None
        self.z_postmean = None
        self.z_postvar = None

    def load_data(self, x_train: np.ndarray, z_train: np.ndarray):
        """
        Load and configure the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training input data.
        z_train : np.ndarray
            Training output data.
        """
        self.x_train = ot.Sample(x_train)
        self.z_train = ot.Sample(np.reshape(z_train, (len(self.x_train), 1)))
        self.input_dim = x_train.shape[1]

    def set_kernel(self, kernel: dict, ard: bool = True):
        """
        Set the kernel function for the Kriging model.

        Parameters
        ----------
        kernel : dict
            Dictionary containing kernel parameters.
        ard : bool, optional
            Automatic Relevance Determination (ARD), default is True.

        Raises
        ------
        ValueError
            If the specified kernel function is not supported.
        """
        kernel_name = kernel.get("name")
        lengthscale = kernel.get("lengthscale")
        variance = kernel.get("variance")

        if kernel_name == "Matern":
            self.kernel_function = ot.MaternModel(
                lengthscale, [math.sqrt(variance)], float(kernel["order"])
            )
        elif kernel_name == "Gaussian":
            self.kernel_function = ot.SquaredExponential(lengthscale, [math.sqrt(variance)])
        else:
            raise ValueError("This library does not support the specified kernel function")

    def set_mean(self, mean: str):
        """
        Set the mean function for the Kriging model.

        Parameters
        ----------
        mean : str
            The mean function type ('constant' or 'zero').

        Raises
        ------
        ValueError
            If the specified mean function is not supported.
        """
        if mean == "constant":
            self.mean_function = ot.ConstantBasisFactory(self.input_dim).build()
        elif mean == "zero":
            self.mean_function = ot.Basis()
        else:
            raise ValueError("This library does not support the specified mean function")

    def init_model(self, noise: float):
        """
        Initialize the Kriging model with the specified noise.

        Parameters
        ----------
        noise : float
            The noise term to add to the diagonal of the covariance matrix.

        Raises
        ------
        ValueError
            If the kernel function or mean function has not been set.
        """
        if not self.kernel_function or not self.mean_function:
            raise ValueError(
                "Kernel function and mean function must be set before initializing the model"
            )

        self.nugget = noise
        self.kriging_algorithm = ot.KrigingAlgorithm(
            self.x_train, self.z_train, self.kernel_function, self.mean_function
        )
        self.kriging_algorithm.setNoise([self.nugget] * len(self.x_train))

    def optimize(self, param_opt: str, itr: int = 100):
        """
        Optimize the Kriging model parameters.

        Parameters
        ----------
        param_opt : str
            Parameter optimization method ('MLE' or 'Not_optimize').
        itr : int, optional
            Number of iterations, default is 100.

        Raises
        ------
        ValueError
            If the specified parameter optimizer is not supported.
        """
        if param_opt == "MLE":
            self.kriging_algorithm.setOptimizeParameters(optimizeParameters=True)
        elif param_opt == "Not_optimize":
            self.kriging_algorithm.setOptimizeParameters(optimizeParameters=False)
        else:
            raise ValueError("This library does not support the specified Parameter optimizer")

        print(self.kernel_function.getFullParameterDescription())
        print("Parameter before optimization: ", self.kernel_function.getFullParameter())

        self.kriging_algorithm.run()

        self.model = self.kriging_algorithm.getResult()
        print("Parameter after optimization: \n", self.model.getCovarianceModel())
        print("Nugget", self.kriging_algorithm.getNoise())

    def get_NLL(self) -> float:
        """
        Get the Negative Log-Likelihood (NLL) of the Kriging model.

        Returns
        -------
        float
            The negative log-likelihood.
        """
        lik_function = self.kriging_algorithm.getReducedLogLikelihoodFunction()
        NLL = -lik_function(self.model.getCovarianceModel().getScale())
        return NLL[0]

    def predict(self, x_test: np.ndarray):
        """
        Make predictions for the test data.

        Parameters
        ----------
        x_test : np.ndarray
            Test input data.

        Returns
        -------
        tuple of np.ndarray
            Predicted mean and variance for the test data.

        Raises
        ------
        ValueError
            If the model has not been initialized or optimized.
        """
        self.x_test = ot.Sample(x_test)

        if not self.model:
            raise ValueError("Model has not been initialized or optimized")

        self.z_postmean = np.array(self.model.getConditionalMean(self.x_test))
        self.z_postvar = np.sqrt(
            np.add(np.diag(np.array(self.model.getConditionalCovariance(self.x_test))), self.nugget)
        )

        return self.z_postmean, self.z_postvar
