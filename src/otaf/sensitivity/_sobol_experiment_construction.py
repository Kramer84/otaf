__author__ = "Kramer84"
__version__ = "0.1"
__date__ = "21.05.24"

__all__ = ["SobolIndicesExperimentWithComposedDistribution"]

import openturns as ot

class SobolIndicesExperimentWithComposedDistribution(ot.SobolIndicesExperiment):
    """Sobol indices experiment wrapper using an OpenTURNS ComposedDistribution.

    Generates the specialized experimental design matrix (mixture matrix) required 
    for estimating first, second, and total-order Sobol sensitivity indices.

    Parameters
    ----------
    composedDistribution : openturns.ComposedDistribution, optional
        The joint distribution of the input variables. Default is None.
    size : int, optional
        The size $N$ of the base samples $A$ and $B$. Default is None.
    second_order : bool, optional
        If True, structures the experimental design to allow estimation of 
        second-order Sobol indices. Default is False.
    """

    def __init__(self, composedDistribution=None, size=None, second_order=False):
        self.composedDistribution = composedDistribution
        self.size = None

        self.__visibility__ = True
        self.__name__ = "Unnamed"
        self.__shadowedId__ = None
        self.__computeSecondOrder__ = second_order

        if size is not None:
            self.setSize(size)
        if composedDistribution is not None:
            self.setComposedDistribution(composedDistribution)

        self._sample_A = None
        self._sample_B = None
        self._experimentSample = None

    def generate(self, **kwargs):
        """Generate and return the final design matrix (mixture matrix).

        Parameters
        ----------
        method : str, optional
            The sampling method used to generate base designs.
            Options are: 'MonteCarlo', 'LHS', 'QMC'. Default is 'MonteCarlo'.
        sequence : str, optional
            The low-discrepancy sequence type. Only evaluated if `method='QMC'`.
            Options are: 'Faure', 'Halton', 'ReverseHalton', 'Haselgrove', 'Sobol'.
            Default is 'Sobol'.

        Returns
        -------
        openturns.Sample
            The combined mixture matrix containing experimental design configurations.

        Raises
        ------
        AssertionError
            If either `size` or `composedDistribution` have not been initialized.
        """
        assert (self.composedDistribution is not None) and (
            self.size is not None
        ), "Please intialize sample size and composed distribution"
        self._generateSample(**kwargs)

        self._mixSamples()
        self._experimentSample.setDescription(self.composedDistribution.getDescription())
        return self._experimentSample

    def generateWithWeights(self, **kwargs):
        """Not implemented. Kept for OpenTURNS API compatibility."""
        pass

    def getClassName(self):
        """Return the name of the class.

        Returns
        -------
        str
            The class name string.
        """
        return self.__class__.__name__

    def getId(self):
        """Return the unique object identifier.

        Returns
        -------
        int
            The unique ID (memory address) of the instance.
        """
        return id(self)

    def getName(self):
        """Return the name of the object.

        Returns
        -------
        str
            The internal name assigned to the instance.
        """
        return self.__name__

    def getShadowedId(self):
        """Return the shadowed ID of the object.

        Returns
        -------
        int or None
            The shadowed identifier.
        """
        return self.__shadowedId__

    def getSize(self):
        """Return the total number of rows in the generated mixture matrix.

        Returns
        -------
        int
            The row count of the generated design matrix, or 0 if it has 
            not yet been generated.
        """
        if self._experimentSample is None:
            return 0
        else:
            return len(self._experimentSample)

    def getVisibility(self):
        """Return the internal visibility flag status.

        Returns
        -------
        bool
            The visibility status tracking parameter.
        """
        return self.__visibility__

    def hasName(self):
        """Check if the object has a valid, non-empty name.

        Returns
        -------
        bool
            False if `__name__` is None or empty; True otherwise.
        """
        if len(self.__name__) == 0 or self.__name__ is None:
            return False
        else:
            return True

    def hasUniformWeights(self):
        """Not implemented. Kept for OpenTURNS API compatibility.

        Returns
        -------
        None
            Always returns None.
        """
        return None

    def hasVisibleName(self):
        """Check if the object name is distinct from its default value.

        Returns
        -------
        bool
            False if the name is "Unnamed" or empty; True otherwise.
        """
        if self.__name__ == "Unnamed" or len(self.__name__) == 0:
            return False
        else:
            return True

    def setComposedDistribution(self, composedDistribution):
        """Set the joint input distribution.

        Parameters
        ----------
        composedDistribution : openturns.ComposedDistribution
            The new joint distribution model to be assigned.
        """
        self.composedDistribution = composedDistribution

    def setName(self, name):
        """Set the internal object name.

        Parameters
        ----------
        name : str or object
            The new name string (internally cast to str).
        """
        self.__name__ = str(name)

    def setShadowedId(self, ids):
        """Set the shadowed ID parameter.

        Parameters
        ----------
        ids : int
            The shadowed identifier integer.
        """
        self.__shadowedId__ = ids

    def setSize(self, N):
        """Set the sample size for the base matrices A and B.

        Resets all existing generated matrices if the size is updated.

        Parameters
        ----------
        N : int
            The number of rows per base sample. Must be a positive integer.

        Raises
        ------
        AssertionError
            If N is not a positive integer.
        """
        assert isinstance(N, int) and N > 0, "Sample size can only be positive integer"
        if self.size is None:
            self.size = N
        else:
            self.size = N
            self._sample_A = self._sample_B = self._experimentSample = None

    def _mixSamples(self):
        """Construct the design matrix from base matrices A and B.

        Interleaves matrix slices based on the problem dimension and whether 
        second-order indices are required.
        """
        n_vars = self.composedDistribution.getDimension()
        N = self.size
        if not self.__computeSecondOrder__ or n_vars <= 2:
            N_tot = int(N * (2 + n_vars))
            self._experimentSample = ot.Sample(N_tot, n_vars)
            self._experimentSample[:N, :] = self._sample_A[:, :]
            self._experimentSample[N : 2 * N, :] = self._sample_B[:, :]
            for i in range(n_vars):
                self._experimentSample[2 * N + N * i : 2 * N + N * (i + 1), :] = self._sample_A[
                    :, :
                ]
                self._experimentSample[2 * N + N * i : 2 * N + N * (i + 1), i] = self._sample_B[
                    :, i
                ]

        else:
            N_tot = int(N * (2 + 2 * n_vars))
            self._experimentSample = ot.Sample(N_tot, n_vars)
            self._experimentSample[:N, :] = self._sample_A[:, :]
            self._experimentSample[N : 2 * N, :] = self._sample_B[:, :]
            for i in range(n_vars):
                self._experimentSample[2 * N + N * i : 2 * N + N * (i + 1), :] = self._sample_A[
                    :, :
                ]
                self._experimentSample[2 * N + N * i : 2 * N + N * (i + 1), i] = self._sample_B[
                    :, i
                ]
            for i in range(n_vars):
                self._experimentSample[
                    2 * N + N * (n_vars + i) : 2 * N + N * (n_vars + i + 1), :
                ] = self._sample_B[:, :]
                self._experimentSample[
                    2 * N + N * (n_vars + i) : 2 * N + N * (n_vars + i + 1), i
                ] = self._sample_A[:, i]

    def _generateSample(self, **kwargs):
        """Generate base samples A and B.

        Executes the requested generation strategy (MonteCarlo, LHS, or QMC)
        and populates the internal `_sample_A` and `_sample_B` structures.

        Parameters
        ----------
        **kwargs : dict
            Forwarded processing specifications (e.g., 'method', 'sequence').

        Raises
        ------
        ValueError
            If an invalid method type or QMC sequence string is requested.
        """
        distribution = self.composedDistribution
        if "method" in kwargs:
            method = kwargs["method"]
        else:
            method = "MonteCarlo"
        N2 = 2 * self.size
        if method == "MonteCarlo":
            sample = distribution.getSample(N2)
        elif method == "LHS":
            lhsExp = ot.LHSExperiment(distribution, N2, False, True)
            sample = lhsExp.generate()
        elif method == "QMC":
            if "sequence" in kwargs:
                sequence = kwargs["sequence"]
                if sequence == "Faure":
                    seq = ot.FaureSequence()
                elif sequence == "Halton":
                    seq = ot.HaltonSequence()
                elif sequence == "ReverseHalton":
                    seq = ot.ReverseHaltonSequence()
                elif sequence == "Haselgrove":
                    seq = ot.HaselgroveSequence()
                elif sequence == "Sobol":
                    seq = ot.SobolSequence()
                else:
                    raise ValueError("Unknown sequence type")
            else:
                print("sequence undefined for low discrepancy experiment, default: SobolSequence")
                seq = ot.SobolSequence()
            LDExperiment = ot.LowDiscrepancyExperiment(seq, distribution, N2, True)
            LDExperiment.setRandomize(False)
            sample = LDExperiment.generate()
        else:
            raise ValueError("Unknown sampling method")
        self._sample_A = sample[: self.size, :]
        self._sample_B = sample[self.size :, :]
