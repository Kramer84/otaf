__author__ = "Kramer84"
__version__ = "0.1"
__date__ = "21.05.24"

__all__ = ["SobolIndicesExperimentWithComposedDistribution"]

import openturns as ot


class SobolIndicesExperimentWithComposedDistribution(ot.SobolIndicesExperiment):
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
        """Generates and returns the final mixture matrix.

        Keyword Arguments
        -----------------
        method : str
            Can be : 'MonteCarlo', 'LHS', 'QMC'
        sequence : str
            Only if using QMC
            Can be : 'Faure', 'Halton', 'ReverseHalton', 'Haselgrove', 'Sobol'
        """
        assert (self.composedDistribution is not None) and (
            self.size is not None
        ), "Please intialize sample size and composed distribution"
        self._generateSample(**kwargs)

        self._mixSamples()
        self._experimentSample.setDescription(self.composedDistribution.getDescription())
        return self._experimentSample

    def generateWithWeights(self, **kwargs):
        """Not implemented, for coherence with openturns library"""
        pass

    def getClassName(self):
        """Returns the name of the class."""
        return self.__class__.__name__

    def getId(self):
        """Returns the ID of the object."""
        return id(self)

    def getName(self):
        """Returns the name of the object."""
        return self.__name__

    def getShadowedId(self):
        """Returns the shadowed ID of the object."""
        return self.__shadowedId__

    def getSize(self):
        """Returns the size of the generated mixture matrix."""
        if self._experimentSample is None:
            return 0
        else:
            return len(self._experimentSample)

    def getVisibility(self):
        """Returns the visibility"""
        return self.__visibility__

    def hasName(self):
        """Returns if the object has a name"""
        if len(self.__name__) == 0 or self.__name__ is None:
            return False
        else:
            return True

    def hasUniformWeights(self):
        """Not implemented, for coherence with openturns library"""
        return None

    def hasVisibleName(self):
        """Returns if yes or not the name is visible"""
        if self.__name__ == "Unnamed" or len(self.__name__) == 0:
            return False
        else:
            return True

    def setComposedDistribution(self, composedDistribution):
        """Sets the composed distribution."""
        self.composedDistribution = composedDistribution

    def setName(self, name):
        """Sets the name of the object"""
        self.__name__ = str(name)

    def setShadowedId(self, ids):
        """Sets the shadowed ID of the object."""
        self.__shadowedId__ = ids

    def setSize(self, N):
        """Sets the size of the samples A and B."""
        assert isinstance(N, int) and N > 0, "Sample size can only be positive integer"
        if self.size is None:
            self.size = N
        else:
            self.size = N
            self._sample_A = self._sample_B = self._experimentSample = None

    def _mixSamples(self):
        """Mixes the samples together for Sobol analysis."""
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
        """Generation of two samples A and B using diverse methods"""
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
