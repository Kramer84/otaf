__all__ = ["SobolKarhunenLoeveFieldSensitivityAlgorithm"]


import openturns as ot
from collections.abc import Iterable, Sequence
from collections import UserList
from copy import copy, deepcopy
from numbers import Complex, Integral, Real, Rational, Number
from math import isnan
import re


def all_same(items=None):
    # Checks if all items of a list are the same
    return all(x == items[0] for x in items)


def atLeastList(elem=None):
    if isinstance(elem, (Iterable, Sequence, list)) and not isinstance(elem, (str, bytes)):
        return list(elem)
    else:
        return [elem]


def list_(*args):
    return list(args)


def zip_(*args):
    return map(list_, *args)


def checkIfNanInSample(sample):
    return isnan(sum(sample.computeMean()))


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def noLogInNotebook(func):
    def inner(*args, **kwargs):
        if isnotebook():
            ot.Log.Show(ot.Log.NONE)
        results = func(*args, **kwargs)
        if isnotebook():
            ot.Log.Show(ot.Log.DEFAULT)
        return results

    return inner


class SobolKarhunenLoeveFieldSensitivityAlgorithm(object):
    """Pure opentTURNS implementation of the sobol indices algorithm
    in the case where the design of the experiment was generated
    using a field function that has been wrapped using the
    KarhunenLoeveGeneralizedWrapper.

    Note
    ----
    There should be no difference at all with the real
    SaltelliSensitivityAlgorithm implementation, only that the orihinal
    implementation checks the input and output desgin's dimension and
    raises an error if the dimensions mismatch.
    """

    def __init__(
        self,
        inputDesign=None,
        outputDesign=None,
        N=0,
        estimator=ot.SaltelliSensitivityAlgorithm(),
        computeSecondOrder=False,
        verbose=0,
    ):
        self.inputDesign = inputDesign
        self.outputDesign = atLeastList(outputDesign)
        self.N = int(N)
        self.size = None
        self.__nOutputs__ = 0
        self.computeSecondOrder = computeSecondOrder
        self.__verbosity__ = verbose
        self.__visibility__ = True
        self.__shadowedId__ = 0
        self.__name__ = "Unnamed"
        self.inputDescription = None
        self.__nSobolIndices__ = None
        self.__Meshes__ = list()
        self.__Classes__ = list()
        self.__BootstrapSize__ = None
        self.flatOutputDesign = list()
        self.__centeredOutputDesign__ = list()
        self.__results__ = list()
        self.estimator = estimator
        if len(self.outputDesign) > 0 and self.outputDesign[0] is not None:
            assert all_same([len(self.outputDesign[i]) for i in range(len(self.outputDesign))])
        if inputDesign is not None and outputDesign is not None:
            try:
                assert isinstance(inputDesign, ot.Sample), "The input design can only be a Sample"
                assert any(
                    [
                        isinstance(outputDesign[i], (ot.Sample, ot.ProcessSample))
                        for i in range(len(outputDesign))
                    ]
                )
            except AssertionError:
                print("\n\n\n\n\n\n\nThe error\n\n\n\n\n\n\n")
                return None
        self.__setDefaultState__()

    def __repr__(self):
        str0 = "SobolKarhunenLoeveFieldSensitivityAlgorithm"
        str1 = "with estimator : {}".format(self.estimator.__class__.__name__)
        str2 = "input dimension : {}".format(
            str(self.inputDesign.getDimension()) if self.inputDesign is not None else "_"
        )
        str3 = "output size : {}".format(self.__nOutputs__)
        str4 = "compute second order : {}".format(self.computeSecondOrder)
        return ", ".join([str0, str1, str2, str3, str4])

    def DrawCorrelationCoefficients(self, *args):
        """Not implemented; here to mimic openTURNS methods."""
        self.__fastResultCheck__()
        print("Drawing is not yet implemented")
        raise NotImplementedError

    def DrawImportanceFactors(self, *args):
        """Not implemented; here to mimic openTURNS methods."""
        self.__fastResultCheck__()
        print("Drawing is not yet implemented")
        raise NotImplementedError

    def DrawSobolIndices(self, *args):
        """Not implemented; here to mimic openTURNS methods."""
        self.__fastResultCheck__()
        print("Drawing is not yet implemented")
        raise NotImplementedError

    def draw(self, *args):
        """Not implemented; here to mimic openTURNS methods."""
        self.__fastResultCheck__()
        print("Drawing is not yet implemented")
        raise NotImplementedError

    @noLogInNotebook
    def getAggregatedFirstOrderIndices(self):
        """Returns the agrregated first order indices

        Returns
        -------
        aggFO_indices : list of ot.Point
        """
        self.__fastResultCheck__()
        aggFO_indices = list()
        for i in range(self.__nOutputs__):
            aggFO_indices.append(
                ot.PointWithDescription(self.__results__[i].getAggregatedFirstOrderIndices())
            )
            aggFO_indices[i].setName("Sobol_" + self.outputDesign[i].getName())
            aggFO_indices[i].setDescription(
                [
                    self.outputDesign[i].getName() + "_" + self.inputDescription[j]
                    for j in range(self.__nSobolIndices__)
                ]
            )
        return aggFO_indices

    @noLogInNotebook
    def getAggregatedTotalOrderIndices(self):
        """Returns the agrregated total order indices

        Returns
        -------
        aggFO_indices : list of ot.Point
        """
        self.__fastResultCheck__()
        aggTO_indices = list()
        for i in range(self.__nOutputs__):
            aggTO_indices.append(
                ot.PointWithDescription(self.__results__[i].getAggregatedTotalOrderIndices())
            )
            aggTO_indices[i].setName("Sobol_" + self.outputDesign[i].getName())
            aggTO_indices[i].setDescription(
                [
                    self.outputDesign[i].getName() + "_" + self.inputDescription[j]
                    for j in range(self.__nSobolIndices__)
                ]
            )
        return aggTO_indices

    def getBootstrapSize(self):
        """Returns the bootstrap size"""
        return self.__BootstrapSize__

    def getClassName(self):
        """Returns the class name"""
        return self.__class__.__name__

    def getConfidenceLevel(self):
        """Returns the confidence level

        Returns
        -------
        ConfidenceLevel : float
        """
        return self.ConfidenceLevel

    @noLogInNotebook
    def getFirstOrderIndices(self):
        """Returns the first order indices

        Returns
        -------
        FO_indices : list of ot.Point
        """
        self.__fastResultCheck__()
        FO_indices = list()
        nMarginals = [
            self.__centeredOutputDesign__[i].getDimension() for i in range(self.__nOutputs__)
        ]
        for i in range(self.__nOutputs__):
            FO_point_list = [
                self.__results__[i].getFirstOrderIndices(j) for j in range(nMarginals[i])
            ]
            FO_point_list = list(zip_(*FO_point_list))
            FO_indices.append(
                [
                    self.__toBaseDataFormat__(ot.Point(FO_point_list[k]), i)
                    for k in range(self.__nSobolIndices__)
                ]
            )
            [
                FO_indices[i][k].setName(
                    "Sobol_" + self.outputDesign[i].getName() + "_" + self.inputDescription[k]
                )
                for k in range(self.__nSobolIndices__)
            ]
        return FO_indices

    @noLogInNotebook
    def getFirstOrderIndicesDistribution(self):
        """Returns the first order indices distribution

        Returns
        -------
        FO_indices_distribution : list of ot.Distribution
        """
        self.__fastResultCheck__()
        FO_indices_distribution = list()
        for i in range(self.__nOutputs__):
            FO_indices_distribution.append(self.__results__[i].getFirstOrderIndicesDistribution())
            FO_indices_distribution[i] = self.__toBaseDataFormat__(FO_indices_distribution[i], i)
            FO_indices_distribution[i].setName(self.outputDesign[i].getName())
            FO_indices_distribution[i].setDescription(
                [
                    self.outputDesign[i].getName() + "_" + self.inputDescription[j]
                    for j in range(self.__nSobolIndices__)
                ]
            )
        return FO_indices_distribution

    @noLogInNotebook
    def getFirstOrderIndicesInterval(self):
        self.__fastResultCheck__()
        FO_indices_interval = list()
        for i in range(self.__nOutputs__):
            FO_indices_interval.append(self.__results__[i].getFirstOrderIndicesInterval())
        [
            FO_indices_interval[i].setName("Bounds_Sobol_" + self.outputDesign[i].getName())
            for i in range(self.__nOutputs__)
        ]
        return FO_indices_interval

    def getId(self):
        return id(self)

    def getName(self):
        return self.__name__

    @noLogInNotebook
    def getSecondOrderIndices(self):
        if self.computeSecondOrder == True:
            self.__fastResultCheck__()
            SO_indices = list()
            nMarginals = [
                self.__centeredOutputDesign__[i].getDimension() for i in range(self.__nOutputs__)
            ]
            for i in range(self.__nOutputs__):
                SO_point_list = list()
                for j in range(nMarginals[i]):
                    SO_point_list.append(self.__results__[i].getSecondOrderIndices(j))
                print("This makes a problem:", SO_point_list)
            return SO_point_list
        else:
            print("The second order indices flag is not set to true.")
            print("Have you passed the right sample to make this calculus?")
            return None

    def getShadowedId(self):
        return self.__shadowedId__

    @noLogInNotebook
    def getTotalOrderIndices(self):
        self.__fastResultCheck__()
        TO_indices = list()
        nMarginals = [
            self.__centeredOutputDesign__[i].getDimension() for i in range(self.__nOutputs__)
        ]
        for i in range(self.__nOutputs__):
            TO_point_list = list()
            for j in range(nMarginals[i]):
                TO_point_list.append(self.__results__[i].getTotalOrderIndices(j))
            TO_point_list = list(zip_(*TO_point_list))
            TO_indices.append(
                [
                    self.__toBaseDataFormat__(ot.Point(TO_point_list[k]), i)
                    for k in range(self.__nSobolIndices__)
                ]
            )
            [
                TO_indices[i][k].setName(
                    "TotalOrderSobol_"
                    + self.outputDesign[i].getName()
                    + "_"
                    + self.inputDescription[k]
                )
                for k in range(self.__nSobolIndices__)
            ]
        return TO_indices

    @noLogInNotebook
    def getTotalOrderIndicesDistribution(self):
        self.__fastResultCheck__()
        TO_indices_distribution = list()
        for i in range(self.__nOutputs__):
            TO_indices_distribution.append(self.__results__[i].getTotalOrderIndicesDistribution())
            TO_indices_distribution[i] = self.__toBaseDataFormat__(TO_indices_distribution[i], i)
            TO_indices_distribution[i].setName(self.outputDesign[i].getName())
            TO_indices_distribution[i].setDescription(
                [
                    self.outputDesign[i].getName() + "_" + self.inputDescription[j]
                    for j in range(self.__nSobolIndices__)
                ]
            )
        return TO_indices_distribution

    @noLogInNotebook
    def getTotalOrderIndicesInterval(self):
        self.__fastResultCheck__()
        TO_indices_interval = list()
        for i in range(self.__nOutputs__):
            TO_indices_interval.append(self.__results__[i].getTotalOrderIndicesInterval())
        [
            TO_indices_interval[i].setName(
                "BoundsTotalOrderSobol_" + self.outputDesign[i].getName()
            )
            for i in range(self.__nOutputs__)
        ]
        return TO_indices_interval

    def getUseAsymptoticDistribution(self):
        self.__fastResultCheck__()
        useAsymptotic = list()
        for i in range(self.__nOutputs__):
            useAsymptotic.append(self.__results__[i].getUseAsymptoticDistribution())
            useAsymptotic[i] = self.__toBaseDataFormat__(useAsymptotic[i], i)
        return useAsymptotic

    def getVisibility(self):
        return self.__visibility__

    def hasName(self):
        if self.__name__ != "Unnamed" and len(self.__name__) > 0:
            return True
        else:
            return False

    def setBootstrapSize(self, bootstrapSize):
        self.__BootstrapSize__ = bootstrapSize

    def setConfidenceLevel(self, confidenceLevel):
        self.ConfidenceLevel = confidenceLevel

    def setDesign(self, inputDesign=None, outputDesign=None, N=0):
        outputDesign = atLeastList(outputDesign)
        assert all_same([len(outputDesign[i]) for i in range(len(outputDesign))])
        assert isinstance(N, (int, Integral)) and N >= 0
        assert isinstance(inputDesign, ot.Sample), "The input design can only be a Sample"
        assert any(
            [
                isinstance(outputDesign[i], (ot.Sample, ot.ProcessSample))
                for i in range(len(outputDesign))
            ]
        )
        self.inputDesign = inputDesign
        self.outputDesign = atLeastList(outputDesign)
        self.N = int(N)
        if self.outputDesign is not None and self.N > 0:
            self.__setDefaultState__()

    def setEstimator(self, estimator):
        self.estimator = estimator

    def setName(self, name):
        self.__name__ = name

    def setShadowedId(self, shadowedId):
        self.__shadowedId__ = shadowedId

    def setUseAsymptoticDistribution(self, useAsymptoticList=False):
        if isinstance(useAsymptoticList, bool):
            asymptoticList = [useAsymptoticList] * self.__nOutputs__
        elif isinstance(useAsymptoticList, (Sequence, Iterable)):
            assert len(useAsymptoticList) == self.__nOutputs__
            asymptoticList = useAsymptoticList
        else:
            raise NotImplementedError
        self.__fastResultCheck__()
        try:
            for i in range(self.__nOutputs__):
                self.__results__[i].setUseAsymptoticDistribution(asymptoticList[i])
        except TypeError:
            print("Check the function arguments. Must be of type Bool or list of Bools")
            raise TypeError

    def setVisibility(self, visible):
        self.__visibility__ = visible

    def setComputeSecondOrder(self, choice):
        self.computeSecondOrder = choice
        self.__setDefaultState__()

    def __setDefaultState__(self):
        try:
            if self.outputDesign is not None and self.N > 0:
                self.size = len(self.outputDesign[0])
                if self.__verbosity__ > 0:
                    print("size initialized", self.size)
                self.__nOutputs__ = len(self.outputDesign)
                if self.computeSecondOrder == True:
                    self.__nSobolIndices__ = int((int(self.size / self.N) - 2) / 2)
                    try:
                        assert (int(self.size / self.N) - 2) % 2 == 0
                    except AssertionError:
                        print(MSG_1)
                else:
                    self.__nSobolIndices__ = int(int(self.size / self.N) - 2)
                    print(MSG_2)
                self.__getDataOutputDesign__()
                self.__flattenOutputDesign__()
                self.__centerOutputDesign__()
                if self.__verbosity__ > 0:
                    self.__confirmationMessage__()
                self.__setDefaultName__()
            else:
                pass
        except Exception as e:
            print("Something unexpected happened")
            raise e

    def __getDataOutputDesign__(self):
        self.__Meshes__.clear()
        self.__Classes__.clear()
        for output in self.outputDesign:
            self.__Classes__.append(output.__class__)
            try:
                self.__Meshes__.append(output.getMesh())
            except AttributeError:
                self.__Meshes__.append(None)

    def __flattenOutputDesign__(self):
        self.flatOutputDesign.clear()
        for outputDes in self.outputDesign:
            if isinstance(outputDes, (ot.Point, ot.Sample)):
                self.flatOutputDesign.append(outputDes)
            if isinstance(outputDes, ot.ProcessSample):
                sample, mesh = self.__splitProcessSample__(outputDes)
                self.flatOutputDesign.append(sample)

    def __splitProcessSample__(self, processSample):
        """Function to split a process sample into a 1D sample and a mesh."""
        sample = ot.Sample(self.size, processSample.getMesh().getVerticesNumber())
        for i in range(self.size):
            sample[i] = processSample[i].asPoint()
        return sample, processSample.getMesh()

    def __centerOutputDesign__(self):
        design_cpy = [self.flatOutputDesign[i][:] for i in range(self.__nOutputs__)]
        self.__centeredOutputDesign__.clear()
        for design_elem in design_cpy:
            if isinstance(design_elem, ot.Point):
                mean = sum(design_elem) / design_elem.getDimension()
                design_elem -= [mean] * design_elem.getDimension()
                self.__centeredOutputDesign__.append(deepcopy(design_elem))
            elif isinstance(design_elem, ot.Sample):
                mean = design_elem.computeMean()
                if self.__verbosity__ > 0:
                    print("Means is\n", mean)
                    print(
                        "design_elem size, dim", design_elem.getSize(), design_elem.getDimension()
                    )
                design_elem -= mean
                self.__centeredOutputDesign__.append(deepcopy(design_elem))
            else:
                raise NotImplementedError

    def __confirmationMessage__(self):
        def getDim(pointOrSample):
            if isinstance(pointOrSample, ot.Point):
                return 1
            elif isinstance(pointOrSample, ot.Sample):
                return pointOrSample.getDimension()
            else:
                return pointOrSample.getDimension()

        dims = ", ".join([str(getDim(self.flatOutputDesign[i])) for i in range(self.__nOutputs__)])
        print(
            "There are {} indices to get for {} outputs with dimensions {} each.".format(
                self.__nSobolIndices__, self.__nOutputs__, dims
            )
        )

    def __setDefaultName__(self):
        if self.inputDesign is None:
            desc = ot.Description.BuildDefault(self.__nSobolIndices__, "X")
            self.inputDescription = desc
        elif all_same(self.inputDesign.getDescription()) == True:
            desc = ot.Description.BuildDefault(self.__nSobolIndices__, "X")
            self.inputDescription = desc
        elif all_same(self.inputDesign.getDescription()) == False:
            inputDescription = self.inputDesign.getDescription()
            # if self.__verbosity__ > 3:
            #    print("Description all same?", inputDescription)
            # SobolIndicesName = []
            # inputWOutLastChar = [
            #    inputDescription[i][
            #        : re.search(r"_\d+$", inputDescription[i]).span()[0]
            #    ]
            #    for i in range(len(inputDescription))
            # ]
            # SobolIndicesName = []
            # for x in inputWOutLastChar:
            #    if x not in SobolIndicesName:
            #        SobolIndicesName.append(x)
            # if self.__verbosity__ > 3:
            #    print("SobolIndicesName", SobolIndicesName)
            self.inputDescription = inputDescription
        if self.__verbosity__ > 1:
            print("Input Description is,", self.inputDescription)

    def __fastResultCheck__(self):
        if not len(self.__results__) > 0:
            try:
                self.__solve__()
            except AssertionError:
                print("You need samples to work on. Check doc")
                raise AssertionError

    def __solve__(self):
        assert self.estimator is not None, "First initialize the estimator (Jansen, Saltelli, etc.)"
        assert len(self.outputDesign) > 0, "You have to pass a Sample to work on"
        assert self.outputDesign[0] is not None, "You have to pass a Sample to work on"
        print("Solving...")
        print(" size of samples: ", self.size)
        print(" number of indices to get", self.__nSobolIndices__)
        dummyInputSample = ot.Sample(self.size, self.__nSobolIndices__)
        dummyInputSample.setDescription(self.inputDescription)
        self.__results__ = list()
        outputDesigns = self.__centeredOutputDesign__
        for i in range(len(outputDesigns)):
            estimator = self.estimator.__class__()
            _input = deepcopy(dummyInputSample)
            self.__results__.append(estimator)
            if not checkIfNanInSample(outputDesigns[i]):
                self.__results__[i].setDesign(_input, outputDesigns[i], self.N)
                self.__results__[i].setName(self.inputDescription[i])
            else:
                print("One of your outputs at idx {} contains Nans".format(i))
                print("Please recheck your ouput samples and correct them")
                raise TypeError

    def __toBaseDataFormat__(self, data, idx):
        mesh = self.__Meshes__[idx]
        if isinstance(data, ot.Point):
            if mesh is not None:
                dataBaseFormat = ot.Field(mesh, [[dat] for dat in data])
                return dataBaseFormat
            else:
                return data
        elif isinstance(data, ot.Interval):
            lowBounds = data.getLowerBound()
            upperBounds = data.getUpperBound()
            if mesh is not None:
                lowBoundsBaseFormat = ot.Field(mesh, [[bnd] for bnd in lowBounds])
                upperBoundsBaseFormat = ot.Field(mesh, [[bnd] for bnd in upperBounds])
                return lowBoundsBaseFormat, upperBoundsBaseFormat
            else:
                return data
        elif isinstance(data, ot.Distribution):
            print("Cannot convert distribution to field, dimensional correlation lost.")
            return data
        elif isinstance(data, (bool, int, float)):
            return data
        else:
            raise NotImplementedError


MSG_1 = (
    "The outputDesign you have passed does not satisfy ",
    "the minimum requirements to do the sensitivity analysis.\n",
    "The total sample size should be : tot = N * (2 + d) = 2*N + 2*d*N\n",
    "In this case : tot-2*N = 2*d*N MUST be divisible by two",
)


MSG_2 = (
    "Warning : Always pass the 'computeSecondOrder' argument, ",
    "if you also pass the data to compute it.\n",
    "Otherwise, the behavior will be unreliable",
)
