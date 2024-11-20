# -*- coding: utf-8
__author__ = "Kramer84"
__all__ = ["ModalInterval"]


class ModalInterval:
    """
    A class to represent a modal interval and perform modal interval arithmetic.

    Modal interval arithmetic is an extension of traditional interval arithmetic that identifies intervals
    by the set of predicates satisfied by the real numbers. It can be used to define operations on intervals,
    vectors, and matrices, particularly useful in tolerance analysis.

    Attributes:
        lower (float): The lower bound of the interval.
        upper (float): The upper bound of the interval.

    Methods:
        __add__(other): Adds two modal intervals.
        __sub__(other): Subtracts one modal interval from another.
        __mul__(other): Multiplies two modal intervals.
        __truediv__(other): Divides one modal interval by another.

    Note:
        From the paper by S. Khodaygan & M. R. Movahhedy & M. Saadat Fomani (2010) :
        Tolerance Analysis of Mechanical Assemblies Based on Modal Interval and Small Degrees of Freedom (MI SDOF) Concepts
    """

    def __init__(self, lower, upper):
        """
        Initializes a modal interval with lower and upper bounds.

        Args:
            lower (float): The lower bound of the interval.
            upper (float): The upper bound of the interval.
        """
        self.lower = lower
        self.upper = upper

    def __add__(self, other):
        """
        Adds two modal intervals.

        The addition of two modal intervals [aL, aU] and [bL, bU] is defined as:
        [aL + bL, aU + bU]

        Args:
            other (ModalInterval): The interval to add.

        Returns:
            ModalInterval: The result of the addition.
        """
        return ModalInterval(self.lower + other.lower, self.upper + other.upper)

    def __sub__(self, other):
        """
        Subtracts one modal interval from another.

        The subtraction of two modal intervals [aL, aU] and [bL, bU] is defined as:
        [aL - bU, aU - bL]

        Args:
            other (ModalInterval): The interval to subtract.

        Returns:
            ModalInterval: The result of the subtraction.
        """
        return ModalInterval(self.lower - other.upper, self.upper - other.lower)

    def __mul__(self, other):
        """
        Multiplies two modal intervals.

        The multiplication of two modal intervals [aL, aU] and [bL, bU] is defined by considering
        various cases for the signs of the bounds.

        Args:
            other (ModalInterval): The interval to multiply.

        Returns:
            ModalInterval: The result of the multiplication.
        """
        aL, aU = self.lower, self.upper
        bL, bU = other.lower, other.upper

        if aL >= 0 and aU >= 0 and bL >= 0 and bU >= 0:
            return ModalInterval(aL * bL, aU * bU)
        elif aL >= 0 and aU >= 0 and bL < 0 and bU < 0:
            return ModalInterval(aU * bL, aL * bU)
        elif aL >= 0 and aU >= 0 and bL < 0 and bU >= 0:
            return ModalInterval(aU * bL, aU * bU)
        elif aL >= 0 and aU >= 0 and bL >= 0 and bU < 0:
            return ModalInterval(aL * bU, aL * bL)
        elif aL < 0 and aU < 0 and bL >= 0 and bU >= 0:
            return ModalInterval(aL * bU, aU * bL)
        elif aL < 0 and aU < 0 and bL < 0 and bU < 0:
            return ModalInterval(aU * bU, aL * bL)
        elif aL < 0 and aU < 0 and bL < 0 and bU >= 0:
            return ModalInterval(aL * bU, aL * bL)
        elif aL < 0 and aU < 0 and bL >= 0 and bU < 0:
            return ModalInterval(aU * bL, aU * bU)
        elif aL < 0 and aU >= 0 and bL >= 0 and bU >= 0:
            return ModalInterval(aL * bU, aU * bU)
        elif aL < 0 and aU >= 0 and bL < 0 and bU < 0:
            return ModalInterval(aU * bL, aL * bL)
        elif aL < 0 and aU >= 0 and bL < 0 and bU >= 0:
            return ModalInterval(min(aL * bU, aU * bL), max(aL * bL, aU * bU))
        elif aL < 0 and aU >= 0 and bL >= 0 and bU < 0:
            return ModalInterval(min(aU * bL, aL * bU), max(aU * bU, aL * bL))
        else:
            raise ValueError("Unhandled case for interval multiplication.")

    def __truediv__(self, other):
        """
        Divides one modal interval by another.

        The division of two modal intervals [aL, aU] and [bL, bU] is defined by considering the inverse of the second interval.

        Args:
            other (ModalInterval): The interval to divide by.

        Returns:
            ModalInterval: The result of the division.
        """
        aL, aU = self.lower, self.upper
        bL, bU = other.lower, other.upper

        if bL <= 0 <= bU:
            raise ZeroDivisionError("Division by an interval containing zero is undefined.")

        if aL >= 0 and aU >= 0 and bL > 0 and bU > 0:
            return ModalInterval(aL / bU, aU / bL)
        elif aL >= 0 and aU >= 0 and bL < 0 and bU < 0:
            return ModalInterval(aU / bL, aL / bU)
        elif aL >= 0 and aU >= 0 and bL < 0 and bU > 0:
            return ModalInterval(aU / bL, aU / bU)
        elif aL >= 0 and aU >= 0 and bL > 0 and bU < 0:
            return ModalInterval(aL / bU, aL / bL)
        elif aL < 0 and aU < 0 and bL > 0 and bU > 0:
            return ModalInterval(aL / bL, aU / bU)
        elif aL < 0 and aU < 0 and bL < 0 and bU < 0:
            return ModalInterval(aU / bU, aL / bL)
        elif aL < 0 and aU < 0 and bL < 0 and bU > 0:
            return ModalInterval(aL / bL, aL / bU)
        elif aL < 0 and aU < 0 and bL > 0 and bU < 0:
            return ModalInterval(aU / bU, aU / bL)
        elif aL < 0 and aU >= 0 and bL > 0 and bU > 0:
            return ModalInterval(aL / bL, aU / bL)
        elif aL < 0 and aU >= 0 and bL < 0 and bU < 0:
            return ModalInterval(aU / bU, aL / bU)
        elif aL < 0 and aU >= 0 and bL < 0 and bU > 0:
            return ModalInterval(min(aL / bL, aU / bU), max(aL / bU, aU / bL))
        elif aL < 0 and aU >= 0 and bL > 0 and bU < 0:
            return ModalInterval(min(aU / bU, aL / bL), max(aU / bL, aL / bU))
        else:
            raise ValueError("Unhandled case for interval division.")

    def __repr__(self):
        """
        Represents the modal interval as a string.

        Returns:
            str: The string representation of the modal interval.
        """
        return f"[{self.lower}, {self.upper}]"
