# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = [
    "condition_lambda_sample",
    "validate_and_extract_indices",
    "condition_sample_array",
    "find_best_worst_quantile",
    "generateLHSExperiment",
    "generate_random_permutations_with_sampling",
    "generate_scaled_permutations",
    "generate_imprecise_probabilistic_samples",
    "generate_and_transform_sequence",
    "compose_defects_with_lambdas",
]

import itertools
import logging
import copy
import re
import numpy as np
import sympy as sp
from scipy.optimize import Bounds, LinearConstraint
import openturns as ot
from beartype import beartype
from beartype.typing import Dict, List, Tuple, Union, Callable, Optional
from functools import partial, lru_cache
from collections.abc import Iterable

import torch


@beartype
def condition_lambda_sample(sample: ot.Sample, squared_sum: bool = False) -> ot.Sample:
    """Condition lambda parameters to sum to 1 for each feature.

    Args:
        sample (ot.Sample): Lambda parameters with each feature ending in an integer sequence.
        squared_sum (bool, optional): Whether to square the resulting values. Defaults to False.

    Returns:
        ot.Sample: Conditioned lambda parameters.

    Note
    ----
    THis function can only be used if all the features are plane./ If thezy are cylindrical
    the constraints are aboluetly different. To use both u need to make a comlpex function
    that checks the feature type for each variable and uses the right constraint.
    """
    logging.info("[condition_lambda_sample] Started. Checking input sample.")

    deviation_symbols = list(sample.getDescription())

    # Validate the description and extract feature indices
    feature_indices = validate_and_extract_indices(deviation_symbols)

    sample_array = np.array(sample)

    conditioned_array = condition_sample_array(sample_array, feature_indices, squared_sum)

    conditioned_sample = ot.Sample(conditioned_array)
    conditioned_sample.setDescription(deviation_symbols)

    logging.info("[condition_lambda_sample] Completed. Conditioned sample generated.")

    return conditioned_sample


@beartype
def validate_and_extract_indices(description: list) -> list[int]:
    """Validate the feature descriptions and extract feature indices.

    Args:
        description (list[str]): List of feature descriptions.

    Returns:
        list[int]: Extracted feature indices.

    Raises:
        ValueError: If the description is missing or invalid.
    """
    # Check if the description exists
    if not description:
        raise ValueError("Description missing. Must be provided.")

    logging.info("[validate_and_extract_indices] Extracting feature indices.")

    # Extract feature indices and ensure each feature ends with an integer
    feature_indices = []
    for symbol in description:
        match = re.search(r"\d+$", str(symbol))
        if match:
            feature_indices.append(int(match.group()))
        else:
            raise ValueError(
                f"Invalid description for feature '{symbol}'. Each feature should end with integers."
            )

    return feature_indices


@beartype
def condition_sample_array(
    sample_array: np.ndarray, feature_indices: list[int], squared_sum: bool
) -> np.ndarray:
    """Condition the sample array by normalizing the features.

    Args:
        sample_array (np.ndarray): The input sample array.
        feature_indices (list[int]): Indices of the features.
        squared_sum (bool): Whether to square the resulting values.

    Returns:
        np.ndarray: Conditioned sample array.
    """
    logging.info("[condition_sample_array] Conditioning sample.")
    try:
        conditioned_array = sample_array.copy()

        for i, feature_id in enumerate(feature_indices):
            same_feature_indices = [j for j, idx in enumerate(feature_indices) if idx == feature_id]
            conditioned_array[..., i] /= np.sum(sample_array[..., same_feature_indices], axis=-1)

        if squared_sum:
            conditioned_array = np.sqrt(conditioned_array)

        return conditioned_array
    except Exception as e:
        # Log the error and print all arguments passed to the function
        logging.error(f"Error in condition_sample_array: {e}")
        print(
            f"Error encountered in condition_sample_array:\n"
            f"sample_array: {sample_array}\n"
            f"feature_indices: {feature_indices}\n"
            f"squared_sum: {squared_sum}"
        )
        raise  # Re-raise the exception after logging and printing


def find_best_worst_quantile(parameters, results, quantile):
    """Find the best and worst performing observations based on a given quantile.

    This function calculates the best and worst performing observations from a given set of parameters
    and corresponding results, based on the specified quantile.

    Args:
        parameters (numpy.ndarray): Array containing parameter values.
        results (numpy.ndarray): Array containing corresponding result values.
        quantile (float): Desired quantile value, must be in the range [0, 1].

    Returns:
        tuple: A tuple containing two tuples:
            - The first tuple contains the parameters and results of the best performing observations.
            - The second tuple contains the parameters and results of the worst performing observations.
    """
    pop_size = len(results)
    pop_quantile = int(pop_size * quantile)

    # Sort the results array in ascending order
    sorted_result_args = np.argsort(results)

    # Select the best performing observations
    best_params = parameters[sorted_result_args[:pop_quantile]]
    best_res = results[sorted_result_args[:pop_quantile]]

    # Select the worst performing observations
    worst_params = parameters[sorted_result_args[-pop_quantile:]]
    worst_res = results[sorted_result_args[-pop_quantile:]]

    return (best_params, best_res), (worst_params, worst_res)


def generateLHSExperiment(composed_distribution, N, SEED=999, T0=10, c=0.95, iMax=2000):
    ot.RandomGenerator.SetSeed(SEED)
    LHSExperimentGenerator = ot.LHSExperiment(composed_distribution, N)
    # Defining space fillings
    spaceFilling = ot.SpaceFillingC2()
    # Geometric profile
    geomProfile = ot.GeometricProfile(T0, c, iMax)
    # Simulated Annealing LHS with geometric temperature profile, C2 optimization
    optimalLHSAlgorithm = ot.SimulatedAnnealingLHS(
        LHSExperimentGenerator, spaceFilling, geomProfile
    )

    optimalDesign = optimalLHSAlgorithm.generate()
    return optimalDesign


def generate_random_permutations_with_sampling(subgroup_sizes, num_samples=None, seed=None):
    """
    Generate a random subset of permutations with both positive and negative sign flips,
    based on the specified number of samples (`num_samples`), to mitigate the curse of dimensionality.

    Each subgroup in `subgroup_sizes` represents a set of one-hot encoded vectors (e.g., for size 3:
    [1, 0, 0], [0, 1, 0], [0, 0, 1]). The function generates a random subset of valid permutations
    where each subgroup has exactly one non-zero element. Additionally, each non-zero element
    is randomly assigned either a +1 or -1 value.

    Parameters:
    ----------
    subgroup_sizes : list of int
        A list where each entry corresponds to the size of a subgroup (the number of possible one-hot
        vectors for that group). For example, [3, 4] would mean one group of size 3 and another group of size 4.

    num_samples : int, optional
        The number of random permutations to generate. If `num_samples` is not provided or exceeds
        the total number of possible permutations, the function will generate all possible permutations.
        Each permutation accounts for both structural configurations and sign flips (+1 or -1).

    seed : int, optional
        A seed for the random number generator to ensure reproducibility of results.

    Returns:
    -------
    numpy.ndarray
        A 2D array where each row is a randomly generated permutation, and each column corresponds
        to an element in the concatenated vector of the one-hot encoded subgroups, including the applied
        sign flips (+1 or -1). The number of rows is either `num_samples` or the total number of
        possible permutations if `num_samples` exceeds the possible unique permutations.

    Example:
    --------
    >>> subgroup_sizes = [2, 2]
    >>> num_samples = 4
    >>> generate_random_permutations_with_sampling(subgroup_sizes, num_samples, seed=42)
    array([[ 0,  1, -1,  0],
           [ 1,  0, -1,  0],
           [-1,  0,  0,  1],
           [ 0,  1,  0, -1]])

    Notes:
    ------
    - The function addresses the curse of dimensionality by limiting the number of permutations to
      `num_samples`, if provided.
    - If `num_samples` exceeds the total number of possible permutations, the function generates
      all possible unique permutations.
    - The `seed` parameter ensures reproducibility for random sampling.
    """

    if seed is not None:
        np.random.seed(seed)

    # Generate all valid subgroups for each size (+1 for positive permutations)
    subgroups = [np.eye(size, dtype=int) for size in subgroup_sizes]

    # Calculate total number of possible structural configurations and sign flips
    num_structural_configs = np.prod(subgroup_sizes)
    num_sign_flips = 2 ** len(subgroup_sizes)

    # Total number of possible permutations
    total_permutations = num_structural_configs * num_sign_flips

    # Ensure num_samples does not exceed total_permutations
    if num_samples is None or num_samples > total_permutations:
        num_samples = total_permutations  # Cap num_samples to total possible permutations

    # Generate structural configurations and randomly sample based on num_samples
    all_indices = np.indices(subgroup_sizes).reshape(len(subgroup_sizes), -1).T

    # We can't generate more structural samples than there are configurations
    structural_sample_size = min(len(all_indices), num_samples)

    sampled_indices = all_indices[
        np.random.choice(all_indices.shape[0], structural_sample_size, replace=False)
    ]

    # Create the sampled structural permutations
    sampled_permutations = np.array(
        [
            np.concatenate([subgroups[i][idx] for i, idx in enumerate(sample)])
            for sample in sampled_indices
        ]
    )

    # Randomly sample sign flips for each structural configuration
    sign_flips = np.random.choice([1, -1], size=(structural_sample_size, len(subgroup_sizes)))

    # Apply the sign flips directly to the corresponding subgroup regions
    start = 0
    for i, size in enumerate(subgroup_sizes):
        sampled_permutations[:, start : start + size] *= sign_flips[:, i].reshape(-1, 1)
        start += size

    return sampled_permutations


def generate_scaled_permutations(subgroup_sizes, scaling_factors, num_samples=None, seed=None):
    """
    Generate scaled random permutations with both positive and negative sign flips,
    based on the specified number of samples (`num_samples`). The resulting permutations
    are scaled by the given scaling factors.

    Parameters:
    ----------
    subgroup_sizes : list of int
        A list where each entry corresponds to the size of a subgroup (the number of possible
        one-hot vectors for that group). For example, [3, 4] would mean one group of size 3
        and another group of size 4.

    scaling_factors : list or numpy array of floats
        A list or numpy array of the same length as the total number of elements in the permutation
        vector, where each entry corresponds to a scaling factor for the corresponding permutation element.

    num_samples : int, optional
        The number of random permutations to generate. If `num_samples` is not provided or exceeds
        the total number of possible permutations, the function will generate all possible permutations.
        Each permutation accounts for both structural configurations and sign flips (+1 or -1).

    seed : int, optional
        A seed for the random number generator to ensure reproducibility of results.

    Returns:
    -------
    numpy.ndarray
        A 2D array where each row is a randomly generated permutation, and each column corresponds
        to an element in the concatenated vector of the one-hot encoded subgroups, scaled by the corresponding
        value in `scaling_factors`.
    """

    # Generate the random permutations using the provided function
    permutations = generate_random_permutations_with_sampling(
        subgroup_sizes, num_samples=num_samples, seed=seed
    )

    # Ensure the scaling_factors array has the correct length
    total_size = np.sum(subgroup_sizes)
    if len(scaling_factors) != total_size:
        raise ValueError(
            f"Length of scaling_factors ({len(scaling_factors)}) must match the total number of elements ({total_size}) in the permutation vector."
        )

    # Scale the generated permutations by the scaling_factors
    scaled_permutations = permutations * scaling_factors

    return scaled_permutations


def generate_imprecise_probabilistic_samples(
    subgroup_sizes, num_samples=None, seed=None, discretization=4
):
    """
    Generate a random subset of valid permutations representing the imprecise probabilistic contributions
    of random variables in a constrained space.

    Each subgroup in `subgroup_sizes` represents a set of variables that contribute to a probabilistic model.
    The sum of the elements in each subgroup is constrained to 1, and the function samples random
    permutations from the imprecise space of possible contributions for these subgroups.

    Parameters:
    ----------
    subgroup_sizes : list of int
        A list where each entry corresponds to the size of a subgroup (the number of variables in that group).
        For each subgroup, the sum of its elements is constrained to 1.

    num_samples : int, optional
        The number of random samples to generate. If `num_samples` is -1, the function will compute all
        valid permutations. Otherwise, it will randomly sample from the space of possible contributions.

    seed : int, optional
        A seed for the random number generator to ensure reproducibility of results.

    discretization : int, optional
        The number of steps to discretize the space between [0, 1]. Higher values allow for finer
        granularity in the space of possible contributions.

    Returns:
    -------
    Generator
        A generator yielding valid samples of probabilistic contributions for the subgroups,
        representing the imprecise space of random variables. Each sample is a 1D array where
        each element belongs to a specific subgroup, and the sum of the elements in each subgroup is 1.

    Example:
    --------
    >>> subgroup_sizes = [3, 4]
    >>> num_samples = 5
    >>> generate_imprecise_probabilistic_samples(subgroup_sizes, num_samples=num_samples, seed=42)
    <generator object generate_imprecise_probabilistic_samples at 0x...>

    Example usage with printing:
    >>> for sample in generate_imprecise_probabilistic_samples(subgroup_sizes, num_samples=num_samples, seed=42):
    >>>     print(sample)

    Notes:
    ------
    - This function is designed to explore the space of imprecise probabilistic contributions of random variables.
    - The generator produces samples lazily to reduce memory consumption, allowing it to handle larger
      models efficiently.
    - If `num_samples` is -1, the function will generate all possible permutations lazily, so it won't
      store all permutations in memory.
    - The `discretization` parameter controls how finely the space between [0, 1] is divided for the
      contribution values in each subgroup. Higher values give finer precision but may increase computation time.
    """

    if seed is not None:
        np.random.seed(seed)

    # Generate all valid subgroups for each size using discretization (but lazily)
    def valid_subgroup_permutations(subgroup_size):
        # Generate discretized values between 0 and 1 with sum = 1
        discretized_values = np.linspace(0, 1, discretization + 1)
        for comb in itertools.product(discretized_values, repeat=subgroup_size):
            if np.isclose(sum(comb), 1):
                yield comb

    # Lazy generator for all combinations of subgroup permutations
    def lazy_permutations():
        subgroup_generators = [valid_subgroup_permutations(size) for size in subgroup_sizes]
        for subgroup_comb in itertools.product(*subgroup_generators):
            yield np.concatenate(subgroup_comb)

    # Total possible permutations count
    num_total_permutations = np.prod(
        [sum(1 for _ in valid_subgroup_permutations(size)) for size in subgroup_sizes]
    )

    # If num_samples is -1 or larger than possible permutations, return all permutations
    if num_samples == -1 or num_samples > num_total_permutations:
        num_samples = num_total_permutations

    # Generate a random sample of permutations without storing all
    sampled_indices = np.random.choice(num_total_permutations, num_samples, replace=False)

    # Lazy generation of sampled permutations
    def sampled_permutations():
        for i, perm in enumerate(lazy_permutations()):
            if i in sampled_indices:
                yield perm

    return sampled_permutations()


def generate_and_transform_sequence(dim, samplesize, target_distribution, sequence_type="sobol"):
    """
    Generate a low-discrepancy sequence in the unit hypercube and map it to the target distribution space
    using the iso-probabilistic transformation.

    Parameters:
    - dim: int, number of dimensions.
    - samplesize: int, number of samples.
    - target_distribution: openturns.Distribution, the target distribution to transform the sequence.
    - sequence_type: str, one of ['sobol', 'halton', 'reverse_halton', 'faure', 'haselgrove'].

    Returns:
    - transformed_sample: np.array, the transformed sample points in the target distribution space.
    """

    # 1. Create a low-discrepancy sequence in the unit hypercube [0, 1]^dim
    if sequence_type == "sobol":
        sequence = ot.SobolSequence(dim)
    elif sequence_type == "halton":
        sequence = ot.HaltonSequence(dim)
    elif sequence_type == "reverse_halton":
        sequence = ot.ReverseHaltonSequence(dim)
    elif sequence_type == "faure":
        sequence = ot.FaureSequence(dim)
    elif sequence_type == "haselgrove":
        sequence = ot.HaselgroveSequence(dim)
    else:
        raise ValueError(
            "Unsupported sequence type. Choose one of ['sobol', 'halton', 'reverse_halton', 'faure', 'haselgrove']."
        )

    uniform_sample = sequence.generate(samplesize)

    # 2. Create a composed uniform distribution in [0,1]^dim
    composed_uniform = ot.ComposedDistribution([ot.Uniform(0.0, 1.0)] * dim)

    # 3. Apply the iso-probabilistic transformation to map the uniform sample to the standard normal space
    standard_normal_sample = composed_uniform.getIsoProbabilisticTransformation()(uniform_sample)

    # 4. Apply the inverse iso-probabilistic transformation of the target distribution
    transformed_sample = target_distribution.getInverseIsoProbabilisticTransformation()(
        standard_normal_sample
    )

    # Return the transformed sample as a NumPy array
    return np.asarray(transformed_sample)


def compose_defects_with_lambdas(lds, rdv):
    """
    Multiplies each defect sample by the corresponding lambda sample.

    For each point in the lambda sample, this function generates a new defect
    sample by multiplying it with the point from the lambda sample. The dimensions
    of the lambda sample and the defect sample must match.

    Parameters:
    -----------
    lds : ot.Sample
        Lambda sample of size Nl.
    rdv : ot.Sample
        Random defects sample of size Nd.

    Returns:
    --------
    list[ot.Sample]
        List of size Nl, where each element is a sample of defects, each with size Nd.
    """

    # Ensure that the dimensions of lambdas and defects match
    assert rdv.getDimension() == lds.getDimension(), "Lambda and defect samples must have the same dimension."

    # Get the description of the defect sample
    defect_description = rdv.getDescription()

    # Create a list of scaled defect samples
    scaled_defect_samples = []
    for lambda_point in lds:
        # Multiply each lambda point by the defect sample
        scaled_sample = ot.Sample(np.array(lambda_point) * np.array(rdv))
        scaled_sample.setDescription(defect_description)  # Preserve the description
        scaled_defect_samples.append(scaled_sample)

    return scaled_defect_samples
