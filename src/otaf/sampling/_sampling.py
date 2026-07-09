from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "condition_lambda_sample",
    "validate_and_extract_indices",
    "condition_sample_array",
    "find_best_worst_quantile",
    "generate_lhs_experiment",
    "generate_random_permutations_with_sampling",
    "generate_scaled_permutations",
    "generate_imprecise_probabilistic_samples",
    "generate_and_transform_sequence",
    "compose_defects_with_lambdas",
    "scale_sample_with_params",
    "calculate_sample_in_deviation_domain"
]

import itertools
import logging
import re
import numpy as np
import openturns as ot
from beartype import beartype
from beartype.typing import Union, Sequence, Optional, Generator, Literal


if hasattr(ot, 'JointDistribution'):
    JointDistribution = ot.JointDistribution
else:
    JointDistribution = ot.ComposedDistribution

@beartype
def condition_lambda_sample(sample: ot.Sample, squared_sum: bool = False) -> ot.Sample:
    """Condition lambda parameters to sum to 1 for each feature.

    Parameters
    ----------
        sample (ot.Sample): Lambda parameters with each feature ending in an integer sequence.
        squared_sum (bool, optional): Whether to square the resulting values. Defaults to False.

    Returns
    -------
        ot.Sample: Conditioned lambda parameters.

    Note
    ----
    This function can only be used if all the features are planar. If they are cylindrical
    the constraints are aboluetly different. To use both you need to make a complex function
    that checks the feature type for each variable and uses the right constraint.
    """
    logging.info("[condition_lambda_sample] Started. Checking input sample.")
    deviation_symbols = list(sample.getDescription())
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

    Parameters
    ----------
        description (list[str]): List of feature descriptions.

    Returns
    -------
        list[int]: Extracted feature indices.

    Raises
    ------
        ValueError: If the description is missing or invalid.
    """
    if not description:
        raise ValueError("Description missing. Must be provided.")
    logging.info("[validate_and_extract_indices] Extracting feature indices.")
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

    Parameters
    ----------
        sample_array (np.ndarray): The input sample array.
        feature_indices (list[int]): Indices of the features.
        squared_sum (bool): Whether to square the resulting values.

    Returns
    -------
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
        logging.error(f"Error in condition_sample_array: {e}")
        print(
            f"Error encountered in condition_sample_array:\n"
            f"sample_array: {sample_array}\n"
            f"feature_indices: {feature_indices}\n"
            f"squared_sum: {squared_sum}"
        )
        raise


def find_best_worst_quantile(parameters, results, quantile):
    """Find the best and worst performing observations based on a given quantile.

    This function calculates the best and worst performing observations from a given set of parameters
    and corresponding results, based on the specified quantile.

    Parameters
    ----------
        parameters (numpy.ndarray): Array containing parameter values.
        results (numpy.ndarray): Array containing corresponding result values.
        quantile (float): Desired quantile value, must be in the range [0, 1].

    Returns
    -------
        tuple: A tuple containing two tuples:
            - The first tuple contains the parameters and results of the best performing observations.
            - The second tuple contains the parameters and results of the worst performing observations.
    """
    pop_size = len(results)
    pop_quantile = int(pop_size * quantile)
    sorted_result_args = np.argsort(results)
    best_params = parameters[sorted_result_args[:pop_quantile]]
    best_res = results[sorted_result_args[:pop_quantile]]
    worst_params = parameters[sorted_result_args[-pop_quantile:]]
    worst_res = results[sorted_result_args[-pop_quantile:]]

    return (best_params, best_res), (worst_params, worst_res)


@beartype
def generate_lhs_experiment(
    composed_distribution: JointDistribution,
    N: int,
    SEED: int = 999,
    T0: float = 10.0,
    c: float = 0.95,
    iMax: int = 2000
) -> ot.Sample:
    """Generate an optimal Latin Hypercube Sample (LHS) design.

    Construct a space-filling design using the Simulated Annealing algorithm
    to minimize the C2 criterion, ensuring a high-quality distribution of samples
    within the input space.

    Parameters
    ----------
    composed_distribution : JointDistribution
        The probability distribution from which to draw samples.
    N : int
        The number of samples to generate in the design.
    SEED : int, optional
        Seed for the random number generator, by default 999.
    T0 : float, optional
        Initial temperature for the simulated annealing process, by default 10.
    c : float, optional
        Geometric cooling factor for the simulated annealing process, by default 0.95.
    iMax : int, optional
        Maximum number of iterations for the optimization, by default 2000.

    Returns
    -------
    ot.Sample
        The optimized Latin Hypercube Sample design.
    """
    ot.RandomGenerator.SetSeed(SEED)
    lhsExperimentGenerator = ot.LHSExperiment(composed_distribution, N)
    spaceFilling = ot.SpaceFillingC2()
    geomProfile = ot.GeometricProfile(T0, c, iMax)
    optimalLHSAlgorithm = ot.SimulatedAnnealingLHS(
        lhsExperimentGenerator, spaceFilling, geomProfile
    )
    optimalDesign = optimalLHSAlgorithm.generate()
    return optimalDesign


@beartype
def generate_random_permutations_with_sampling(
    subgroup_sizes: list[int],
    num_samples: Optional[int] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate a random subset of signed permutations for structural configurations.

    Create a random subset of valid permutations for concatenated one-hot encoded 
    subgroups, where each non-zero element is randomly assigned a +1 or -1 sign 
    to mitigate the curse of dimensionality.

    Parameters
    ----------
    subgroup_sizes : list[int]
        List where each entry is the size of a subgroup (number of one-hot vectors).
    num_samples : int, optional
        Number of permutations to generate. If None or exceeding the total number 
        of possible permutations, all possible permutations are generated.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    np.ndarray
        A 2D array where each row is a generated permutation, including 
        randomized sign flips.

    Notes
    -----
    - This function addresses the curse of dimensionality by limiting permutations 
      to `num_samples`.
    - If `num_samples` exceeds total possible permutations, all unique combinations 
      are returned.
    - Each subgroup is constrained to exactly one non-zero element per sample.
    """
    if seed is not None:
        np.random.seed(seed)
    subgroups = [np.eye(size, dtype=int) for size in subgroup_sizes]
    num_structural_configs = np.prod(subgroup_sizes)
    num_sign_flips = 2 ** len(subgroup_sizes)
    total_permutations = num_structural_configs * num_sign_flips
    if num_samples is None or num_samples > total_permutations:
        num_samples = total_permutations
    all_indices = np.indices(subgroup_sizes).reshape(len(subgroup_sizes), -1).T
    structural_sample_size = min(len(all_indices), num_samples)
    sampled_indices = all_indices[
        np.random.choice(all_indices.shape[0], structural_sample_size, replace=False)
    ]
    sampled_permutations = np.array(
        [
            np.concatenate([subgroups[i][idx] for i, idx in enumerate(sample)])
            for sample in sampled_indices
        ]
    )
    sign_flips = np.random.choice([1, -1], size=(structural_sample_size, len(subgroup_sizes)))
    start = 0
    for i, size in enumerate(subgroup_sizes):
        sampled_permutations[:, start : start + size] *= sign_flips[:, i].reshape(-1, 1)
        start += size
    return sampled_permutations


@beartype
def generate_scaled_permutations(
    subgroup_sizes: list[int],
    scaling_factors: Union[list[float], np.ndarray],
    num_samples: Optional[int] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate scaled random permutations with sign flips.

    Create a set of random permutations for concatenated one-hot encoded subgroups, 
    applying random sign flips (+1 or -1) and scaling each element by the provided 
    factors.

    Parameters
    ----------
    subgroup_sizes : list[int]
        Size of each subgroup, representing the number of possible one-hot vectors.
    scaling_factors : Union[list[float], np.ndarray]
        Scaling factors for each element in the permutation vector. Must match 
        the total sum of `subgroup_sizes`.
    num_samples : int, optional
        Number of permutations to generate. If None or exceeding the total possible 
        combinations, all combinations are generated.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        A 2D array where each row is a generated permutation, scaled and 
        sign-flipped according to the logic defined in the underlying sampling.

    Raises
    ------
    ValueError
        If the length of `scaling_factors` does not match the sum of `subgroup_sizes`.
    """
    permutations = generate_random_permutations_with_sampling(
        subgroup_sizes, num_samples=num_samples, seed=seed
    )
    total_size = np.sum(subgroup_sizes)
    if len(scaling_factors) != total_size:
        raise ValueError(
            f"Length of scaling_factors ({len(scaling_factors)}) must match the total number of elements ({total_size}) in the permutation vector."
        )
    scaled_permutations = permutations * scaling_factors
    return scaled_permutations


@beartype
def generate_imprecise_probabilistic_samples(
    subgroup_sizes: list[int],
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
    discretization: int = 4
) -> Generator[np.ndarray, None, None]:
    """Generate a random subset of valid permutations for imprecise probabilistic models.

    Each subgroup in `subgroup_sizes` represents a set of variables contributing to a 
    probabilistic model, where the sum of elements within each subgroup is constrained to 1. 
    This function samples random permutations from this constrained space.

    Parameters
    ----------
    subgroup_sizes : list[int]
        List where each entry corresponds to the size of a subgroup (number of variables).
    num_samples : int, optional
        Number of random samples to generate. If -1, all valid permutations are computed.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
    discretization : int, optional
        Number of steps to discretize the space between [0, 1]. Higher values provide
        finer granularity, by default 4.

    Returns
    -------
    Generator[np.ndarray, None, None]
        A generator yielding valid samples of probabilistic contributions. Each sample is a 
        1D array where the sum of elements in each subgroup is 1.

    Example
    -------
    >>> subgroup_sizes = [3, 4]
    >>> samples = generate_imprecise_probabilistic_samples(subgroup_sizes, num_samples=5, seed=42)
    >>> for sample in samples:
    ...     print(sample)

    Notes
    -----
    - This function explores the space of imprecise probabilistic contributions.
    - The generator produces samples lazily to reduce memory consumption.
    - If `num_samples` is -1, the function generates all possible permutations lazily.
    - The `discretization` parameter controls the precision of the [0, 1] interval; 
      higher values increase precision but may significantly increase computation time.
    """
    if seed is not None:
        np.random.seed(seed)
        
    def valid_subgroup_permutations(subgroup_size):

        discretized_values = np.linspace(0, 1, discretization + 1)
        for comb in itertools.product(discretized_values, repeat=subgroup_size):
            if np.isclose(sum(comb), 1):
                yield comb
                
    def lazy_permutations():
        subgroup_generators = [valid_subgroup_permutations(size) for size in subgroup_sizes]
        for subgroup_comb in itertools.product(*subgroup_generators):
            yield np.concatenate(subgroup_comb)
            
    num_total_permutations = np.prod(
        [sum(1 for _ in valid_subgroup_permutations(size)) for size in subgroup_sizes]
    )
    if num_samples == -1 or num_samples > num_total_permutations:
        num_samples = num_total_permutations
    sampled_indices = np.random.choice(num_total_permutations, num_samples, replace=False)
    
    def sampled_permutations():
        for i, perm in enumerate(lazy_permutations()):
            if i in sampled_indices:
                yield perm

    return sampled_permutations()


@beartype
def generate_and_transform_sequence(
    dim: int,
    samplesize: int,
    target_distribution: ot.Distribution,
    sequence_type: Literal['sobol', 'halton', 'reverse_halton', 'faure', 'haselgrove'] = "halton"
) -> np.ndarray:
    """Generate a low-discrepancy sequence and transform it to the target distribution.

    Map a low-discrepancy sequence from the unit hypercube to the target distribution 
    space using iso-probabilistic transformations.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    samplesize : int
        Number of samples to generate.
    target_distribution : ot.Distribution
        The target OpenTURNS distribution to transform the sequence into.
    sequence_type : str, optional
        Type of low-discrepancy sequence, by default "halton". 
        Must be one of ['sobol', 'halton', 'reverse_halton', 'faure', 'haselgrove'].

    Returns
    -------
    np.ndarray
        Transformed sample points in the target distribution space.

    Raises
    ------
    ValueError
        If an unsupported sequence type is provided.
    """
    sequences = {
        "sobol": ot.SobolSequence,
        "halton": ot.HaltonSequence,
        "reverse_halton": ot.ReverseHaltonSequence,
        "faure": ot.FaureSequence,
        "haselgrove": ot.HaselgroveSequence
    }
    if sequence_type not in sequences:
        raise ValueError(f"Unsupported sequence type. Choose one of {list(sequences.keys())}.")
    
    sequence = sequences[sequence_type](dim)
    uniform_sample = sequence.generate(samplesize)
    composed_uniform = JointDistribution([ot.Uniform(0.0, 1.0)] * dim)
    standard_normal_sample = composed_uniform.getIsoProbabilisticTransformation()(uniform_sample)
    transformed_sample = target_distribution.getInverseIsoProbabilisticTransformation()(
        standard_normal_sample
    )
    return np.asarray(transformed_sample)


@beartype
def compose_defects_with_lambdas(lds: ot.Sample, rdv: ot.Sample) -> list[ot.Sample]:
    """Calculate scaled defect samples by multiplying with lambda values.

    For each point in the lambda sample, compute a new defect sample by performing
    an element-wise multiplication with the input defect sample.

    Parameters
    ----------
    lds : ot.Sample
        Lambda sample of size Nl.
    rdv : ot.Sample
        Random defects sample of size Nd.

    Returns
    -------
    list[ot.Sample]
        A list of size Nl, where each element contains the scaled defect sample 
        of size Nd.

    Raises
    ------
    AssertionError
        If the dimensions of the lambda sample and defect sample do not match.
    """
    assert rdv.getDimension() == lds.getDimension(), "Lambda and defect samples must have the same dimension."
    defect_description = rdv.getDescription()
    scaled_defect_samples = []
    for lambda_point in lds:
        scaled_sample = ot.Sample(np.array(lambda_point) * np.array(rdv))
        scaled_sample.setDescription(defect_description)
        scaled_defect_samples.append(scaled_sample)
    return scaled_defect_samples


@beartype
def scale_sample_with_params(
    sample: np.ndarray,
    parameters: Union[np.ndarray, Sequence[float]]
) -> np.ndarray:
    r"""Scale a sample of shape (N, M) using mean and standard deviation parameters.

    Transform each column of the input sample, assuming it originates from a 
    standard normal unit distribution, by applying the provided mean and 
    standard deviation values.

    Parameters
    ----------
    sample : np.ndarray
        A 2D array of shape (N, M), where each column represents a random vector 
        component from a standard normal distribution.
    parameters : Union[np.ndarray, Sequence[float]]
        A 1D array or sequence of size 2 * M, where elements alternate between 
        mean and standard deviation ($\mu_0, \sigma_0, \mu_1, \sigma_1, \dots$).

    Returns
    -------
    np.ndarray
        A 2D array of shape (N, M), with each column scaled using the parameters.

    Raises
    ------
    ValueError
        If the sample is not a 2D array.
    ValueError
        If the parameters size does not equal 2 * M.
    ValueError
        If any standard deviation is negative.
    """
    if sample.ndim != 2:
        raise ValueError(f"Sample must be a 2D array, but got shape {sample.shape}.")
    N, M = sample.shape
    parameters = np.asarray(parameters)
    if parameters.shape[0] != 2 * M:
        raise ValueError(
            f"Parameters must have size 2 * M (expected {2 * M}, got {parameters.shape[0]})."
        )
    means = parameters[::2]
    stds = parameters[1::2]
    if np.any(stds < 0):
        raise ValueError(f"All standard deviations must be positive : {stds}")
    scaled_sample = sample * stds + means
    return scaled_sample


def calculate_sample_in_deviation_domain(
    pos_u: np.ndarray, theta: np.ndarray, lambda_pos: float, lambda_theta: float
) -> np.ndarray:
    """
    Scale and stack position and theta arrays into a 2D coordinate array.

    Applies independent scaling factors to both the positional and angular 
    input vectors, then reshapes them into a paired (N, 2) array.

    Parameters
    ----------
    pos_u : array_like
        An array or sequence of raw positional coordinates.
    theta : array_like
        An array or sequence of raw angular coordinates.
    lambda_pos : float
        The scalar multiplier applied to scale the position data.
    lambda_theta : float
        The scalar multiplier applied to scale the theta data.

    Returns
    -------
    array_like
        An (N, 2) NumPy array where each row represents a paired 
        (scaled_pos, scaled_theta) coordinate.
    """
    scaled_pos = lambda_pos * pos_u
    scaled_theta = lambda_theta * theta
    sample = np.vstack((scaled_pos, scaled_theta)).T
    return sample
