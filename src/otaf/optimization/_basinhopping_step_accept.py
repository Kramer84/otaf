from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "StepTaking",
    "AcceptTest",
    "Callback",
    "check_constraint_with_tolerance",
]

import numpy as np
from scipy.optimize import LinearConstraint

from otaf.sampling import validate_and_extract_indices, condition_sample_array
from otaf.exceptions import timeout as o_timeout

class StepTaking:
    """
    Custom class for generating bounded steps in the optimization space.

    Attributes:
    ----------

    linear_condition : callable
        A function to enforce linear constraints on subsets of variables.
    stepsize : float
        The size of the step to be taken (default 0.5).
    """

    def __init__(self, storage, defect_description, stepsize=0.5):
        """
        Initialize the StepTaking class.

        Parameters:
        ----------
        storage : OptimizationStorage
            An instance of the OptimizationStorage class to manage optimization points.
        defect_description : list
            Descriptions of the features, used to normalize the steps.
        stepsize : float, optional
            The size of the step to be taken (default: 0.5).
        """
        self.storage = storage  # Use OptimizationStorage to store and manage points
        self.feature_indices = validate_and_extract_indices(
            defect_description
        )  # Normalize within features
        self.linear_constraints = linear_constraint_from_feature_indices(
            self.feature_indices
        )  # Linear constraint object
        self.stepsize = stepsize

    def __call__(self, x):
        """
        Generate a new candidate step based on the current position 'x'.
        The step respects the provided bounds and enforces the linear condition.

        Parameters:
        ----------
        x : array-like
            The current point in the search space.

        Returns:
        -------
        new_x : array-like
            The new point after taking a step.
        """
        # Generate a random step vector
        step = np.random.uniform(-self.stepsize, self.stepsize, size=x.shape)
        # Add step to current position
        new_x = x + step
        new_x = np.where(
            new_x < 0, 0, new_x
        )  # Only positive, needs to be changed if working with imprecise corelation.

        return condition_sample_array(
            new_x, self.feature_indices, squared_sum=False
        )  # Normalizing.


class AcceptTest:
    """
    Custom class for accepting or rejecting new steps in the optimization process.

    Attributes:
    ----------
    storage : OptimizationStorage
        An instance of the OptimizationStorage class to store and manage optimization points.
    linear_constraints : LinearConstraint
        A function to enforce linear constraints on the variables.
    distance_threshold : float
        Minimum distance between points to avoid revisiting.
    surrogate_model : callable
        A callable to update a surrogate model based on new points.
    """

    def __init__(self, storage, defect_description, distance_threshold=1.0, surrogate_model=None):
        self.storage = storage  # Instance of OptimizationStorage for storing and managing points
        self.feature_indices = validate_and_extract_indices(
            defect_description
        )  # Normalize within features
        self.linear_constraints = linear_constraint_from_feature_indices(
            self.feature_indices
        )  # Linear constraint object
        self.distance_threshold = distance_threshold
        self.surrogate_model = surrogate_model

    def __call__(self, f_new, x_new, f_old, x_old):
        """
        Decide whether to accept or reject a new step in the optimization.

        Parameters:
        ----------
        f_new : float
            The function value at the new point.
        x_new : array-like
            The coordinates of the new point.
        f_old : float
            The function value at the previous point.
        x_old : array-like
            The coordinates of the previous point.

        Returns:
        -------
        bool
            True if the new step is accepted, False otherwise.
        """
        # Check if the new point satisfies the linear constraints
        constraint_satisfied, _ = check_constraint_with_tolerance(self.linear_constraints, x_new)
        if not constraint_satisfied:
            return False

        # Check distance from previous points
        if self._is_too_close(x_new):
            return False

        # Optionally update the surrogate model with new data
        if self.surrogate_model:
            self.surrogate_model.update(x_new, f_new)

        # Store the new point using OptimizationStorage
        self.storage.add_point(x_new, f_new, source="local")

        # Accept new point if the function value improves, else use the Metropolis criterion
        return self._accept_based_on_value(f_new, f_old)

    @o_timeout(seconds=5, error_message="Execution took too long!")
    def _is_too_close(self, x_new):
        """
        Check if the new point is too close to any existing points in the storage.

        Parameters:
        ----------
        x_new : array-like
            The coordinates of the new point.

        Returns:
        -------
        bool
            True if the point is too close to any existing point, False otherwise.
        """
        for _, row in self.storage.iter_points(
            source="global", constraints_respected=True, bounds_respected=True
        ):
            x_existing = row["point"]
            distance = np.linalg.norm(x_new - x_existing)
            if distance < self.distance_threshold:
                return True
        return False

    def _accept_based_on_value(self, f_new, f_old):
        """
        Decide whether to accept the new point based on its function value.

        Parameters:
        ----------
        f_new : float
            The function value at the new point.
        f_old : float
            The function value at the previous point.

        Returns:
        -------
        bool
            True if the new point is accepted, False otherwise.
        """
        if f_new < f_old:
            return True
        else:
            # Metropolis criterion for worse solutions
            return np.random.uniform(0, 1) < np.exp(f_old - f_new)


class Callback:
    """A callback class to store accepted local minima and allow data sharing between components."""

    def __init__(self, storage, stop_criterion=None):
        """
        Initialize the Callback class.

        Parameters:
        ----------
        storage : OptimizationStorage
            An instance of the OptimizationStorage class to manage optimization data.
        stop_criterion : callable or None, optional
            A user-defined function that determines whether to stop the basinhopping routine.
        """
        self.storage = storage  # Use OptimizationStorage to store points
        self.stop_criterion = stop_criterion

    def __call__(self, x, f, accept):
        """
        Store the current local minimum if accepted and apply the stop criterion if provided.

        Parameters:
        ----------
        x : array-like
            The coordinates of the current minimum.
        f : float
            The function value at the current minimum.
        accept : bool
            Whether the current minimum was accepted.

        Returns:
        -------
        bool or None
            Return True to stop the basinhopping routine if the stop criterion is met, otherwise return None.
        """
        # Only store accepted minima
        if accept:
            self.storage.add_point(x, f, source="local")

        # Check user-defined stop criterion
        if self.stop_criterion and self.stop_criterion(x, f, accept):
            return True  # Stop the basinhopping routine

    def get_data(self):
        """Retrieve the stored local minima points."""
        return self.storage.get_all_data()


@o_timeout(seconds=5, error_message="Execution took too long!")
def linear_constraint_from_feature_indices(
    feature_indices: list[int], tol: float = 0, keep_feasible: bool = True
):
    """Create a linear constraint ensuring that lambda parameters sum to 1 for each feature.

    Args:
        feature_indices (list[int]): List of feature indices extracted from the sample description.
        tol (float, optional): Tolerance for the constraint bounds. Defaults to 1e-12.
        keep_feasible (bool, optional): Whether to keep the constraint feasible. Defaults to True.

    Returns:
        LinearConstraint: Linear constraint ensuring that lambda parameters for each feature sum to 1.
    """
    if not feature_indices:
        raise ValueError("Feature indices must be provided.")

    unique_features_ids = np.unique(feature_indices).tolist()
    n_symb = len(feature_indices)
    n_feat = len(unique_features_ids)

    # Initialize the constraint matrix
    constraint_matrix = np.zeros((n_feat, n_symb), dtype="float64")

    # Build the constraint matrix where each row corresponds to a feature
    for i, feature_idx in enumerate(unique_features_ids):
        for j, idx in enumerate(feature_indices):
            if feature_idx == idx:
                constraint_matrix[i, j] = 1.0

    # Define the lower and upper bounds for the linear constraint
    lb = np.ones((n_feat,)) - tol
    ub = np.ones((n_feat,)) + tol

    # Create the linear constraint
    linear_constraint = LinearConstraint(constraint_matrix, lb, ub, keep_feasible)

    return linear_constraint


def check_constraint_with_tolerance(linear_constraint, x, tolerance=1e-6):
    """
    Check if the vector x satisfies the linear constraint within a specified tolerance.

    Parameters
    ----------
    linear_constraint : scipy.optimize.LinearConstraint
        The linear constraint object containing A, lb, and ub.
    x : array_like
        The vector of independent variables to check.
    tolerance : float, optional
        The tolerance within which the constraint should be satisfied (default is 1e-6).

    Returns
    -------
    bool
        True if the constraint is satisfied within the tolerance, False otherwise.
    dict
        A dictionary containing the lower and upper residuals.
    """
    sl, sb = linear_constraint.residual(x)

    # Check if all residuals are greater than or equal to -tolerance
    lower_residuals_check = np.all(sl >= -tolerance)
    upper_residuals_check = np.all(sb >= -tolerance)

    # If both checks pass, the constraint is satisfied
    constraint_satisfied = lower_residuals_check and upper_residuals_check

    # Return the result and the residuals for further analysis if needed
    return constraint_satisfied, {"lower_residuals": sl, "upper_residuals": sb}
