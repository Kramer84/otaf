from __future__ import annotations

__author__ = "Kramer84"
__all__ = ["OptimizationTracker"]
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from beartype import beartype
from scipy.optimize import Bounds


class OptimizationTracker:
    """
    High-performance storage for optimization loops.

    Uses byte-hashing for ``O(1)`` lookups during execution, and exports to 
    Pandas for post-processing and filtering.

    Parameters
    ----------
    bounds : Bounds, optional
        The optimization bounds object exposing a ``.residual(x)`` method.
        Default is None.
    constraint_tolerance : float, optional
        The maximum allowed tolerance for constraint violations. 
        Default is 1e-4.
    precision_decimals : int, optional
        The number of decimals to round coordinate values before 
        byte-hashing. Default is 8.

    Attributes
    ----------
    history : dict
        Nested dictionary tracking evaluation points by experiment key and 
        point hash.
    precision_decimals : int
        Decimal precision for coordinate rounding.
    bounds : Bounds or None
        Optimization bounds constraint definition.
    constraint_tolerance : float
        Tolerance threshold for constraint validation.
    """

    def __init__(
        self,
        bounds: Optional[Bounds] = None,
        constraint_tolerance: float = 0.0001,
        precision_decimals: int = 8,
    ) -> None:
        self.history: dict[Any, dict[bytes, dict[str, Any]]] = {}
        self.precision_decimals = precision_decimals
        self.bounds = bounds
        self.constraint_tolerance = constraint_tolerance
        logging.info("OptimizationTracker initialized with bounds: %s", self.bounds)

    def _hash_x(self, x: Union[np.ndarray, List[float], float]) -> bytes:
        """
        Hash the coordinates of a point using rounded byte representations.

        Parameters
        ----------
        x : array_like
            The input coordinate point.

        Returns
        -------
        bytes
            The immutable byte-string representation of the rounded input 
            array.
        """
        return np.round(x, self.precision_decimals).tobytes()

    def _init_entry(
        self, exp_key: Any, x_hash: bytes, x: Union[np.ndarray, List[float], float]
    ) -> None:
        """
        Initialize an empty structured entry in the evaluation history.

        Parameters
        ----------
        exp_key : Any
            The identifier of the active experiment.
        x_hash : bytes
            The calculated hash key of the point coordinate.
        x : array_like
            The raw coordinate array.
        """
        if exp_key not in self.history:
            self.history[exp_key] = {}
        if x_hash not in self.history[exp_key]:
            self.history[exp_key][x_hash] = {
                "point": np.array(x),
                "bounds_respected": self._check_bounds(x),
            }

    def _check_bounds(self, x: Union[np.ndarray, List[float], float]) -> bool:
        """
        Evaluate whether a given point stays within the registered bounds.

        Parameters
        ----------
        x : array_like
            The target coordinate array to evaluate.

        Returns
        -------
        bool
            True if all bounds residuals are greater than or equal to 0, 
            or if bounds are omitted.
        """
        if self.bounds is None:
            return True
        sl, sb = self.bounds.residual(x)
        return bool(np.all(sl >= 0) and np.all(sb >= 0))

    def update_objective_data(
        self,
        exp_key: Any,
        x: Union[np.ndarray, List[float], float],
        fp_gld: float,
        fp_slack: float,
        gld_params: Union[np.ndarray, List[float]],
        failure_slack: float,
        source: str = "local",
    ) -> None:
        """
        Update objective function metrics for a specific point.

        Parameters
        ----------
        exp_key : Any
            The identifier of the active experiment.
        x : array_like
            The coordinates of the evaluated point.
        fp_gld : float
            Failure probability obtained from the GLD distribution 
            approximation (converted to ``NaN`` if input is ``NaN``).
        fp_slack : float
            Failure probability obtained from number of failed points 
            in sample.
        gld_params : array_like
            Parameters of the GLD distribution.
        failure_slack : float
            The slack value used to define failure.
        source : str, optional
            The metadata tracking origin string of the evaluation. 
            Default is "local".
        """
        x_hash = self._hash_x(x)
        self._init_entry(exp_key, x_hash, x)
        self.history[exp_key][x_hash].update(
            {
                "source": source,
                "FP_GLD": float(fp_gld) if not np.isnan(fp_gld) else np.nan,
                "FP_SLACK": float(fp_slack),
                "GLD_PARAMS": np.array(gld_params),
                "FAILURE_SLACK": float(failure_slack),
            }
        )

    def update_constraint_data(
        self,
        exp_key: Any,
        x: Union[np.ndarray, List[float], float],
        constraints: Union[np.ndarray, List[float], float],
    ) -> None:
        """
        Update constraint evaluation metrics and assess tolerance breaches.

        Parameters
        ----------
        exp_key : Any
            The identifier of the active experiment.
        x : array_like
            The coordinates of the evaluated point.
        constraints : array_like
            Array or values representing current constraint evaluation 
            outputs.
        """
        x_hash = self._hash_x(x)
        self._init_entry(exp_key, x_hash, x)
        constraints_respected = bool(np.all(constraints <= self.constraint_tolerance))
        self.history[exp_key][x_hash].update(
            {
                "constraint_values": np.array(constraints, dtype=float),
                "constraints_respected": constraints_respected,
            }
        )

    def get_data(
        self, exp_key: Any, x: Union[np.ndarray, List[float], float]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve internal stored attributes for a point if existing.

        Parameters
        ----------
        exp_key : Any
            The identification key for the tracking sequence.
        x : array_like
            The exact coordinate values to fetch.

        Returns
        -------
        dict or None
            The dictionary of matching parameters, or ``None`` if the 
            point cannot be found.
        """
        x_hash = self._hash_x(x)
        return self.history.get(exp_key, {}).get(x_hash, None)

    def to_dataframe(self, exp_key: Optional[Any] = None) -> pd.DataFrame:
        """
        Convert the internal dictionary to a Pandas DataFrame.

        Replaces the old ``get_all_data`` and avoids loop bottlenecks.

        Parameters
        ----------
        exp_key : Any, optional
            The specific experiment key to extract. If None, flattens and 
            extracts all stored items across tracking sequences. 
            Default is None.

        Returns
        -------
        pd.DataFrame
            Flattened historical evaluation frame containing structural 
            parameter rows.
        """
        rows = []
        keys_to_process = [exp_key] if exp_key else self.history.keys()
        for k in keys_to_process:
            if k not in self.history:
                continue
            for entry in self.history[k].values():
                row = entry.copy()
                row["experiment"] = k
                rows.append(row)
        return pd.DataFrame(rows)

    def filter_points(
        self,
        exp_key: Optional[Any] = None,
        source: Optional[str] = None,
        bounds_respected: Optional[bool] = None,
        constraints_respected: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Filter tracking records against targeted properties via DataFrame.

        Parameters
        ----------
        exp_key : Any, optional
            Filters historical points to a single tracked experiment key 
            sequence. Default is None.
        source : str, optional
            Filters matching points by execution string label. Default 
            is None.
        bounds_respected : bool, optional
            Filters items based on validation boundary constraints 
            compliance. Default is None.
        constraints_respected : bool, optional
            Filters items based on tolerance limit boundary validation. 
            Default is None.

        Returns
        -------
        pd.DataFrame
            A targeted slice of history matching all provided validation 
            conditions.
        """
        df = self.to_dataframe(exp_key)
        if df.empty:
            return df
        if source is not None and "source" in df.columns:
            df = df[df["source"] == source]
        if bounds_respected is not None and "bounds_respected" in df.columns:
            df = df[df["bounds_respected"] == bounds_respected]
        if constraints_respected is not None and "constraints_respected" in df.columns:
            df = df[df["constraints_respected"] == constraints_respected]
        return df
