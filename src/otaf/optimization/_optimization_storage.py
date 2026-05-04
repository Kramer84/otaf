from __future__ import annotations
# -*- coding: utf-8 -*-

__author__ = "Kramer84"
__all__ = [
    "OptimizationTracker",
]

import numpy as np
import pandas as pd
import logging

class OptimizationTracker:
    """
    High-performance storage for optimization loops.
    Uses byte-hashing for O(1) lookups during execution, and exports to Pandas
    for post-processing and filtering.
    """
    def __init__(self, bounds=None, constraint_tolerance=1e-4, precision_decimals=8):
        self.history = {}
        self.precision_decimals = precision_decimals
        self.bounds = bounds
        self.constraint_tolerance = constraint_tolerance

        logging.info("OptimizationTracker initialized with bounds: %s", self.bounds)

    def _hash_x(self, x):
        return np.round(x, self.precision_decimals).tobytes()

    def _init_entry(self, exp_key, x_hash, x):
        if exp_key not in self.history:
            self.history[exp_key] = {}
        if x_hash not in self.history[exp_key]:
            self.history[exp_key][x_hash] = {
                "point": np.array(x),
                "bounds_respected": self._check_bounds(x)
            }

    def _check_bounds(self, x):
        if self.bounds is None:
            return True
        sl, sb = self.bounds.residual(x)
        return np.all(sl >= 0) and np.all(sb >= 0)

    def update_objective_data(self, exp_key, x, fp_gld, fp_slack, gld_params, failure_slack, source="local"):
        x_hash = self._hash_x(x)
        self._init_entry(exp_key, x_hash, x)

        self.history[exp_key][x_hash].update({
            "source": source,
            "FP_GLD": float(fp_gld) if not np.isnan(fp_gld) else np.nan,
            "FP_SLACK": float(fp_slack),
            "GLD_PARAMS": np.array(gld_params),
            "FAILURE_SLACK": float(failure_slack)
        })

    def update_constraint_data(self, exp_key, x, constraints):
        x_hash = self._hash_x(x)
        self._init_entry(exp_key, x_hash, x)

        # Check if the maximum constraint violation is within the allowed tolerance
        # (Assuming constraints are formulated as g(x) <= 0 or slacks within [-tol, tol])
        constraints_respected = bool(np.all(constraints <= self.constraint_tolerance))

        self.history[exp_key][x_hash].update({
            "constraint_values": np.array(constraints, dtype=float),
            "constraints_respected": constraints_respected
        })

    def get_data(self, exp_key, x):
        x_hash = self._hash_x(x)
        return self.history.get(exp_key, {}).get(x_hash, None)

    def to_dataframe(self, exp_key=None):
        """
        Converts the internal dictionary to a Pandas DataFrame.
        Replaces the old 'get_all_data' and avoids loop bottleneck.
        """
        rows = []
        # If no key is provided, aggregate all experiments
        keys_to_process = [exp_key] if exp_key else self.history.keys()

        for k in keys_to_process:
            if k not in self.history:
                continue
            for entry in self.history[k].values():
                row = entry.copy()
                row["experiment"] = k
                rows.append(row)

        return pd.DataFrame(rows)

    def filter_points(self, exp_key=None, source=None, bounds_respected=None, constraints_respected=None):
        """
        Re-implementation of your old filter_points method.
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



def store_results(x, fp_gld, fp_slack, gld_params, result_dict, experiment_key=None):
    # Legacy function
    x_key = otaf.common.bidirectional_string_to_array_conversion(x)
    x_dict = {"FP_GLD": fp_gld, "FP_SLACK":fp_slack, "GLD_PARAMS": gld_params}
    if experiment_key is None:
        if x_key in result_dict.keys():
            result_dict[x_key].update(x_dict)
        else :
            result_dict[x_key] = x_dict
    else :
        if experiment_key not in result_dict:
            result_dict[experiment_key] = {}
        if x_key in result_dict[experiment_key].keys():
            result_dict[experiment_key][x_key].update(x_dict)
        else:
            result_dict[experiment_key][x_key] = x_dict
