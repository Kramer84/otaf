# -*- coding: utf-8 -*-
__author__ = "Kramer84"
__all__ = [
    "ModifiedBasinHopping",
]

import numpy as np
from scipy.optimize import (
    basinhopping,
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
    minimize,
)
import otaf


class ModifiedBasinHopping:
    def __init__(
        self,
        func,
        x0,
        constraints=None,
        niter=100,
        T=1.0,
        stepsize=0.5,
        minimizer_kwargs=None,
        interval=50,
        disp=False,
        niter_success=None,
        seed=None,
        jac=False,
        jac_threshold=1e-5,
    ):
        self.func = func
        self.x0 = x0
        self.constraints = constraints or []
        self.niter = niter
        self.T = T
        self.stepsize = stepsize
        self.minimizer_kwargs = minimizer_kwargs or {}
        self.interval = interval
        self.disp = disp
        self.niter_success = niter_success
        self.seed = seed
        self.jac = jac
        self.jac_threshold = jac_threshold

    def run(self):
        result = basinhopping(
            self._wrapped_func,
            self.x0,
            niter=self.niter,
            T=self.T,
            stepsize=self.stepsize,
            minimizer_kwargs=self.minimizer_kwargs,
            take_step=self.take_step,
            accept_test=self.accept_test,
            callback=self.callback,
            interval=self.interval,
            disp=self.disp,
            niter_success=self.niter_success,
            seed=self.seed,
        )
        return result

    def _wrapped_func(self, x, *args):
        if self.jac:
            f, j = self.func(x, *args)
            return f, j
        return self.func(x, *args)

    def take_step(self, x):
        """Takes a random step and projects it back into the feasible space if constraints are violated."""
        if self.jac:
            f, jacobian = self.func(x)
            if np.linalg.norm(jacobian) > self.jac_threshold:
                # Move in the direction of negative gradient (steepest descent)
                step = -self.stepsize * jacobian / np.linalg.norm(jacobian)
                new_x = x + step
            else:
                new_x = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
        else:
            new_x = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))

        # Ensure the new point satisfies the constraints via projection
        new_x = self._project_to_constraints(new_x)
        return new_x

    def accept_test(self, f_new, x_new, f_old, x_old):
        """Metropolis criterion for acceptance of the new point."""
        return f_new < f_old or np.random.uniform() < np.exp((f_old - f_new) / self.T)

    def callback(self, x, f, accept):
        """Optional callback function to store minima."""
        if accept:
            if not hasattr(self, "minima"):
                self.minima = []
            self.minima.append((x, f))

    def _project_to_constraints(self, x):
        """Project point onto the feasible region defined by constraints."""
        if not self.constraints:
            return x

        def objective(delta):
            return np.sum(delta**2)  # Minimize the squared difference

        updated_constraints = []
        for constraint in self.constraints:
            if isinstance(constraint, dict):
                if constraint["type"] == "eq":
                    updated_constraints.append(
                        {"type": "eq", "fun": lambda delta, fun=constraint["fun"]: fun(x + delta)}
                    )
                elif constraint["type"] == "ineq":
                    updated_constraints.append(
                        {"type": "ineq", "fun": lambda delta, fun=constraint["fun"]: fun(x + delta)}
                    )
            elif isinstance(constraint, LinearConstraint):
                A = constraint.A
                lb = constraint.lb - np.dot(A, x)
                ub = constraint.ub - np.dot(A, x)
                updated_constraints.append(LinearConstraint(A, lb, ub))
            elif isinstance(constraint, NonlinearConstraint):
                updated_constraints.append(
                    NonlinearConstraint(
                        lambda delta: constraint.fun(x + delta), constraint.lb, constraint.ub
                    )
                )

        delta0 = np.zeros_like(x)
        result = minimize(objective, delta0, constraints=updated_constraints)

        if result.success:
            return x + result.x
        else:
            raise ValueError("Projection to constraints failed.")
