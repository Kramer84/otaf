# Theory: system of constraints and slack variable

OTAF decides whether a rigid, over-constrained assembly can be assembled for a given set
of geometric deviations. The question is turned into a linear program in the relative displacements
(*gaps*) between the parts. The value of the program, the *slack*, is negative when the assembly
cannot be built. The page follows the order of the pipeline in `CODEBASE.md`:
deviations and gaps, compatibility loops, interface conditions, slack. For the keys of the input,
see {doc}`input_format`; for what the model does not cover, {doc}`limitations`.

## Deviations and gaps

Every feature (plane or cylinder) of a part is placed by a rigid-body displacement from its
nominal position, the **deviation** of the feature. Every pair of features that can touch
has a **gap**: the rigid displacement of one feature relative to the other that the contact allows.
Both are small displacements, written as a 4×4 homogeneous matrix close to the identity:

$$
\mathbf{M} = \mathbf{I} + u\,\mathbf{E}_1 + v\,\mathbf{E}_2 + w\,\mathbf{E}_3
  + \alpha\,\mathbf{E}_4 + \beta\,\mathbf{E}_5 + \gamma\,\mathbf{E}_6 ,
$$

where $u, v, w$ are translations along the local $x, y, z$ axes of the feature, $\alpha,
\beta, \gamma$ are rotations about them and $\mathbf{E}_k$ are the generators of
`otaf.constants.BASIS_DICT`. The variables are called `u_d_<k>`, `alpha_d_<k>`, … for
deviations and `u_g_<k>`, … for gaps, where $k$ is the index of the feature or of the gap. Which of
the six components exist is set by `CONSTRAINTS_D` (deviations), `CONSTRAINTS_G`,
`BLOCK_*_G` (gaps) and `GLOBAL_CONSTRAINTS` (2D or 3D).

The deviations $\mathbf{d}$ are the uncertain inputs (they follow the tolerance model of the
study). The gaps $\mathbf{g}$ are unknowns that the assembly is free to choose.

## Compatibility loops: equalities

A compatibility loop follows a closed chain of the assembly: from a point of a part to a feature
of the same part, through a contact to the next part, and so on until it comes back. Going once around, the product of the matrices must be the identity:

$$
\mathbf{T}_1\,\mathbf{D}_1\,\mathbf{G}_1\,\mathbf{D}_2^{-1}\,\mathbf{T}_2 \cdots = \mathbf{I},
$$

with $\mathbf{T}$ the nominal transforms inside a part, $\mathbf{D}$ the deviations of
the features and $\mathbf{G}$ the gaps (see {doc}`input_format` for the loop syntax). Expanding to first order in the small
variables (`FirstOrderMatrixExpansion`; products of two small terms are dropped) gives one set of
linear equations per loop,

$$
\mathbf{A}_{\mathrm{eq}}^{d}\,\mathbf{d} + \mathbf{A}_{\mathrm{eq}}^{g}\,\mathbf{g} + \mathbf{k}_{\mathrm{eq}} = \mathbf{0},
$$

where $\mathbf{k}_{\mathrm{eq}}$ collects what the nominal geometry contributes (for instance the
nominal clearance). These are the attributes `A_eq_Def`, `A_eq_Gap` and `K_eq` of
`SystemOfConstraintsAssemblyModel`. For given deviations, these equalities usually leave
some gap components free; the inequalities below decide whether a choice of these free
components exists that respects every contact.

## Interface conditions: inequalities

The loops close the chain, but not the parts' bodies: they could interpenetrate. Contact is
expressed by requiring that the two facing features do not overlap at the places where they could touch.

**Planes.** Two parallel, facing planes are paired at the vertices of their overlapping area
(the segment ends in 2D, the convex polygon in 3D). For each pair, the displacement of the
point of one plane relative to the other, along the common normal, must not be negative:

$$
n_i(\mathbf{d}, \mathbf{g}) = \mathbf{b}_i^{d}\,\mathbf{d} + \mathbf{b}_i^{g}\,\mathbf{g} + k_i \ge 0 .
$$

It is linear, so a point of the overlap region cannot be more constraining than the vertices:
checking the vertices is sufficient for planes.

**Cylinders.** A pin in a hole is paired at the two ends of the axial overlap of the two cylinders.
In the cross-section there, the axis of the pin is offset from the axis of the hole by
$(y, z)$, a linear function of $\mathbf{d}$ and $\mathbf{g}$. The pin does not touch the hole if

$$
\sqrt{y^2 + z^2} \le c, \qquad c = R_{\mathrm{hole}} - R_{\mathrm{pin}}
  \;(+\; \tfrac{1}{2}(d_{\mathrm{hole}} - d_{\mathrm{pin}}) \text{ with diameter deviations}),
$$

the radial clearance $c$. This is not linear. It is replaced by $N$ half-planes (see below):

$$
r_f\,c - (y\cos\theta_k + z\sin\theta_k) \ge 0, \qquad \theta_k = \frac{2\pi k}{N}, \quad k = 0, \dots, N-1 .
$$

All the conditions are collected as

$$
\mathbf{A}_{\mathrm{ub}}^{d}\,\mathbf{d} + \mathbf{A}_{\mathrm{ub}}^{g}\,\mathbf{g} + \mathbf{k}_{\mathrm{ub}} \ge \mathbf{0}.
$$

In the interface conditions the deviations enter only through the diameters of cylinders
(`DIAMETER_DEVIATION`, variables `d_d_<k>`); the position and orientation deviations act through the
loops above. For the planes of the pocket example, $\mathbf{A}_{\mathrm{ub}}^{d}$ is zero.

## Slack variable

For given deviations $\mathbf{d}$, the assembly can be built if some $\mathbf{g}$ satisfies the equalities and
all the inequalities. Asking for this directly gives a feasibility problem that has no value when it fails.
OTAF adds one variable, the **slack** $s$, to every inequality and maximizes it
(`SystemOfConstraintsAssemblyModel.embedOptimizationVariable`):

$$
s^\star(\mathbf{d}) = \max_{\mathbf{g},\, s} \; s
\quad\text{subject to}\quad
\begin{cases}
\mathbf{A}_{\mathrm{eq}}^{g}\,\mathbf{g} = -\mathbf{A}_{\mathrm{eq}}^{d}\,\mathbf{d} - \mathbf{k}_{\mathrm{eq}},\\[2pt]
\mathbf{A}_{\mathrm{ub}}^{g}\,\mathbf{g} + \mathbf{A}_{\mathrm{ub}}^{d}\,\mathbf{d} + \mathbf{k}_{\mathrm{ub}} - s\,\mathbf{1} \ge \mathbf{0},\\[2pt]
\mathbf{g}_{\min} \le \mathbf{g} \le \mathbf{g}_{\max}.
\end{cases}
$$

Equivalently, $s^\star$ is the best achievable value, over the gaps, of the **smallest margin**
among all the interface conditions. Reading it:

- $s^\star \ge 0$: there is a gap configuration with all contacts respected; the assembly can be built.
  The value is the margin of the tightest contact, in the units of the conditions (mm for planes;
  mm of clearance for cylinders, scaled by $r_f$).
- $s^\star < 0$: no such configuration exists; the assembly fails. The magnitude says by how
  much the most violated condition has to be relaxed.

The program always has a solution as long as the equalities can be satisfied inside the bounds of the
gap variables (`get_gap_symbol_bounds`: ±3 mm for translations, ±π/4 for rotations; $s$ is
unbounded), whereas the plain feasibility problem would simply be infeasible. This is what
makes $s^\star$ usable as a response for sampling, FORM or surrogate
models. It is solved with `scipy.optimize.linprog` (or `milp` for the batch functions of
`otaf.uncertainty`) from the arrays returned by calling the model on the deviations.

$s^\star(\mathbf{d})$ is the optimal value of a linear program whose right-hand side is linear in
$\mathbf{d}$, so it is a **concave, piecewise linear** function of the deviations. The probability of failure of the
assembly is

$$
P_f = \mathbb{P}\bigl(s^\star(\mathbf{D}) < 0\bigr),
$$

for random deviations $\mathbf{D}$. As an example, for the 2D pocket of {doc}`input_format`
the slack of the nominal assembly is exactly half the clearance, $s^\star = (X_2 - X_1)/2$,
and `tests/test_physical_invariants.py` checks that identity on random samples.

## Circle linearization strategies

The polygon replacing the circular clearance zone has $N$ sides (`circle_resolution`). The
factor $r_f$ applied to the clearance sets whether the polygon touches the circle from outside,
from inside or in between (`otaf.geometry.LinearizationStrategy`, `circle_linearization_factor`):

| Strategy | $r_f$ | Polygon | Effect on the failure probability |
|---|---|---|---|
| `CIRCUMSCRIBED` (default) | $1$ | contains the circle, edges tangent to it | optimistic: $P_f$ too low |
| `INSCRIBED` | $\cos(\pi/N)$ | inside the circle, vertices on it | conservative: $P_f$ too high |
| `MEAN` | $\tfrac{1}{2}\bigl(1 + \cos(\pi/N)\bigr)$ | in between | no bound on either side |

Because the feasible region of the polygon is nested with the circle, the decision "can be
assembled" is ordered: if the assembly is feasible with the inscribed polygon, it is feasible with
the circle, and if it is feasible with the circle, it is feasible with the circumscribed polygon.
Hence $P_f^{\text{inscribed}} \ge P_f^{\text{circle}} \ge P_f^{\text{circumscribed}}$,
and running both strategies brackets the failure probability of the circle. The *values* of
$s^\star$ differ between strategies and are not comparable, only their sign is.

The size of the error is given in {doc}`limitations`. The reference models `model3` and `model4`
use the same three strategies with a hand-written constraint matrix, and the automatic path
reproduces `model4` to $10^{-8}$ for each of them (`tests/test_cylinder_interfaces.py`).
