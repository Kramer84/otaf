# Validity and limitations

OTAF answers one question: can a set of rigid parts, with given small geometric deviations,
be assembled? The model is exact for that question only under the assumptions below. Read this page
before using a failure probability for a decision. The mathematical model is in {doc}`theory`,
the input keys in {doc}`input_format`.

## What is modelled

- **Rigid parts.** There is no elastic deformation, no clamping force, no friction, no
  assembly sequence. A configuration is either geometrically compatible or not.
- **Perfect features with rigid-body deviations.** A plane or a cylinder is displaced and tilted as a
  whole (`CONSTRAINTS_D`). Form errors (flatness, cylindricity, waviness) are not represented, and
  neither is the roughness of the contact.
- **A fixed contact topology.** The set of surfaces that touch (`INTERACTIONS`) and the loops that are
  closed are chosen by the user and are the same for every sample. The model does not detect that a
  different pair of faces comes into contact for large deviations.
- **Static feasibility.** The slack says whether a gap configuration exists; it does not say that the
  assembly process finds it.

## Small-displacement linearization

All matrices are expanded to first order in the deviations and gaps. Products of two small
variables are dropped. For a rotation $\theta$ and a lever arm $L$, the displacement perpendicular to
the arm is $L\sin\theta \approx L\theta$, and the dropped displacement along the arm is
$L(1-\cos\theta) \approx L\theta^2/2$. Orders of magnitude for $L = 100$ mm:

| $\theta$ (rad) | Dropped term $L\theta^2/2$ (mm) |
|---|---|
| 0.005 | 0.001 |
| 0.01 | 0.005 |
| 0.02 | 0.02 |
| 0.05 | 0.125 |

The model is valid when this term is small against the smallest clearance of the assembly.
Tolerances of a few tenths of a millimetre over parts of a hundred millimetres give rotations of
$10^{-3}$ to $10^{-2}$ rad, which is within range; wide tolerances on long lever arms, or
clearances of a few microns, are not. The check is yours: compare the rotations of the
solutions with the table above.

The solver bounds the gap variables at ±3 mm and ±π/4 rad (`get_gap_symbol_bounds`). A sample whose
compatible gaps fall outside these bounds is reported as infeasible; rotations near π/4 are far
outside the validity of the linearization anyway. The bounds are not adjustable through the input
dictionary.

## Supported contact types

| Contact | Status |
|---|---|
| plane–plane | Supported. The two planes must be parallel (within about 0.26 rad of the opposed normals, `are_planes_facing`) and overlap over a non-empty area; otherwise the pair is skipped with a warning, or `NoPlanarOverlapException` is raised if they face each other without overlap. The contact is written at the vertices of the overlap, which is exact for the linear model. |
| cylinder–cylinder | Supported for a pin in a hole: the axes must be nominally collinear (`NonConcentricCylindersException` otherwise), the radii different (the same radius raises `ValueError`), and the extents must overlap (`NoAxialOverlapException`). The contact is written at the two ends of the axial overlap. Conflicting or impossible `SURFACE_DIRECTION` raise `ConflictingSurfaceDirectionsException` or `CylindricalInterferenceGeometricException`. |
| plane–cylinder | **Not supported yet.** A `NotImplementedError` is raised when the dictionary is validated. |
| cone, sphere | **Not supported.** They have no deviation or gap model; `TYPE` accepts only `plane` and `cylinder` (`UnsupportedSurfaceTypeError`). |
| Any other combination | `NotImplementedError`. |

Consequences for the geometries that can be modelled: planar contacts and pins in holes, with
the pin axis parallel to the hole axis nominally. A pin resting on a plane, a ball in a cone, a
cylinder rolling on a plane, or non-parallel nominal axes cannot be represented.

## Linearization error of the polygon

The circle of radial clearance is replaced by a regular polygon with $N$ sides
(`circle_resolution`). The radial extent of the polygon relative to the circle varies with the
direction, from the apothem (the middle of an edge) to the vertex:

| Strategy | Radial extent relative to the circle |
|---|---|
| `CIRCUMSCRIBED` | from $1$ to $1/\cos(\pi/N)$ |
| `INSCRIBED` | from $\cos(\pi/N)$ to $1$ |
| `MEAN` | from $\tfrac12(1+\cos(\pi/N))$ to $\tfrac12(1+\cos(\pi/N))/\cos(\pi/N)$ |

Numerically:

| $N$ | `CIRCUMSCRIBED` | `INSCRIBED` | `MEAN` |
|---|---|---|---|
| 8 | 0 to +8.24 % | −7.61 % to 0 | −3.81 % to +4.12 % |
| 16 | 0 to +1.96 % | −1.92 % to 0 | −0.96 % to +0.98 % |
| 32 | 0 to +0.48 % | −0.48 % to 0 | −0.24 % to +0.24 % |
| 64 | 0 to +0.12 % | −0.12 % to 0 | −0.06 % to +0.06 % |

The error acts on the clearance, so for a radial clearance of 0.1 mm the worst-case error of the
circumscribed polygon is 0.008 mm with 8 sides and 0.0001 mm with 64 sides. The default of
`InterfaceLoopHandling` is `circle_resolution=8`; the pin models use 64. Choose the resolution so
that this error is small against the quantities you want to resolve, and run `INSCRIBED` and
`CIRCUMSCRIBED` to bracket the failure probability: the first never under-estimates it and the second never
over-estimates it (see {doc}`theory`). The more facets, the more rows in the linear program (one set
of $N$ per contact section), and therefore the longer each solve.

Planes have no such error: their constraints are exactly linear.

## Input checks

The input dictionary is validated as a whole when `AssemblyDataProcessor` is created, against the
pydantic models of `otaf.schema` (see {doc}`input_format`): labels, types, frames, the keys each
surface needs, the references of `INTERACTIONS` and `LOOPS`, and the `CONSTRAINTS_G` of both sides
of each contact. What it does not check:

- whether the loops close and are independent: that is only seen when the loops are expanded or
  the system is solved;
- whether every contact is crossed by a loop: this needs the expanded loops, so it is checked when
  `InterfaceLoopHandling` is built, and a contact that no loop crosses raises an
  `UncoveredContactWarning` (it gets no contact condition);
- the geometry of the contacts (parallel planes, concentric cylinders, overlap): checked when the
  contact points are computed, see the table above;
- the units and the direction convention of the plane normals.

## Statistics

The slack is deterministic. Failure probabilities additionally rely on the distribution assumed
for the deviations (`otaf.distribution`, `otaf.tolerances`) and on the accuracy of the sampling
method. Monte Carlo results come with a sampling error that must be reported with them, and the
surrogate-based studies add the error of the surrogate. None of these are checked by the
modelling code.
