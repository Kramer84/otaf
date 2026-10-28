# Input format of the assembly dictionary

The whole assembly is described by one nested dictionary, the *system data*. It is
validated by `otaf.AssemblyDataProcessor`, which is the first step of the modelling
pipeline (see {doc}`theory`), against the pydantic models of `otaf.schema`
(`SystemData`, `Part`, `Surface`, `ExtentLocal`, `Loops`). Lengths are in millimetres and
angles in radians.

```text
system_data
├── "PARTS"               {part id: {surface id: surface dictionary}}
├── "LOOPS"               {"COMPATIBILITY": {loop name: loop string}}
└── "GLOBAL_CONSTRAINTS"  "3D" | "2D_NX" | "2D_NY" | "2D_NZ"
```

The three top-level keys are required (`MissingKeyError` otherwise); other keys are kept and
ignored. The dictionary is copied, so the one you pass is never modified; read the completed data
back from `AssemblyDataProcessor.system_data`. See {ref}`validation` for the errors.

## Labels

| Element | Rule | Example |
|---|---|---|
| Part id | string of an integer | `"1"` |
| Surface id | lowercase ASCII letters only, unique within the part | `"a"`, `"b"` |
| Point label | upper-case prefix + integer suffix; the same prefix for all points of a surface (conventionally the upper-case surface id); suffixes unique (`A0` and `A00` clash) | `"A0"`, `"A1"` |
| Origin point | the label whose suffix is all zeros | `"A0"` |
| Interaction | `P<part id><surface id>` | `"P2a"` |

A part with a single surface is treated as a single-feature part: no deviation matrix is
generated for it in the loops.

## Surface dictionary

`PARTS[part][surface]` is a dictionary with the following keys (model `otaf.schema.Surface`).
Other keys, such as `META` or `CAD_ID`, are kept and ignored.

| Key | Required | Allowed values | Meaning |
|---|---|---|---|
| `TYPE` | always | `"plane"` or `"cylinder"` (`otaf.constants.SUPPORTED_SURFACE_TYPES`) | Geometry of the feature. `"cone"` and `"sphere"` have no deviation or gap model and are rejected (`UnsupportedSurfaceTypeError`, see {doc}`limitations`). |
| `FRAME` | always | 3×3 array, orthonormal, determinant +1 (to `otaf.constants.FRAME_ATOL`) | Local frame. **Columns** are the local axes. Local x (first column) is the plane normal, or the cylinder axis. |
| `ORIGIN` | when `POINTS` is absent | 3 floats | Origin of the surface frame. If `POINTS` is given, it is taken from the point whose label ends in zeros. |
| `POINTS` | optional | `{label: (x, y, z)}` | Points on the feature, in global coordinates: the contour of a plane (in 3D, 3 or more coplanar points; in 2D, the origin and the ends of the segment), or points on a cylinder axis. Must contain the origin. Labels and coordinates must be unique. |
| `CONTOUR_GLOBAL` | optional, instead of `POINTS` | array of shape (n, 3) | Contour points in global coordinates. Labelled `<SURFACE>1 … <SURFACE>n` after the origin `<SURFACE>0`. Coplanarity is checked in 3D. |
| `RADIUS` | cylinders with `INTERACTIONS` | number > 0 | Nominal radius. |
| `EXTENT_LOCAL` | cylinders with `INTERACTIONS` | `{"x_min": number, "x_max": number}`, no other key | Axial extent along the local x axis, from `ORIGIN`. `x_max >= x_min`. |
| `INTERACTIONS` | when `POINTS` is absent; required for contact | list of `P<part><surface>`, existing surfaces of the same `TYPE` | Surfaces this one can touch. List the interaction on both surfaces. |
| `CONSTRAINTS_D` | optional (default `["NONE"]`) | list with one of the values below | Which deviations the feature has. |
| `CONSTRAINTS_G` | optional (default `["FLOATING"]`) | list with one of the values below, the same on both surfaces of a contact | Which relative motions the contact blocks. |
| `BLOCK_TRANSLATIONS_G`, `BLOCK_ROTATIONS_G` | optional (default `""`) | string of the axes `x`, `y`, `z`, e.g. `"xz"` | Extra blocked gap translations/rotations, added to those of `CONSTRAINTS_G`. |
| `SURFACE_DIRECTION` | optional, cylinders only | `"centripetal"` or `"centrifugal"` | `"centripetal"`: the material is outside the cylinder (a hole). `"centrifugal"`: the material is inside (a pin). Deduced from the radii when absent on both cylinders of a pair. |
| `DIAMETER_DEVIATION` | optional, cylinders only | `True` / `False` | Whether the diameter is a deviation variable `d_d_<k>`. |

When `POINTS` is absent, `ORIGIN` and `INTERACTIONS` must be present (`MissingKeyError`). When
`POINTS` is given, it must contain the origin point (`MissingOriginPointError`) and `ORIGIN` is
taken from it. `FRAME` is checked in both cases.

### `CONSTRAINTS_D`: deviations of a feature

The key is the value of `TYPE` combined with the entry, as in
`otaf.constants.SURF_TYPE_TO_DEVIATION_DOF`. The entry is case-insensitive.

| Entry | Plane | Cylinder |
|---|---|---|
| `"NONE"` | translation x, rotations y, z | translations y, z, rotations y, z |
| `"PERFECT"` | none | none |
| `"TRANSLATION"` | translation x | translations y, z |
| `"ROTATION"` | rotations y, z | rotations y, z |

The axes are those of the feature frame. In the 2D modes, the translation along the removed axis
and the rotations about the other two axes are dropped (see `GLOBAL_CONSTRAINTS`).
The feature's form defects are not modelled: the feature is a perfect plane or cylinder
with position and orientation deviations only.

### `CONSTRAINTS_G`: relative motion at a contact

Both surfaces of a pair must carry the **same** entry (case-insensitive). A surface without the
key counts as `"FLOATING"`, so if one has `"SLIDING"` and the other none,
`InconsistentContactConstraintsError` is raised.

| Entry | Plane–plane: blocked gap motions | Cylinder–cylinder: blocked gap motions |
|---|---|---|
| `"FLOATING"` | none | none |
| `"SLIDING"` | translation x, rotations y, z (slides in the plane, rotates about the normal) | translations y, z, rotations y, z (slides and turns about the axis) |
| `"FIXED"` | all six | all six |

Gap variables are named `u_g, v_g, w_g` (translations x, y, z) and `alpha_g, beta_g, gamma_g`
(rotations about x, y, z), followed by an index. Blocked components are removed from the
unknowns. Pins are typically free to slide along their axis in the hole, hence
`BLOCK_TRANSLATIONS_G` / `BLOCK_ROTATIONS_G` on a hole to block the axial motion, as in the
pin/hole example below.

## `GLOBAL_CONSTRAINTS`

| Value | Meaning |
|---|---|
| `"3D"` | Full 6 degrees of freedom per deviation and gap. |
| `"2D_NZ"` | Planar problem in the x–y plane: no translation along z, no rotations about x and y. |
| `"2D_NY"` | Planar problem in the x–z plane. |
| `"2D_NX"` | Planar problem in the y–z plane. |

Any other value raises `InvalidGlobalConstraintError`.

## `LOOPS`

`LOOPS["COMPATIBILITY"]` maps a name to a loop string: surface points joined by `" -> "`.
Each element is `P<part><surface><point>`, for example `P1cC0`.

```text
"P1cC0 -> P2cC0 -> P2aA0 -> P1aA0"
```

Moving from one element to the next means:

- **same part**: a rigid transform inside the part, from one feature point to another;
- **different parts**: a contact. The loop passes the deviation of the first surface, the gap
  between the two surfaces, and the inverse deviation of the second surface.

The loop must close:

- If the first two elements belong to different parts, the loop must start and end on the same part;
  it is closed by a transform inside that part.
- If the first two elements belong to the same part, the last element must be on a different part;
  the loop is closed by a gap between the last and first elements.
- A loop with two elements is a direct contact between two different parts.

Each element must match `P<digits><letters><LETTERS><digits>` (`SystemDataValidationError`
otherwise) and name an existing part and surface (`PartNotFoundError`, `SurfaceNotFoundError`);
a loop that stays in one part raises a `ValueError` when the loops are expanded. One loop is
needed per independent closed chain of the assembly. Every contact must be crossed by some loop:
the contact conditions are generated for the gaps that appear in the loops, so a contact in
`INTERACTIONS` that no loop crosses gets no condition, and `InterfaceLoopHandling` warns about it
(`otaf.exceptions.UncoveredContactWarning`).

## Example 1: two parts in 2D

This is the 4-degree-of-freedom model `otaf.example_models.model1`: part 1 sits in a pocket
of part 2, with `X1 = 99.8` mm of width against `X2 = 100.0` mm.

```python
import numpy as np
import otaf

X1, X2, X3 = 99.8, 100.0, 10.0
x, y, z = np.eye(3)


def part(width, partner, frames):
    """Plane features a (left wall), b (right wall) and c (bottom) of a part."""
    points = {
        "a": {"A0": (0, X3 / 2, 0), "A1": (0, X3, 0), "A2": (0, 0, 0)},
        "b": {"B0": (width, X3 / 2, 0), "B1": (width, X3, 0), "B2": (width, 0, 0)},
        "c": {"C0": (width / 2, 0, 0), "C1": (0, 0, 0), "C2": (width, 0, 0)},
    }
    return {
        face: {
            "FRAME": frames[face],
            "POINTS": {k: np.array(v, dtype=float) for k, v in points[face].items()},
            "TYPE": "plane",
            "INTERACTIONS": [f"P{partner}{face}"],
            "CONSTRAINTS_D": ["NONE"] if face == "b" else ["PERFECT"],
            "CONSTRAINTS_G": ["SLIDING"] if face == "c" else ["FLOATING"],
        }
        for face in "abc"
    }


frames_1 = {"a": np.array([-x, -y, z]), "b": np.eye(3), "c": np.array([-y, x, z])}
frames_2 = {"a": np.eye(3), "b": np.array([-x, -y, z]), "c": np.array([y, -x, z])}

system_data = {
    "PARTS": {
        "1": part(X1, "2", frames_1),
        "2": part(X2, "1", frames_2),
    },
    "LOOPS": {
        "COMPATIBILITY": {
            "L0": "P1cC0 -> P2cC0 -> P2aA0 -> P1aA0",
            "L1": "P1cC0 -> P2cC0 -> P2bB0 -> P1bB0",
        }
    },
    "GLOBAL_CONSTRAINTS": "2D_NZ",
}
```

Each part has three planes. The faces `a` and `b` are the side walls, `c` the bottom.
Only the walls `b` have deviations (`"NONE"`: after the `2D_NZ` reduction, a translation along
the normal and a rotation about z remain); the other faces are taken as perfect
(`"PERFECT"`). The bottom faces slide on each other (`"SLIDING"`), the walls float.

The loops start at the middle of the bottom of part 1 (`P1cC0`), go to the bottom of part 2
(`P2cC0`), then to the wall `a` of part 2 (`P2aA0`) and come back to the wall `a` of part 1 (`P1aA0`),
closing through a gap. `L1` does the same for the walls `b`.

Build the model and solve the nominal assembly:

```python
from scipy.optimize import linprog

adp = otaf.AssemblyDataProcessor(system_data)
adp.generate_expanded_loops()
clh = otaf.CompatibilityLoopHandling(adp)
ilh = otaf.InterfaceLoopHandling(adp, clh)
soc = otaf.SystemOfConstraintsAssemblyModel(
    clh.get_compatibility_expressions(), ilh.get_interface_expressions()
)
soc.embedOptimizationVariable()  # adds the slack variable s

c, a_ub, b_ub, a_eq, b_eq, bounds = soc(np.zeros(soc.nD))  # no deviation
result = linprog(c, A_ub=a_ub, b_ub=b_ub[:, 0], A_eq=a_eq, b_eq=b_eq[:, 0], bounds=bounds)
slack = -result.fun
print(round(slack, 6))
```

The slack of the nominal assembly is half the clearance, `(X2 - X1) / 2 = 0.1` mm.
It is positive, so the nominal assembly can be assembled. The deviation variables of
`soc.deviation_symbols` are the inputs sampled by the uncertainty study.

(validation)=
## Validation

`AssemblyDataProcessor` validates the dictionary before using it. The same check can be run on
its own, and a model built once can be passed in place of the dictionary:

```python
from otaf.schema import SystemData, validate_system_data

model = validate_system_data(system_data)  # SystemData; system_data is not modified
assert model.PARTS["1"].root["c"].CONSTRAINTS_G == ["SLIDING"]
assert model.contact_pairs()[0] == (("1", "a"), ("2", "a"))
adp_from_model = otaf.AssemblyDataProcessor(model)
json_schema = SystemData.model_json_schema()  # e.g. for a CAD converter or an editor

broken = {**system_data, "GLOBAL_CONSTRAINTS": "2D"}
try:
    validate_system_data(broken)
except otaf.exceptions.InvalidGlobalConstraintError as error:
    print(error)
```

Errors that have their own class in `otaf.exceptions` are raised with it, their message
prefixed by the location of the problem (`PARTS -> 1 -> a -> POINTS: ...`):

| Problem | Exception |
|---|---|
| missing key (top level, `TYPE`, keys required by the rules above) | `MissingKeyError` (`MissingSurfaceTypeKeyError` for `TYPE`) |
| part or surface label | `InvalidPartLabelError`, `InvalidSurfaceLabelError` |
| `TYPE` not `plane` or `cylinder` | `UnsupportedSurfaceTypeError` |
| point labels or duplicate points | `LabelPatternError`, `LabelPrefixError`, `UniqueLabelSuffixError`, `NonUniqueCoordinatesError` |
| no origin point in `POINTS` | `MissingOriginPointError` |
| `INTERACTIONS` entry format, unknown target | `InvalidInteractionFormatError`, `PartNotFoundError`, `SurfaceNotFoundError` |
| `SURFACE_DIRECTION` value | `InvalidSurfaceDirectionError` |
| `GLOBAL_CONSTRAINTS` value | `InvalidGlobalConstraintError` |
| different `CONSTRAINTS_G` on the two sides of a contact | `InconsistentContactConstraintsError` |
| plane–cylinder contact | `NotImplementedError` |
| anything else (`FRAME` not a rotation, negative `RADIUS`, `CONSTRAINTS_D` not a one-entry list, ...) | `SystemDataValidationError` |

When several problems are found, the first is raised and chained to a
`SystemDataValidationError` that lists all of them (`error.__cause__.messages`).

## Example 2: a pin in two plates (3D)

A pin through a hole in each of two stacked plates, as in
`otaf.example_models.model4_auto` with one pin (`get_assembly_data(NX=1, NY=1)`). The plates
are 50 × 50 mm and 30 mm thick, the holes have a diameter of 20 mm, the pin 19.8 mm, and
the pin is 60 mm long. Plate `"0"` is below z = 0 and plate `"1"` above, they touch on the
plane z = 0. The pin (part `"2"`) crosses both.

```python
frame_down = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)  # local x = -z
frame_up = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)  # local x = +z
contour = {"A0": (0, 0, 0), "A1": (50, 0, 0), "A2": (50, 50, 0), "A3": (0, 50, 0)}
axis_point = np.array([25.0, 25.0, 0.0])


def plate(frame, other):
    return {
        "a": {  # the plane where the plates touch
            "FRAME": frame,
            "POINTS": {k: np.array(v, dtype=float) for k, v in contour.items()},
            "TYPE": "plane",
            "INTERACTIONS": [f"P{other}a"],
            "CONSTRAINTS_D": ["PERFECT"],
            "CONSTRAINTS_G": ["SLIDING"],
        }
    }


pin_system_data = {
    "PARTS": {"0": plate(frame_down, "1"), "1": plate(frame_up, "0")},
    "LOOPS": {"COMPATIBILITY": {"L1": "P0aA0 -> P0bB0 -> P2bB0 -> P1bB0 -> P1aA0"}},
    "GLOBAL_CONSTRAINTS": "3D",
}
pin_system_data["PARTS"]["2"] = {  # the pin: its surface is "b"
    "b": {
        "FRAME": frame_down,
        "ORIGIN": axis_point,
        "TYPE": "cylinder",
        "RADIUS": 19.8 / 2,
        "EXTENT_LOCAL": {"x_max": 30.0, "x_min": -30.0},
        "INTERACTIONS": ["P0b", "P1b"],
        "SURFACE_DIRECTION": "centrifugal",
        "CONSTRAINTS_D": ["PERFECT"],
        "DIAMETER_DEVIATION": True,
    }
}
pin_system_data["PARTS"]["0"]["b"] = {  # the hole of plate 0
    "FRAME": frame_down,
    "ORIGIN": axis_point + np.array([0.0, 0.0, -15.0]),
    "TYPE": "cylinder",
    "RADIUS": 20.0 / 2,
    "EXTENT_LOCAL": {"x_max": 15.0, "x_min": -15.0},
    "INTERACTIONS": ["P2b"],
    "SURFACE_DIRECTION": "centripetal",
    "DIAMETER_DEVIATION": True,
    "BLOCK_ROTATIONS_G": "x",
    "BLOCK_TRANSLATIONS_G": "x",
}
pin_system_data["PARTS"]["1"]["b"] = {  # the hole of plate 1
    "FRAME": frame_up,
    "ORIGIN": axis_point + np.array([0.0, 0.0, 15.0]),
    "TYPE": "cylinder",
    "RADIUS": 20.0 / 2,
    "EXTENT_LOCAL": {"x_max": 15.0, "x_min": -15.0},
    "INTERACTIONS": ["P2b"],
    "SURFACE_DIRECTION": "centripetal",
    "DIAMETER_DEVIATION": True,
}
```

- The two plates are single-plane parts apart from the holes; their touching plane `a` has no
  deviation (`"PERFECT"`) and slides.
- Each cylinder lists the surface it faces in `INTERACTIONS`. The pin lists both holes, each hole
  lists the pin.
- `RADIUS` gives the nominal radial clearance, `(20 - 19.8) / 2 = 0.1` mm. The `SURFACE_DIRECTION` is
  deduced from the radii when absent on both; here it is stated.
- `EXTENT_LOCAL` is measured from `ORIGIN` along the local x axis, which points along the pin
  axis (here `-z` for `frame_down`). The contact sections of a pin and a hole are the ends of the
  axial overlap of their extents.
- `DIAMETER_DEVIATION: True` adds one diameter variable `d_d_<k>` per cylinder to the deviations.
- `BLOCK_TRANSLATIONS_G` and `BLOCK_ROTATIONS_G` of `"x"` on the first hole block the
  axial translation and the rotation about the axis between the pin and plate 0. Neither
  changes the contact, so blocking them in one gap keeps the system determined; the other
  hole is left free.
- The loop `L1` goes from the origin of the plate-0 contact plane to the hole of plate 0 (`P0bB0`),
  to the pin (`P2bB0`), to the hole of plate 1 (`P1bB0`) and back to the contact plane
  (`P1aA0`). With more pins, one loop is added per pin.

The circle of radial clearance is replaced by a polygon, chosen when the interface constraints
are built:

```python
from otaf.geometry import LinearizationStrategy

pin_adp = otaf.AssemblyDataProcessor(pin_system_data)
pin_adp.generate_expanded_loops()
pin_clh = otaf.CompatibilityLoopHandling(pin_adp)
pin_ilh = otaf.InterfaceLoopHandling(
    pin_adp,
    pin_clh,
    circle_resolution=64,
    linearization=LinearizationStrategy.CIRCUMSCRIBED,
)
pin_soc = otaf.SystemOfConstraintsAssemblyModel(
    pin_clh.get_compatibility_expressions(), pin_ilh.get_interface_expressions()
)
pin_soc.embedOptimizationVariable()
c, a_ub, b_ub, a_eq, b_eq, bounds = pin_soc(np.zeros(pin_soc.nD))
result = linprog(c, A_ub=a_ub, b_ub=b_ub[:, 0], A_eq=a_eq, b_eq=b_eq[:, 0], bounds=bounds)
pin_slack = -result.fun
print(round(pin_slack, 6))
```

The nominal slack is the radial clearance, 0.1 mm. See {doc}`theory` for the strategies and
{doc}`limitations` for the error of the polygon.
