"""Pydantic models of the assembly dictionary (the *system data*).

The models validate a plain dictionary up front: types, labels, frames, the keys
each surface needs, references between surfaces and loops. They do not convert
the dictionary: `otaf.AssemblyDataProcessor` keeps working on (a copy of) the
dictionary it is given, so validation never changes a result.

Errors are reported with the exceptions of `otaf.exceptions` where one applies
(e.g. `~otaf.exceptions.InvalidPartLabelError`), their message prefixed by the
location of the problem. Other problems raise
`~otaf.exceptions.SystemDataValidationError`, which lists all of them.
"""

from __future__ import annotations

__author__ = "Kramer84"
__all__ = [
    "Surface",
    "ExtentLocal",
    "Part",
    "Loops",
    "SystemData",
    "validate_system_data",
    "validate_point_dict",
]

import re
from numbers import Real
from typing import Annotated, Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    PlainSerializer,
    PlainValidator,
    RootModel,
    ValidationError,
    WithJsonSchema,
    field_validator,
    model_validator,
)

import otaf.constants as otaf_constants
import otaf.exceptions as otaf_exceptions

_SURFACE_LABEL_PATTERN = re.compile(r"^[a-z]+$")
_AXES_PATTERN = re.compile(r"^[xyz]*$")

# Entries of CONSTRAINTS_D, from the keys of SURF_TYPE_TO_DEVIATION_DOF ("plane-none" -> "NONE").
_DEVIATION_CONSTRAINTS = sorted(
    {key.split("-", 1)[1].upper() for key in otaf_constants.SURF_TYPE_TO_DEVIATION_DOF}
)


# -----------------------------------------------------------------------------
# Value types
# -----------------------------------------------------------------------------


def _float_array(value: Any, shape: tuple, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype="float64")
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be an array of numbers of shape {shape}") from e
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _vector3(value: Any) -> np.ndarray:
    return _float_array(value, (3,), "a point")


def _frame(value: Any) -> np.ndarray:
    frame = _float_array(value, (3, 3), "FRAME")
    if not (
        np.allclose(frame @ frame.T, np.identity(3), atol=otaf_constants.FRAME_ATOL)
        and np.isclose(np.linalg.det(frame), 1.0, atol=otaf_constants.FRAME_ATOL)
    ):
        raise ValueError(
            "FRAME must be a rotation matrix (orthonormal columns, determinant +1); "
            f"got max |F F^T - I| = {np.abs(frame @ frame.T - np.identity(3)).max():.3g}, "
            f"det = {np.linalg.det(frame):.6g}"
        )
    return frame


def _contour(value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype="float64")
    except (TypeError, ValueError) as e:
        raise ValueError("CONTOUR_GLOBAL must be an array of numbers of shape (n, 3)") from e
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 1:
        raise ValueError(f"CONTOUR_GLOBAL must have shape (n, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("CONTOUR_GLOBAL must be finite")
    return array


def _real(value: Any) -> float:
    # bool is an int subclass; a radius of True is a mistake.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"expected a number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("expected a finite number")
    return value


def _positive_real(value: Any) -> float:
    value = _real(value)
    if value <= 0:
        raise ValueError(f"must be > 0, got {value}")
    return value


def _strict_bool(value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"expected True or False, got {value!r}")
    return bool(value)


def _to_list(array: np.ndarray) -> list:
    return array.tolist()


_NUMBER_ARRAY = {"type": "array", "items": {"type": "number"}}

Vector3 = Annotated[
    np.ndarray,
    PlainValidator(_vector3),
    PlainSerializer(_to_list, when_used="json"),
    WithJsonSchema({**_NUMBER_ARRAY, "minItems": 3, "maxItems": 3}),
]
"""Three finite floats, stored as a float64 array."""

Frame = Annotated[
    np.ndarray,
    PlainValidator(_frame),
    PlainSerializer(_to_list, when_used="json"),
    WithJsonSchema(
        {
            "type": "array",
            "items": {**_NUMBER_ARRAY, "minItems": 3, "maxItems": 3},
            "minItems": 3,
            "maxItems": 3,
            "description": "Rotation matrix; columns are the local axes.",
        }
    ),
]
"""3x3 rotation matrix (orthonormal, determinant +1)."""

Contour = Annotated[
    np.ndarray,
    PlainValidator(_contour),
    PlainSerializer(_to_list, when_used="json"),
    WithJsonSchema({"type": "array", "items": {**_NUMBER_ARRAY, "minItems": 3, "maxItems": 3}}),
]
"""Array of shape (n, 3)."""

Number = Annotated[float, PlainValidator(_real), WithJsonSchema({"type": "number"})]
PositiveNumber = Annotated[
    float, PlainValidator(_positive_real), WithJsonSchema({"type": "number", "exclusiveMinimum": 0})
]
StrictBool = Annotated[bool, PlainValidator(_strict_bool), WithJsonSchema({"type": "boolean"})]


# -----------------------------------------------------------------------------
# Point dictionaries
# -----------------------------------------------------------------------------


def validate_point_dict(point_dict: Dict[str, Any]) -> None:
    """Check the labels and coordinates of a ``POINTS`` dictionary.

    Parameters
    ----------
    point_dict : dict
        ``{label: (x, y, z)}``.

    Raises
    ------
    otaf.exceptions.LabelPatternError
        If a label does not match ``otaf.constants.SURF_POINT_PATTERN``.
    otaf.exceptions.LabelPrefixError
        If the labels do not all have the same upper-case prefix.
    otaf.exceptions.UniqueLabelSuffixError
        If two labels have the same integer suffix (``A0`` and ``A00``).
    otaf.exceptions.NonUniqueCoordinatesError
        If two points have the same coordinates.

    Examples
    --------
    >>> validate_point_dict({"A0": (0, 0, 0), "A1": (1, 0, 0)})
    >>> validate_point_dict({"A0": (0, 0, 0), "B1": (1, 0, 0)})
    Traceback (most recent call last):
    ...
    otaf.exceptions._exceptions.LabelPrefixError: Some labels in the point dict do not comply to the expected pattern, all prefixes should be identical.
    """
    labels = [str(label) for label in point_dict]
    if not all(otaf_constants.SURF_POINT_PATTERN.fullmatch(label) for label in labels):
        raise otaf_exceptions.LabelPatternError()
    if len({re.match(r"^[A-Z]+", label).group() for label in labels}) > 1:
        raise otaf_exceptions.LabelPrefixError()
    if len({int(re.search(r"[0-9]+$", label).group()) for label in labels}) < len(labels):
        raise otaf_exceptions.UniqueLabelSuffixError()
    point_to_labels = {}
    for label, point in point_dict.items():
        point_to_labels.setdefault(tuple(np.asarray(point, dtype="float64")), []).append(label)
    duplicates = {
        str(np.array(point)): labels for point, labels in point_to_labels.items() if len(labels) > 1
    }
    if duplicates:
        raise otaf_exceptions.NonUniqueCoordinatesError(duplicates)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class ExtentLocal(BaseModel):
    """Axial extent of a cylinder along its local x axis, measured from ``ORIGIN``.

    Attributes
    ----------
    x_min, x_max : float
        Ends of the extent, ``x_max >= x_min``.
    """

    model_config = ConfigDict(extra="forbid")

    x_min: Number
    x_max: Number

    @model_validator(mode="after")
    def _ordered(self) -> "ExtentLocal":
        if self.x_max < self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must be >= x_min ({self.x_min})")
        return self


class Surface(BaseModel):
    """One feature (plane or cylinder) of a part: ``PARTS[part][surface]``.

    The keys are those of the dictionary (see ``docs/guide/input_format.md``).
    Keys that are not listed (e.g. ``META``, ``CAD_ID``) are kept and ignored.
    Which keys are required depends on the other keys and on the interactions;
    those checks need the part and surface labels and are made by `SystemData`.

    Attributes
    ----------
    TYPE : str
        One of ``otaf.constants.SUPPORTED_SURFACE_TYPES``.
    FRAME : ndarray of shape (3, 3)
        Rotation matrix whose columns are the local axes (x: plane normal or
        cylinder axis).
    ORIGIN : ndarray of shape (3,)
        Origin of the surface frame. Taken from the origin point when ``POINTS``
        is given.
    POINTS : dict of str to ndarray of shape (3,)
        Points of the feature, labelled ``<PREFIX><int>``; must contain the
        origin (suffix all zeros).
    CONTOUR_GLOBAL : ndarray of shape (n, 3)
        Contour points in global coordinates, used when ``POINTS`` is absent.
    RADIUS : float
        Nominal radius of a cylinder, > 0.
    EXTENT_LOCAL : ExtentLocal
        Axial extent of a cylinder.
    INTERACTIONS : list of str
        Surfaces this one can touch, as ``P<part><surface>``.
    CONSTRAINTS_D : list of one str
        Deviations of the feature: ``NONE``, ``PERFECT``, ``TRANSLATION`` or
        ``ROTATION`` (case-insensitive).
    CONSTRAINTS_G : list of one str
        Relative motion at the contact: ``FLOATING``, ``SLIDING`` or ``FIXED``
        (case-insensitive). Must be the same on both surfaces of a contact.
    BLOCK_TRANSLATIONS_G, BLOCK_ROTATIONS_G : str
        Extra blocked gap axes, letters among ``xyz``.
    SURFACE_DIRECTION : str
        ``centripetal`` (hole) or ``centrifugal`` (pin); cylinders only.
    DIAMETER_DEVIATION : bool
        Diameter as a deviation variable; cylinders only.
    """

    model_config = ConfigDict(extra="allow")

    TYPE: str
    FRAME: Optional[Frame] = None
    ORIGIN: Optional[Vector3] = None
    POINTS: Optional[Dict[str, Vector3]] = None
    CONTOUR_GLOBAL: Optional[Contour] = None
    RADIUS: Optional[PositiveNumber] = None
    EXTENT_LOCAL: Optional[ExtentLocal] = None
    INTERACTIONS: Optional[List[str]] = None
    CONSTRAINTS_D: Optional[List[str]] = None
    CONSTRAINTS_G: Optional[List[str]] = None
    BLOCK_TRANSLATIONS_G: Optional[str] = None
    BLOCK_ROTATIONS_G: Optional[str] = None
    SURFACE_DIRECTION: Optional[str] = None
    DIAMETER_DEVIATION: Optional[StrictBool] = None

    @field_validator("TYPE")
    @classmethod
    def _supported_type(cls, value: str) -> str:
        # Turned into UnsupportedSurfaceTypeError (which needs the labels) by validate_system_data.
        if value not in otaf_constants.SUPPORTED_SURFACE_TYPES:
            raise ValueError(f"unsupported surface type {value!r}")
        return value

    @field_validator("POINTS", mode="before")
    @classmethod
    def _point_labels(cls, value: Any) -> Any:
        if isinstance(value, dict):
            validate_point_dict(value)
        return value

    @field_validator("INTERACTIONS")
    @classmethod
    def _interaction_format(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        for interaction in value or []:
            if not otaf_constants.BASE_PART_SURF_PATTERN.fullmatch(interaction):
                raise otaf_exceptions.InvalidInteractionFormatError(interaction)
        return value

    @field_validator("CONSTRAINTS_D", "CONSTRAINTS_G", mode="before")
    @classmethod
    def _one_entry_list(cls, value: Any) -> Any:
        if value is not None and (not isinstance(value, (list, tuple)) or len(value) != 1):
            raise ValueError(f"must be a list with exactly one entry, e.g. ['NONE'], got {value!r}")
        return list(value) if value is not None else None

    @field_validator("CONSTRAINTS_D")
    @classmethod
    def _deviation_constraint(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is not None and value[0].upper() not in _DEVIATION_CONSTRAINTS:
            raise ValueError(f"{value[0]!r} is not one of {_DEVIATION_CONSTRAINTS}")
        return value

    @field_validator("CONSTRAINTS_G")
    @classmethod
    def _contact_constraint(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is not None and value[0].upper() not in otaf_constants.CONTACT_TYPES:
            raise ValueError(f"{value[0]!r} is not one of {otaf_constants.CONTACT_TYPES}")
        return value

    @field_validator("BLOCK_TRANSLATIONS_G", "BLOCK_ROTATIONS_G")
    @classmethod
    def _axes(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not _AXES_PATTERN.fullmatch(value):
            raise ValueError(f"must be a string of the axes x, y, z (e.g. 'xz'), got {value!r}")
        return value

    @field_validator("SURFACE_DIRECTION")
    @classmethod
    def _direction(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in otaf_constants.SURFACE_DIRECTIONS:
            raise otaf_exceptions.InvalidSurfaceDirectionError(value)
        return value

    @model_validator(mode="after")
    def _cylinder_only_keys(self) -> "Surface":
        if self.DIAMETER_DEVIATION is not None and self.TYPE != "cylinder":
            raise ValueError("DIAMETER_DEVIATION is only valid for cylinders")
        if self.SURFACE_DIRECTION is not None and self.TYPE != "cylinder":
            raise ValueError("SURFACE_DIRECTION is only valid for cylinders")
        if self.CONSTRAINTS_D is not None:
            key = f"{self.TYPE}-{self.CONSTRAINTS_D[0].lower()}"
            if key not in otaf_constants.SURF_TYPE_TO_DEVIATION_DOF:
                raise ValueError(f"CONSTRAINTS_D {self.CONSTRAINTS_D} is not defined for a {self.TYPE}")
        return self

    def contact_constraint(self) -> str:
        """Return the ``CONSTRAINTS_G`` entry, upper case, ``FLOATING`` by default.

        Returns
        -------
        str
            The entry used for the contact type.
        """
        return (self.CONSTRAINTS_G or ["FLOATING"])[0].upper()


class Part(RootModel[Dict[str, Surface]]):
    """The surfaces of one part: ``PARTS[part]``, keyed by lowercase surface labels."""

    @model_validator(mode="before")
    @classmethod
    def _surface_labels(cls, value: Any) -> Any:
        if isinstance(value, dict) and not all(
            isinstance(k, str) and _SURFACE_LABEL_PATTERN.fullmatch(k) for k in value
        ):
            raise otaf_exceptions.InvalidSurfaceLabelError()
        return value


class Loops(BaseModel):
    """Compatibility loops: ``LOOPS``.

    Attributes
    ----------
    COMPATIBILITY : dict of str to str
        Loop name to loop string, elements ``P<part><surface><point>`` joined
        by ``" -> "``.
    """

    model_config = ConfigDict(extra="allow")

    COMPATIBILITY: Dict[str, str]

    @field_validator("COMPATIBILITY")
    @classmethod
    def _loop_elements(cls, value: Dict[str, str]) -> Dict[str, str]:
        for name, loop in value.items():
            elements = loop.split(" -> ")
            if len(elements) < 2:
                raise ValueError(f"loop {name!r} needs at least two elements joined by ' -> '")
            for element in elements:
                if not otaf_constants.LOOP_ELEMENT_PATTERN.fullmatch(element):
                    raise ValueError(
                        f"invalid element {element!r} in loop {name!r}: expected "
                        "P<part><surface><POINT>, e.g. P1aA0"
                    )
        return value


class SystemData(BaseModel):
    """The whole assembly dictionary.

    Validate a plain dictionary with ``SystemData.model_validate(data)`` or
    `validate_system_data` (which raises the `otaf.exceptions` errors). A
    `SystemData` can be passed to `otaf.AssemblyDataProcessor` in place of the
    dictionary.

    Attributes
    ----------
    PARTS : dict of str to Part
        Parts keyed by the string of an integer.
    LOOPS : Loops
        Compatibility loops.
    GLOBAL_CONSTRAINTS : str
        ``3D``, ``2D_NX``, ``2D_NY`` or ``2D_NZ``.

    Examples
    --------
    >>> import numpy as np
    >>> data = {
    ...     "PARTS": {"1": {"a": {"TYPE": "plane", "FRAME": np.eye(3),
    ...                          "POINTS": {"A0": (0, 0, 0), "A1": (0, 1, 0)}}}},
    ...     "LOOPS": {"COMPATIBILITY": {}},
    ...     "GLOBAL_CONSTRAINTS": "2D_NZ",
    ... }
    >>> SystemData.model_validate(data).PARTS["1"].root["a"].TYPE
    'plane'
    """

    model_config = ConfigDict(extra="allow")

    PARTS: Dict[str, Part]
    LOOPS: Loops
    GLOBAL_CONSTRAINTS: Literal["3D", "2D_NX", "2D_NY", "2D_NZ"]

    @field_validator("PARTS", mode="before")
    @classmethod
    def _part_labels(cls, value: Any) -> Any:
        if isinstance(value, dict) and not all(isinstance(k, str) and k.isdigit() for k in value):
            raise otaf_exceptions.InvalidPartLabelError()
        return value

    @field_validator("GLOBAL_CONSTRAINTS", mode="before")
    @classmethod
    def _global_constraints(cls, value: Any) -> Any:
        if value not in otaf_constants.GLOBAL_CONSTRAINTS_TO_DEVIATION_DOF:
            raise otaf_exceptions.InvalidGlobalConstraintError(
                otaf_constants.GLOBAL_CONSTRAINTS_TO_DEVIATION_DOF.keys()
            )
        return value

    def surface(self, part_id: str, surf_id: str) -> Surface:
        """Return a surface by its labels.

        Parameters
        ----------
        part_id, surf_id : str
            Labels of the part and of the surface.

        Returns
        -------
        Surface
            The surface.

        Raises
        ------
        otaf.exceptions.PartNotFoundError, otaf.exceptions.SurfaceNotFoundError
            If the part or the surface does not exist.
        """
        if part_id not in self.PARTS:
            raise otaf_exceptions.PartNotFoundError(part_id)
        if surf_id not in self.PARTS[part_id].root:
            raise otaf_exceptions.SurfaceNotFoundError(surf_id, part_id)
        return self.PARTS[part_id].root[surf_id]

    def contact_pairs(self) -> List[tuple]:
        """Return the contacts listed in ``INTERACTIONS``, each pair once.

        Returns
        -------
        list of tuple
            Sorted ``((part1, surf1), (part2, surf2))`` pairs.
        """
        pairs = set()
        for part_id, part in self.PARTS.items():
            for surf_id, surface in part.root.items():
                for interaction in surface.INTERACTIONS or []:
                    other = otaf_constants.BASE_PART_SURF_PATTERN.fullmatch(interaction).groups()
                    pairs.add(tuple(sorted([(part_id, surf_id), other])))
        return sorted(pairs)

    @model_validator(mode="after")
    def _surface_requirements(self) -> "SystemData":
        for part_id, part in self.PARTS.items():
            for surf_id, surface in part.root.items():
                where = f"surface {surf_id} on part {part_id}"
                if surface.FRAME is None:
                    raise otaf_exceptions.MissingKeyError("FRAME", f"Surface dictionary for {where}")
                if surface.POINTS is None:
                    for key in ("ORIGIN", "INTERACTIONS"):
                        if getattr(surface, key) is None:
                            raise otaf_exceptions.MissingKeyError(
                                key, f"Surface dictionary for {where} (no POINTS given)"
                            )
                elif not any(
                    otaf_constants.SURF_ORIGIN_PATTERN.fullmatch(label) for label in surface.POINTS
                ):
                    raise otaf_exceptions.MissingOriginPointError(part_id, surf_id)
                if surface.TYPE == "cylinder" and surface.INTERACTIONS:
                    for key in ("RADIUS", "EXTENT_LOCAL"):
                        if getattr(surface, key) is None:
                            raise otaf_exceptions.MissingKeyError(
                                key, f"Surface dictionary for {where} (cylinder in an interaction)"
                            )
        return self

    @model_validator(mode="after")
    def _contacts(self) -> "SystemData":
        for (p1, s1), (p2, s2) in self.contact_pairs():
            surf1, surf2 = self.surface(p1, s1), self.surface(p2, s2)
            if surf1.TYPE != surf2.TYPE:
                raise NotImplementedError(
                    f"Interaction between a {surf1.TYPE} (P{p1}{s1}) and a {surf2.TYPE} "
                    f"(P{p2}{s2}) is not supported."
                )
            if surf1.contact_constraint() != surf2.contact_constraint():
                raise otaf_exceptions.InconsistentContactConstraintsError(
                    p1, s1, surf1.CONSTRAINTS_G or ["FLOATING"], p2, s2, surf2.CONSTRAINTS_G or ["FLOATING"]
                )
        return self

    @model_validator(mode="after")
    def _loop_references(self) -> "SystemData":
        for loop in self.LOOPS.COMPATIBILITY.values():
            for element in loop.split(" -> "):
                part_id, surf_id, _ = otaf_constants.LOC_STAMP_PATTERN.fullmatch(element).groups()
                self.surface(part_id, surf_id)
        return self

    def to_system_data(self) -> Dict[str, Any]:
        """Return the plain dictionary (keys that were not given stay absent).

        Returns
        -------
        dict
            The system data, arrays as float64 NumPy arrays.
        """
        data = self.model_dump(exclude_unset=True)
        for part in data["PARTS"].values():
            for surface in part.values():
                if "EXTENT_LOCAL" in surface:
                    surface["EXTENT_LOCAL"] = dict(surface["EXTENT_LOCAL"])
        return data


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def _location(loc: tuple) -> str:
    return " -> ".join(str(item) for item in loc)


def _specific_error(error: dict) -> Optional[Exception]:
    """Return the `otaf.exceptions` error matching one pydantic error, if there is one."""
    loc = tuple(error["loc"])
    if error["type"] == "missing":
        if len(loc) == 4 and loc[0] == "PARTS" and loc[3] == "TYPE":
            return otaf_exceptions.MissingSurfaceTypeKeyError(loc[1], loc[2])
        parent = _location(loc[:-1]) or "system_data"
        return otaf_exceptions.MissingKeyError(loc[-1], parent)
    if len(loc) == 4 and loc[0] == "PARTS" and loc[3] == "TYPE" and error["type"] == "value_error":
        return otaf_exceptions.UnsupportedSurfaceTypeError(loc[1], loc[2], error["input"])
    cause = (error.get("ctx") or {}).get("error")
    if isinstance(cause, Exception) and type(cause).__module__.startswith("otaf.exceptions"):
        if loc:
            cause.args = (f"{_location(loc)}: {cause.args[0]}", *cause.args[1:])
        return cause
    return None


def validate_system_data(system_data: Any) -> SystemData:
    """Validate an assembly dictionary against the input schema.

    Parameters
    ----------
    system_data : dict or SystemData
        The assembly dictionary (see ``docs/guide/input_format.md``).

    Returns
    -------
    SystemData
        The validated model. The dictionary itself is not modified.

    Raises
    ------
    otaf.exceptions.MissingKeyError
        If a required key is missing (``MissingSurfaceTypeKeyError`` for
        ``TYPE``).
    otaf.exceptions.InvalidPartLabelError, otaf.exceptions.InvalidSurfaceLabelError
        For malformed part or surface labels.
    otaf.exceptions.UnsupportedSurfaceTypeError
        If ``TYPE`` is not ``plane`` or ``cylinder``.
    otaf.exceptions.InconsistentContactConstraintsError
        If the two surfaces of a contact have different ``CONSTRAINTS_G``.
    NotImplementedError
        For a contact between a plane and a cylinder.
    otaf.exceptions.SystemDataValidationError
        For any other problem; lists all of them.

    Other `otaf.exceptions` errors (point labels, interaction format, missing
    origin point, unknown part or surface, ...) are raised as they are, with the
    location of the problem in front of their message. When several problems
    are found, the first one is raised, chained to a
    `~otaf.exceptions.SystemDataValidationError` listing all of them.
    """
    if isinstance(system_data, SystemData):
        return system_data
    try:
        return SystemData.model_validate(system_data)
    except ValidationError as e:
        errors = e.errors()
        messages = [f"{_location(err['loc']) or 'system_data'}: {err['msg']}" for err in errors]
        summary = otaf_exceptions.SystemDataValidationError(messages)
        for err in errors:
            specific = _specific_error(err)
            if specific is not None:
                raise specific from (summary if len(errors) > 1 else None)
        raise summary from None
