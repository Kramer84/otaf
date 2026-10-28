"""Validation of the assembly dictionary (otaf.schema) and the input errors of Phase 4."""

import json
import warnings

import numpy as np
import pytest

import otaf
import otaf.exceptions as otaf_exceptions
from otaf.example_models import model1, model2, model4_auto
from otaf.schema import SystemData, validate_system_data


def _build_ilh(data):
    adp = otaf.AssemblyDataProcessor(data)
    adp.generate_expanded_loops()
    clh = otaf.CompatibilityLoopHandling(adp)
    return otaf.InterfaceLoopHandling(adp, clh, circle_resolution=8)


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, (list, tuple, np.ndarray)) and all(not isinstance(x, (str, dict)) for x in np.ravel(a)):
        return np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


# --- The example models are valid --------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [model1.get_assembly_data(), model2.get_assembly_data(), model4_auto.get_assembly_data()],
    ids=["model1", "model2", "model4_auto"],
)
def test_example_models_validate(data):
    model = validate_system_data(data)
    assert set(model.PARTS) == set(data["PARTS"])


def test_validation_does_not_modify_the_dictionary():
    data = model4_auto.get_assembly_data(NX=1, NY=1)
    before = repr(data)
    validate_system_data(data)
    assert repr(data) == before


def test_system_data_model_is_accepted_by_the_processor():
    data = model1.get_assembly_data()
    from_dict = otaf.AssemblyDataProcessor(data)
    from_model = otaf.AssemblyDataProcessor(SystemData.model_validate(data))
    assert _same(from_model.system_data, from_dict.system_data)


def test_json_schema():
    schema = SystemData.model_json_schema()
    json.dumps(schema)
    assert {"PARTS", "LOOPS", "GLOBAL_CONSTRAINTS"} <= set(schema["required"])


def test_extra_keys_are_kept():
    data = model1.get_assembly_data()
    data["PARTS"]["1"]["a"]["CAD_ID"] = "face#12"
    data["META"] = {"source": "test"}
    validate_system_data(data)


# --- Frames ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frame",
    [
        np.diag([1.0, 1.0, 1.1]),  # not orthonormal
        np.diag([1.0, 1.0, -1.0]),  # orthonormal, left-handed
        np.eye(2),  # wrong shape
    ],
    ids=["scaled", "left-handed", "2x2"],
)
def test_frame_is_checked_when_points_are_given(frame):
    data = model1.get_assembly_data()
    assert "POINTS" in data["PARTS"]["1"]["a"]
    data["PARTS"]["1"]["a"]["FRAME"] = frame
    with pytest.raises(otaf_exceptions.SystemDataValidationError, match="PARTS -> 1 -> a -> FRAME"):
        otaf.AssemblyDataProcessor(data)


def test_frame_is_required():
    data = model1.get_assembly_data()
    del data["PARTS"]["2"]["c"]["FRAME"]
    with pytest.raises(otaf_exceptions.MissingKeyError, match="FRAME.*surface c on part 2"):
        otaf.AssemblyDataProcessor(data)


# --- Surface types and keys --------------------------------------------------------------


@pytest.mark.parametrize("surf_type", ["cone", "sphere", "torus"])
def test_unsupported_surface_types_are_rejected_up_front(surf_type):
    data = model1.get_assembly_data()
    data["PARTS"]["1"]["a"]["TYPE"] = surf_type
    with pytest.raises(otaf_exceptions.UnsupportedSurfaceTypeError, match="Supported types"):
        otaf.AssemblyDataProcessor(data)


def test_missing_type_names_part_and_surface_in_the_right_order():
    data = model2.get_assembly_data()
    del data["PARTS"]["1"]["c"]["TYPE"]
    with pytest.raises(otaf_exceptions.MissingSurfaceTypeKeyError) as info:
        otaf.AssemblyDataProcessor(data)
    assert "surface c in part 1" in str(info.value)


@pytest.mark.parametrize("key", ["RADIUS", "EXTENT_LOCAL"])
def test_cylinders_in_an_interaction_need_radius_and_extent(key):
    data = model4_auto.get_assembly_data(NX=1, NY=1)
    del data["PARTS"]["0"]["b"][key]
    with pytest.raises(otaf_exceptions.MissingKeyError, match=f"{key}.*surface b on part 0"):
        otaf.AssemblyDataProcessor(data)


@pytest.mark.parametrize(
    "key, value",
    [
        ("RADIUS", -1.0),
        ("RADIUS", "10"),
        ("EXTENT_LOCAL", {"x_min": 15.0, "x_max": -15.0}),
        ("DIAMETER_DEVIATION", 1),
        ("CONSTRAINTS_D", "PERFECT"),
        ("CONSTRAINTS_D", ["BENT"]),
        ("CONSTRAINTS_G", ["SLIDING", "FIXED"]),
        ("BLOCK_ROTATIONS_G", "xw"),
    ],
)
def test_invalid_surface_values(key, value):
    data = model4_auto.get_assembly_data(NX=1, NY=1)
    data["PARTS"]["0"]["b"][key] = value
    with pytest.raises(otaf_exceptions.SystemDataValidationError, match=f"PARTS -> 0 -> b -> {key}"):
        otaf.AssemblyDataProcessor(data)


def test_missing_nested_key():
    data = model4_auto.get_assembly_data(NX=1, NY=1)
    data["PARTS"]["0"]["b"]["EXTENT_LOCAL"] = {"x_min": -15.0}
    with pytest.raises(otaf_exceptions.MissingKeyError, match="x_max.*PARTS -> 0 -> b -> EXTENT_LOCAL"):
        otaf.AssemblyDataProcessor(data)


def test_surface_direction_keeps_its_exception():
    data = model4_auto.get_assembly_data(NX=1, NY=1)
    data["PARTS"]["0"]["b"]["SURFACE_DIRECTION"] = "outward"
    with pytest.raises(otaf_exceptions.InvalidSurfaceDirectionError, match="PARTS -> 0 -> b"):
        otaf.AssemblyDataProcessor(data)


def test_several_problems_are_all_listed():
    data = model4_auto.get_assembly_data(NX=1, NY=1)
    data["PARTS"]["0"]["b"]["RADIUS"] = -1.0
    data["PARTS"]["1"]["b"]["SURFACE_DIRECTION"] = "outward"
    with pytest.raises(otaf_exceptions.InvalidSurfaceDirectionError) as info:
        otaf.AssemblyDataProcessor(data)
    summary = info.value.__cause__
    assert isinstance(summary, otaf_exceptions.SystemDataValidationError)
    assert len(summary.messages) == 2


# --- References between surfaces and loops ------------------------------------------------


def test_interaction_with_an_unknown_surface():
    data = model1.get_assembly_data()
    data["PARTS"]["1"]["a"]["INTERACTIONS"] = ["P2z"]
    with pytest.raises(otaf_exceptions.SurfaceNotFoundError):
        otaf.AssemblyDataProcessor(data)


def test_loop_with_an_unknown_part():
    data = model1.get_assembly_data()
    data["LOOPS"]["COMPATIBILITY"]["L2"] = "P1cC0 -> P7cC0 -> P7aA0 -> P1aA0"
    with pytest.raises(otaf_exceptions.PartNotFoundError):
        otaf.AssemblyDataProcessor(data)


def test_malformed_loop_element():
    data = model1.get_assembly_data()
    data["LOOPS"]["COMPATIBILITY"]["L2"] = "P1cC0 -> P2c -> P1aA0"
    with pytest.raises(otaf_exceptions.SystemDataValidationError, match="invalid element 'P2c'"):
        otaf.AssemblyDataProcessor(data)


# --- Contacts (Phase 3 findings) ---------------------------------------------------------


def test_inconsistent_contact_constraints_are_explicit():
    data = model1.get_assembly_data()
    # The bottoms slide; leaving one side to the default gave KeyError: 'plane-plane-floating-sliding'.
    del data["PARTS"]["2"]["c"]["CONSTRAINTS_G"]
    with pytest.raises(otaf_exceptions.InconsistentContactConstraintsError, match="P1c-P2c"):
        otaf.AssemblyDataProcessor(data)


def test_inconsistent_contact_constraints_on_a_loop_gap_outside_interactions():
    # Without INTERACTIONS the schema has no pair to compare; the gap of the loop is checked.
    data = model1.get_assembly_data()
    for part in ("1", "2"):
        del data["PARTS"][part]["c"]["INTERACTIONS"]
    data["PARTS"]["2"]["c"]["CONSTRAINTS_G"] = ["FIXED"]
    adp = otaf.AssemblyDataProcessor(data)
    adp.generate_expanded_loops()
    with pytest.raises(otaf_exceptions.InconsistentContactConstraintsError):
        otaf.CompatibilityLoopHandling(adp)


def test_contact_constraints_are_case_insensitive():
    data = model1.get_assembly_data()
    data["PARTS"]["2"]["c"]["CONSTRAINTS_G"] = ["sliding"]
    adp = otaf.AssemblyDataProcessor(data)
    adp.generate_expanded_loops()
    otaf.CompatibilityLoopHandling(adp)


def test_contact_not_crossed_by_a_loop_warns():
    data = model1.get_assembly_data()
    del data["LOOPS"]["COMPATIBILITY"]["L1"]  # the only loop through the right walls
    with pytest.warns(otaf_exceptions.UncoveredContactWarning, match="P1b-P2b"):
        ilh = _build_ilh(data)
    assert ilh.uncovered_contacts() == [(("1", "b"), ("2", "b"))]


@pytest.mark.slow
@pytest.mark.parametrize(
    "data",
    [model1.get_assembly_data(), model2.get_assembly_data(), model4_auto.get_assembly_data(NX=1, NY=1)],
    ids=["model1", "model2", "model4_auto"],
)
def test_example_models_cover_every_contact(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error", otaf_exceptions.UncoveredContactWarning)
        assert _build_ilh(data).uncovered_contacts() == []


def test_plane_cylinder_contact_is_rejected_up_front():
    data = model4_auto.get_assembly_data(NX=1, NY=1)
    data["PARTS"]["0"]["a"]["INTERACTIONS"].append("P2b")
    data["PARTS"]["2"]["b"]["INTERACTIONS"].append("P0a")
    with pytest.raises(NotImplementedError, match="plane"):
        otaf.AssemblyDataProcessor(data)
