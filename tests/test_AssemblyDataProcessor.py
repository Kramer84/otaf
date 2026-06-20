import pytest
import numpy as np

from otaf import AssemblyDataProcessor
import otaf.exceptions as otaf_exceptions

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def empty_processor():
    """Returns a processor initialized with no data, generating the default empty structure."""
    return AssemblyDataProcessor()

@pytest.fixture
def minimal_valid_system_data():
    """Returns a valid baseline dictionary to initialize the processor."""
    return {
        "PARTS": {
            "1": {
                "a": {
                    "TYPE": "plane",
                    "FRAME": np.eye(3).tolist(),
                    "ORIGIN": np.array([0.0, 0.0, 0.0]),
                    "POINTS": {
                        "A0": np.array([0.0, 0.0, 0.0]),
                        "A1": np.array([1.0, 0.0, 0.0]),
                        "A2": np.array([0.0, 1.0, 0.0])
                    }
                }
            }
        },
        "LOOPS": {"COMPATIBILITY": {}},
        "GLOBAL_CONSTRAINTS": "3D"
    }

@pytest.fixture
def populated_processor(minimal_valid_system_data):
    """Returns a processor initialized with valid data."""
    return AssemblyDataProcessor(minimal_valid_system_data)

# -----------------------------------------------------------------------------
# Tests for Small / Trivial Methods
# -----------------------------------------------------------------------------

def test_initialization_empty(empty_processor):
    assert "PARTS" in empty_processor.system_data
    assert "LOOPS" in empty_processor.system_data
    assert "GLOBAL_CONSTRAINTS" in empty_processor.system_data
    assert empty_processor.system_data["GLOBAL_CONSTRAINTS"] == "3D"

def test_dunder_methods(populated_processor):
    # Test __getitem__
    assert "1" in populated_processor["PARTS"]
    
    # Test __setitem__
    populated_processor["GLOBAL_CONSTRAINTS"] = "2D_NX"
    assert populated_processor["GLOBAL_CONSTRAINTS"] == "2D_NX"
    
    # Test __repr__
    repr_str = repr(populated_processor)
    assert isinstance(repr_str, str)
    assert "PARTS" in repr_str

def test_is_single_feature_part(populated_processor):
    # Part "1" only has surface "a"
    assert populated_processor._is_single_feature_part("1") is True
    
    # Add another feature
    populated_processor["PARTS"]["1"]["b"] = {}
    assert populated_processor._is_single_feature_part("1") is False

def test_get_surface_points(populated_processor):
    points = populated_processor.get_surface_points("1", "a")
    assert "A0" in points
    assert np.array_equal(points["A0"], np.array([0.0, 0.0, 0.0]))
    
    with pytest.raises(KeyError):
        populated_processor.get_surface_points("99", "a")

# -----------------------------------------------------------------------------
# Tests for Point Dictionary Validation (Data Variety)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("point_dict, expected_exception", [
    (
        # Invalid pattern (lowercase)
        {"a1": np.array([1, 2, 3])}, 
        otaf_exceptions.LabelPatternError
    ),
    (
        # Mixed prefixes (A and B)
        {"A1": np.array([1, 2, 3]), "B2": np.array([4, 5, 6])}, 
        otaf_exceptions.LabelPrefixError
    ),
    (
        # Non-unique suffix logic (A0 and A00 are both suffix 0)
        {"A0": np.array([0, 0, 0]), "A00": np.array([1, 1, 1])}, 
        otaf_exceptions.UniqueLabelSuffixError
    ),
    (
        # Non-unique coordinates
        {"A1": np.array([1.0, 2.0, 3.0]), "A2": np.array([1.0, 2.0, 3.0])}, 
        otaf_exceptions.NonUniqueCoordinatesError
    ),
])
def test_validate_point_dict_exceptions(empty_processor, point_dict, expected_exception):
    with pytest.raises(expected_exception):
        empty_processor.validate_point_dict(point_dict)

def test_validate_point_dict_valid(empty_processor):
    valid_dict = {
        "A0": np.array([0.0, 0.0, 0.0]),
        "A1": np.array([1.0, 0.0, 0.0]),
        "A2": np.array([0.0, 1.0, 0.0])
    }
    # Should not raise any exceptions
    empty_processor.validate_point_dict(valid_dict)

# -----------------------------------------------------------------------------
# Tests for Adding Surface Points
# -----------------------------------------------------------------------------

def test_add_surface_points(populated_processor):
    new_points = {"A3": np.array([0.0, 0.0, 1.0])}
    populated_processor.add_surface_points("1", "a", new_points)
    
    assert "A3" in populated_processor["PARTS"]["1"]["a"]["POINTS"]

def test_add_surface_points_duplicate_handling(populated_processor):
    duplicate_points = {"A0": np.array([0.0, 0.0, 0.0])}
    
    # By default, should raise DuplicatePointError
    with pytest.raises(otaf_exceptions.DuplicatePointError):
        populated_processor.add_surface_points("1", "a", duplicate_points)
        
    # With ignore_duplicates=True, exact match should pass silently
    populated_processor.add_surface_points("1", "a", duplicate_points, ignore_duplicates=True)

def test_add_surface_points_origin_update(populated_processor):
    # Original origin is [0,0,0]
    new_origin_point = {"A00": np.array([5.0, 5.0, 5.0])}
    populated_processor.add_surface_points("1", "a", new_origin_point)
    
    # Because A00 matches SURF_ORIGIN_PATTERN, ORIGIN should update
    assert np.array_equal(populated_processor["PARTS"]["1"]["a"]["ORIGIN"], np.array([5.0, 5.0, 5.0]))

# -----------------------------------------------------------------------------
# Tests for System Data Validation Matrix
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("corrupt_action, expected_exception", [
    (
        lambda data: data.pop("PARTS"), 
        otaf_exceptions.MissingKeyError
    ),
    (
        lambda data: data["PARTS"].update({"StringLabel": {}}), 
        otaf_exceptions.InvalidPartLabelError
    ),
    (
        lambda data: data["PARTS"]["1"].update({"A_upper": {}}), 
        otaf_exceptions.InvalidSurfaceLabelError
    ),
    (
        lambda data: data["PARTS"]["1"]["a"].update({"TYPE": "torus"}), 
        otaf_exceptions.UnsupportedSurfaceTypeError
    ),
    (
        lambda data: data["PARTS"]["1"]["a"].pop("TYPE"), 
        otaf_exceptions.MissingSurfaceTypeKeyError
    ),
    (
        lambda data: data.update({"GLOBAL_CONSTRAINTS": "4D_SPACE"}), 
        otaf_exceptions.InvalidGlobalConstraintError
    )
])
def test_validate_system_data_structure(minimal_valid_system_data, corrupt_action, expected_exception):
    corrupt_action(minimal_valid_system_data)
    with pytest.raises(expected_exception):
        AssemblyDataProcessor(minimal_valid_system_data)
        
# -----------------------------------------------------------------------------
# Data Generation Helpers (Simulating your input functions)
# -----------------------------------------------------------------------------

def _generate_base_geometry(X1=99.8, X2=100.0, X3=10.0):
    """Helper to generate the raw numpy coordinates and frames."""
    R0 = np.eye(3)
    x_, y_, z_ = R0[0], R0[1], R0[2]
    
    # Points
    P1A = [np.array((0, X3 / 2, 0.0)), np.array((0, X3, 0.0)), np.array((0, 0, 0.0))]
    P1B = [np.array((X1, X3 / 2, 0.0)), np.array((X1, X3, 0.0)), np.array((X1, 0, 0.0))]
    P1C = [np.array((X1 / 2, 0, 0.0)), np.array((0, 0, 0.0)), np.array((X1, 0, 0.0))]
    
    P2A = [np.array((0, X3 / 2, 0.0)), np.array((0, X3, 0.0)), np.array((0, 0, 0.0))]
    P2B = [np.array((X2, X3 / 2, 0.0)), np.array((X2, X3, 0.0)), np.array((X2, 0, 0.0))]
    P2C = [np.array((X2 / 2, 0, 0.0)), np.array((0, 0, 0.0)), np.array((X2, 0, 0.0))]

    # Frames
    RP1 = [np.array([-x_, -y_, z_]), R0, np.array([-y_, x_, z_])]
    RP2 = [R0, np.array([-x_, -y_, z_]), np.array([y_, -x_, z_])]
    
    return P1A, P1B, P1C, P2A, P2B, P2C, RP1, RP2

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def original_system_data():
    """Fixture returning the explicit manual point definition."""
    P1A, P1B, P1C, P2A, P2B, P2C, RP1, RP2 = _generate_base_geometry()
    return {
        "PARTS": {
            '1': {
                "a": {"FRAME": RP1[0], "POINTS": {'A0': P1A[0], 'A1': P1A[1], 'A2': P1A[2]}, "TYPE": "plane", "INTERACTIONS": ['P2a'], "CONSTRAINTS_D": ["PERFECT"], "CONSTRAINTS_G": ["FLOATING"]},
                "b": {"FRAME": RP1[1], "POINTS": {'B0': P1B[0], 'B1': P1B[1], 'B2': P1B[2]}, "TYPE": "plane", "INTERACTIONS": ['P2b'], "CONSTRAINTS_D": ["NONE"], "CONSTRAINTS_G": ["FLOATING"]},
                "c": {"FRAME": RP1[2], "POINTS": {'C0': P1C[0], 'C1': P1C[1], 'C2': P1C[2]}, "TYPE": "plane", "INTERACTIONS": ['P2c'], "CONSTRAINTS_D": ["PERFECT"], "CONSTRAINTS_G": ["SLIDING"]},
            },
            '2': {
                "a": {"FRAME": RP2[0], "POINTS": {'A0': P2A[0], 'A1': P2A[1], 'A2': P2A[2]}, "TYPE": "plane", "INTERACTIONS": ['P1a'], "CONSTRAINTS_D": ["PERFECT"], "CONSTRAINTS_G": ["FLOATING"]},
                "b": {"FRAME": RP2[1], "POINTS": {'B0': P2B[0], 'B1': P2B[1], 'B2': P2B[2]}, "TYPE": "plane", "INTERACTIONS": ['P1b'], "CONSTRAINTS_D": ["NONE"], "CONSTRAINTS_G": ["FLOATING"]},
                "c": {"FRAME": RP2[2], "POINTS": {'C0': P2C[0], 'C1': P2C[1], 'C2': P2C[2]}, "TYPE": "plane", "INTERACTIONS": ['P1c'], "CONSTRAINTS_D": ["PERFECT"], "CONSTRAINTS_G": ["SLIDING"]},
            }
        },
        "LOOPS": {"COMPATIBILITY": {"L0": "P1cC0 -> P2cC0 -> P2aA0 -> P1aA0", "L1": "P1cC0 -> P2cC0 -> P2bB0 -> P1bB0"}},
        "GLOBAL_CONSTRAINTS": "2D_NZ",
    }

@pytest.fixture
def variation_1_data():
    """Fixture returning the auto-generated CONTOUR_GLOBAL definition."""
    P1A, P1B, P1C, P2A, P2B, P2C, RP1, RP2 = _generate_base_geometry()
    return {
        "PARTS": {
            '1': {
                "a": {"FRAME": RP1[0], "ORIGIN": P1A[0], "CONTOUR_GLOBAL": np.array([P1A[1], P1A[2]]), "TYPE": "plane", "INTERACTIONS": ['P2a'], "CONSTRAINTS_D": ["PERFECT"], "CONSTRAINTS_G": ["FLOATING"]},
                "b": {"FRAME": RP1[1], "ORIGIN": P1B[0], "CONTOUR_GLOBAL": np.array([P1B[1], P1B[2]]), "TYPE": "plane", "INTERACTIONS": ['P2b'], "CONSTRAINTS_D": ["NONE"], "CONSTRAINTS_G": ["FLOATING"]},
                "c": {"FRAME": RP1[2], "ORIGIN": P1C[0], "CONTOUR_GLOBAL": np.array([P1C[1], P1C[2]]), "TYPE": "plane", "INTERACTIONS": ['P2c'], "CONSTRAINTS_D": ["PERFECT"], "CONSTRAINTS_G": ["SLIDING"]},
            },
            '2': {
                "a": {"FRAME": RP2[0], "ORIGIN": P2A[0], "CONTOUR_GLOBAL": np.array([P2A[1], P2A[2]]), "TYPE": "plane", "INTERACTIONS": ['P1a'], "CONSTRAINTS_D": ["PERFECT"], "CONSTRAINTS_G": ["FLOATING"]},
                "b": {"FRAME": RP2[1], "ORIGIN": P2B[0], "CONTOUR_GLOBAL": np.array([P2B[1], P2B[2]]), "TYPE": "plane", "INTERACTIONS": ['P1b'], "CONSTRAINTS_D": ["NONE"], "CONSTRAINTS_G": ["FLOATING"]},
                "c": {"FRAME": RP2[2], "ORIGIN": P2C[0], "CONTOUR_GLOBAL": np.array([P2C[1], P2C[2]]), "TYPE": "plane", "INTERACTIONS": ['P1c'], "CONSTRAINTS_D": ["PERFECT"], "CONSTRAINTS_G": ["SLIDING"]},
            }
        },
        "LOOPS": {"COMPATIBILITY": {"L0": "P1cC0 -> P2cC0 -> P2aA0 -> P1aA0", "L1": "P1cC0 -> P2cC0 -> P2bB0 -> P1bB0"}},
        "GLOBAL_CONSTRAINTS": "2D_NZ",
    }

@pytest.fixture
def variation_2_data():
    """Fixture returning the minimalist, origin-only definition."""
    P1A, P1B, P1C, P2A, P2B, P2C, RP1, RP2 = _generate_base_geometry()
    return {
        "PARTS": {
            '1': {
                "a": {"FRAME": RP1[0], "ORIGIN": P1A[0], "TYPE": "plane", "INTERACTIONS": ['P2a'], "CONSTRAINTS_D": ["PERFECT"], "CONSTRAINTS_G": ["FLOATING"]},
                "b": {"FRAME": RP1[1], "ORIGIN": P1B[0], "TYPE": "plane", "INTERACTIONS": ['P2b'], "CONSTRAINTS_D": ["NONE"], "CONSTRAINTS_G": ["FLOATING"]},
                "c": {"FRAME": RP1[2], "ORIGIN": P1C[0], "TYPE": "plane", "INTERACTIONS": ['P2c'], "CONSTRAINTS_D": ["PERFECT"], "CONSTRAINTS_G": ["SLIDING"]},
            },
            '2': {
                "a": {"FRAME": RP2[0], "ORIGIN": P2A[0], "TYPE": "plane", "INTERACTIONS": ['P1a'], "CONSTRAINTS_D": ["PERFECT"], "CONSTRAINTS_G": ["FLOATING"]},
                "b": {"FRAME": RP2[1], "ORIGIN": P2B[0], "TYPE": "plane", "INTERACTIONS": ['P1b'], "CONSTRAINTS_D": ["NONE"], "CONSTRAINTS_G": ["FLOATING"]},
                "c": {"FRAME": RP2[2], "ORIGIN": P2C[0], "TYPE": "plane", "INTERACTIONS": ['P1c'], "CONSTRAINTS_D": ["PERFECT"], "CONSTRAINTS_G": ["SLIDING"]},
            }
        },
        "LOOPS": {"COMPATIBILITY": {"L0": "P1cC0 -> P2cC0 -> P2aA0 -> P1aA0", "L1": "P1cC0 -> P2cC0 -> P2bB0 -> P1bB0"}},
        "GLOBAL_CONSTRAINTS": "2D_NZ",
    }

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_initialization_success_all_variations(original_system_data, variation_1_data, variation_2_data):
    """Ensures all 3 data definitions pass strict structural validation without throwing exceptions."""
    AssemblyDataProcessor(original_system_data)
    AssemblyDataProcessor(variation_1_data)
    AssemblyDataProcessor(variation_2_data)

def test_variation_1_point_parity(original_system_data, variation_1_data):
    """
    Tests that utilizing the CONTOUR_GLOBAL array logic rebuilds the 
    points dictionary exactly as if it had been manually entered.
    """
    processor_orig = AssemblyDataProcessor(original_system_data)
    processor_var1 = AssemblyDataProcessor(variation_1_data)

    for part_id in processor_orig["PARTS"]:
        for surf_id in processor_orig["PARTS"][part_id]:
            points_orig = processor_orig.get_surface_points(part_id, surf_id)
            points_var1 = processor_var1.get_surface_points(part_id, surf_id)
            
            assert points_orig.keys() == points_var1.keys(), f"Point labels mismatch on Part {part_id}, Surface {surf_id}"
            
            for pt_label in points_orig:
                assert np.array_equal(points_orig[pt_label], points_var1[pt_label]), \
                    f"Coordinates for {pt_label} mismatch on Part {part_id}, Surface {surf_id}"

def test_variation_2_loop_expansion_parity(original_system_data, variation_2_data):
    """
    Tests that a minimalist definition (missing extents, relying only on origins) 
    still successfully generates identical compatibility loop expansions, 
    as the loops only rely on the origin points.
    """
    processor_orig = AssemblyDataProcessor(original_system_data)
    processor_var2 = AssemblyDataProcessor(variation_2_data)

    processor_orig.generate_expanded_loops()
    processor_var2.generate_expanded_loops()

    assert processor_orig.compatibility_loops_expanded is not None
    assert processor_var2.compatibility_loops_expanded is not None

    assert processor_orig.compatibility_loops_expanded == processor_var2.compatibility_loops_expanded, \
        "Loop expansion deviated between original and minimalist data definitions."