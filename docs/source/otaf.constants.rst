otaf.constants module
=====================

The `otaf.constants` module defines constants, regex patterns, mappings, and utility dictionaries essential for surface and gap modeling in the OTAF framework. These include definitions of surface types, regular expressions for validating naming patterns, and mappings for constraints, degrees of freedom, and nullified components.

Constants Overview
------------------

Below is a detailed description of each constant, its purpose, and examples.

---

Base Surface Types and Directions
---------------------------------

**BASE_SURFACE_TYPES**

List of supported surface types for modeling:
- `plane`
- `cylinder`
- `cone`
- `sphere`

**SURFACE_DIRECTIONS**

List of valid surface directions for interactions:
- `centripetal`
- `centrifugal`

**CONTACT_TYPES**

Types of contact interactions between surfaces:
- `FIXED`
- `SLIDING`
- `FLOATING`

---

Regular Expression Patterns
---------------------------

**SURF_POINT_PATTERN**

Regex to validate surface point names, e.g., `'AA01'`.

::

    [A-Z]+[0-9]+$

**SURF_ORIGIN_PATTERN**

Regex to validate surface origin names, e.g., `'AA0000'`.

::

    [A-Z]+0+$

**BASE_PART_SURF_PATTERN**

Regex for part-surface naming patterns, e.g., `'P1a'`.

::

    ^P(\d+)([a-z]+)$

**LOOP_ELEMENT_PATTERN**

Regex for loop element names, e.g., `'P1aA5'`.

::

    ^P\d+[a-z]+[A-Z]+\d+$

**LOC_STAMP_PATTERN**

Regex for location stamp naming, e.g., `'P1aA5'`.

::

    ^P(\d+)([a-z]+)([A-Z]+\d+)$

**Transformation and Deviation Patterns**

- **T_MATRIX_PATTERN**: Regex for transformation matrix naming, e.g., `'TP1aA0bB99'`.

::

    ^TP(\d+)([a-z]+)([A-Z]+\d+)([a-z]+)([A-Z]+\d+)$

- **D_MATRIX_PATTERN1**: Regex for deviation matrix type 1, e.g., `'D1a'` or `'Di1a'`.

::

    ^D(i*)(\d+)([a-z]+)$

- **D_MATRIX_PATTERN2**: Regex for deviation matrix type 2, e.g., `'D1aA5'`.

::

    ^D(i*)(\d+)([a-z]+)([A-Z]+\d+)$

- **D_MATRIX_PATTERN3**: Regex for deviation matrix type 3, e.g., `'D1a5b'`.

::

    ^D(i*)(\d+)([a-z]+)(\d+)([a-z]+)$

---

Mappings for Constraints and Degrees of Freedom
-----------------------------------------------

**GLOBAL_CONSTRAINTS_TO_DEVIATION_DOF**

Mapping of global constraints to degrees of freedom for deviations:

- **3D**: `translations_2remove = ""`, `rotations_2remove = ""`
- **2D_NX**: `translations_2remove = "x"`, `rotations_2remove = "yz"`
- **2D_NY**: `translations_2remove = "y"`, `rotations_2remove = "xz"`
- **2D_NZ**: `translations_2remove = "z"`, `rotations_2remove = "xy"`

**SURF_TYPE_TO_DEVIATION_DOF**

Mapping of surface types to deviation degrees of freedom (DOF):

- **plane-none**: `translations = "x"`, `rotations = "yz"`
- **cylinder-none**: `translations = "yz"`, `rotations = "yz"`

**GLOBAL_CONSTRAINTS_TO_GAP_DOF**

Mapping of global constraints to blocked degrees of freedom for gaps:

- **3D**: `translations_blocked = ""`, `rotations_blocked = ""`
- **2D_NX**: `translations_blocked = "x"`, `rotations_blocked = "yz"`

---

Gap Interaction and Nullified Components
-----------------------------------------

**GAP_TYPE_TO_NULLIFIED_NOMINAL_COMPONENTS**

Mapping gap type interaction to gap components nullified in the nominal gap matrix:

- **plane-plane**: `nullify_x = True`, `nullify_y = False`, `nullify_z = False`
- **cylinder-cylinder**: `nullify_x = True`, `nullify_y = True`, `nullify_z = True`

---

Matrix Basis for Modeling
-------------------------

**BASIS_DICT**

Defines components of the matrix basis used for modeling defects and rotations under the small displacement hypothesis.

Example:
- **Basis 1**:
  - `AXIS = "x"`
  - `VARIABLE_D = "u_d"`
  - `VARIABLE_G = "u_g"`

---

Surface Property Validation
---------------------------

**SURFACE_DICT_VALUE_CHECKS**

Validation functions for surface property values:
- **TYPE**: Validates surface type matches `BASE_SURFACE_TYPES`.
- **FRAME**: Checks for orthogonality and determinant of 1.
- **ORIGIN**: Validates the shape of `np.array(x)` is `(3,)`.
