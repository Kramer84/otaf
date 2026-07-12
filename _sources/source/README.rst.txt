.. _readme-top:

|Contributors| |Forks| |Stargazers| |Issues| |MIT License|

.. image:: logo/logo.png
   :width: 240px
   :align: center
   :alt: OTAF Logo

.. role:: center
   :class: center

**OTAF: Open Tolerance Analysis Framework**

A scientific Python framework for statistical tolerance analysis of over-constrained mechanical assemblies using linear system-of-constraints optimization.

`Explore the Docs <https://kramer84.github.io/otaf/>`_ | `Report Bug <https://github.com/Kramer84/otaf/issues>`_ | `Request Feature <https://github.com/Kramer84/otaf/issues>`_

.. contents:: Table of Contents
   :depth: 3

----

About The Project
-----------------

**OTAF (Open Tolerance Analysis Framework)** is an open-source,
research-driven Python library designed to perform statistical tolerance
analysis on three-dimensional over-constrained rigid mechanical
assemblies. The tool allows to construct a linearized mathematical model
of an assembly of parts, under the hypothsis of rigid parts and first
oder defects (modeled by translations and rotations of the nominal
feature). The module then maps rigid manufacturing deviations and
clearances into a set of matrices representing a linear programing
problem, where deviations can be fixed and the clearance space is
explored using optimization to find valid configurations. This
modelization can then be used to perform reliability analysis (tolerance
analysis), construct surrogate models etc.

.. raw:: html

   <p align="right"><a href="#readme-top">back to top</a></p>

Key Methodology
~~~~~~~~~~~~~~~

The framework transforms a high-level assembly description into the
linear programing model: 1. **Compatibility Loops (Equalities):** Closed
kinematic chains (3D tolerance stack-ups) that model global structural
behavior. Represented as linear matrix operations:

.. math:: A_{eq,Def} \cdot X + A_{eq,Gap} \cdot Y + K_{eq} = 0

2. **Interface Constraints (Inequalities):** Local contact interactions
at boundary contour points across matching feature surfaces. Ensures no
material interpenetration occurs:

.. math:: A_{ub,Def} \cdot X + A_{ub,Gap} \cdot Y + K_{ub} \ge 0

Where :math:`X` represents the elementar manufacturing defect vectors
(:math:`D`), :math:`Y` captures internal clearance and gap freedoms
(:math:`g`), and :math:`K` represents nominal constants.

By introducing an auxiliary slack variable :math:`s` into the interface
constraints via ``embedOptimizationVariable()``, the framework
determines structural assemblability based on the sign of :math:`s`. If
optimization yields :math:`s < 0`, the manufacturing defects exceed the
system’s geometric compensation capacity, signaling an assembly failure.

.. raw:: html

   <p align="right"><a href="#readme-top">back to top</a></p>

Built With
~~~~~~~~~~

-  **Core Mathematics & Symbols:** `NumPy <https://numpy.org/>`__,
   `SciPy <https://scipy.org/>`__, `SymPy <https://www.sympy.org/>`__
-  **Uncertainty Quantification:**
   `OpenTURNS <https://openturns.github.io/www/>`__
-  **Execution & Parallelization:**
   `Joblib <https://joblib.readthedocs.io/>`__
-  **Type Integrity Checking:**
   `Beartype <https://github.com/beartype/beartype>`__
-  **3D Mesh Rendering & Visualization:**
   `Trimesh <https://github.com/mikedh/trimesh>`__

.. raw:: html

   <p align="right"><a href="#readme-top">back to top</a></p>

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

OTAF requires Python :math:`\ge` 3.8. It is highly recommended to
isolate installation dependencies inside a virtual environment or Conda
environment.

.. code:: bash

   python3 -m venv otaf_env
   source otaf_env/bin/activate

Installation
~~~~~~~~~~~~

Install OTAF directly from source along with your required dependency
subsets:

.. code:: bash

   # Clone the repository
   git clone [https://github.com/Kramer84/otaf.git](https://github.com/Kramer84/otaf.git)
   cd otaf

   # Install the base structural engine
   pip install .

   # Install with 3D visualization and geometry tools
   pip install .[viz]

   # Install with neural network surrogate modeling tools
   pip install .[surrogate]

   # Install all components including Sphinx documentation pipelines
   pip install .[all]

.. raw:: html

   <p align="right"><a href="#readme-top">back to top</a></p>

Usage
-----

1. High-Level Assembly Input Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Design configurations are passed to the ``AssemblyDataProcessor`` as
nested dictionaries. Features are configured via explicit alphanumeric
keys (e.g., Part ``1``, surface ``a``, point ``A0`` is labeled
``P1aA0``). Coordinate points can be defined natively within global
reference frames or mapped relative to local surfaces.

.. code:: python

   import otaf
   import numpy as np

   system_data = {
       "PARTS": {
           "1": {
               "a": {
                   "FRAME": np.eye(3),  # Surface frame orientation matrix
                   "POINTS": {"A0": np.array([0,0,0]), "A1": np.array([1,0,0])},
                   "TYPE": "plane",
                   "INTERACTIONS": ["P2a"],
                   "CONSTRAINTS_D": ["PERFECT"],
                   "CONSTRAINTS_G": ["FLOATING"],
               },
           },
           "2": {
               "a": {
                   "FRAME": np.eye(3),
                   "POINTS": {"A0": np.array([0,0,0]), "A1": np.array([1,0,0])},
                   "TYPE": "plane",
                   "INTERACTIONS": ["P1a"],
                   "CONSTRAINTS_D": ["PERFECT"],
                   "CONSTRAINTS_G": ["FLOATING"],
               },
           }
       },
       "LOOPS": {
           "COMPATIBILITY": {
               "L0": "P1aA0 -> P2aA0"  # Compact kinematic chain loop sequence
           }
       },
       "GLOBAL_CONSTRAINTS": "2D_NZ",
   }

2. Matrix Compilation and Feasibility Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once parsed, structural boundaries pass into the constraint compiler to
generate standard optimization targets:

.. code:: python

   from otaf import (
       AssemblyDataProcessor,
       CompatibilityLoopHandling,
       InterfaceLoopHandling,
       SystemOfConstraintsAssemblyModel
   )

   # 1. Parse raw structural maps and expand topological vector loops
   processor = AssemblyDataProcessor(system_data)
   processor.generate_expanded_loops()

   # 2. Pass the processor to intermediate handlers to compile symbolic equations
   comp_handler = CompatibilityLoopHandling(processor)
   inter_handler = InterfaceLoopHandling(processor, comp_handler)

   # 3. Extract the SymPy expressions and generate the numerical constraint matrices
   soc_model = SystemOfConstraintsAssemblyModel(
       compatibility_eqs=comp_handler.get_compatibility_expression_from_FO_matrices(),
       interface_eqs=inter_handler.get_interface_loop_expressions()
   )

   # 4. Test ideal structural alignment (Zero-Deviation Case)
   ideal_test = soc_model.test_zero_deviation_feasibility()
   print("Feasible without defects:", ideal_test["success"])

.. raw:: html

   <p align="right"><a href="#readme-top">back to top</a></p>

Local Documentation Compilation
-------------------------------

To build full API reference catalogs locally, verify ``pandoc`` is
present on your local system:

.. code:: bash

   sudo apt install pandoc
   sphinx-apidoc -o docs/source src/otaf
   cd docs/
   make clean
   make html

.. raw:: html

   <p align="right"><a href="#readme-top">back to top</a></p>

Framework Assumptions & Capabilities
------------------------------------

-  **Dual Probabilistic Scope:** Classical statistical tolerance
   analyses using fixed vectors of random standard distributions can be
   performed. Additionally methods to include epistemic uncertainty
   (imprecise probabilities) to trace upper and lower response envelopes
   (probability-boxes) have been developed here and are part of the
   examples to showcase the workflow.
-  **Rigid Variational Modeling:** Analysis is currently restricted to
   rigid transformations (translations and rotations). The linear
   programing problem generation scripts must be modified to include
   higher order defects.
-  **Functionality no explicitly considered (yet):** Models for
   tolerance analysis are usually comprised of 3 sets of equations:
   compatibiliy, interface and functionality. The latter has not been
   included here, but can be manually added to the set of equations
   modeling the interfaces, as a mock interface inequality.
-  **Computational Scaling & Surrogates:** Standard Monte Carlo studies
   (:math:`10^6` trials) scale well through the linear programming
   backend and typically complete in under an hour via ``joblib``
   parallel processing. The neural network-based surrogate modeling
   package (``otaf.surrogate``) is provided to construct a high
   performance surrogate (a simple MLP structure seem to work well for
   simple examples) to reduce probability estimation time during credal
   set explorations.
-  **Cylinder Processing Workaround:** An issue exists regarding the
   automatic cylinder-to-cylinder interactions handling and still needs
   to be resolved. Users analyzing complex multi-primitive cylindrical
   joints should bypass automated generation scripts and construct the
   assembly model manually, as shocased in the 50-DOF reference model
   located in ``src/otaf/example_models/models_3_D/``.

.. raw:: html

   <p align="right"><a href="#readme-top">back to top</a></p>

Roadmap
-------

OTAF is functional for standard planar operations and manual
Torsor-based matrix allocations. The active development roadmap targets
upgrading to automated generation of graph stack-ups, and develop
integration with classic CAD formats:

-  ☐ **Graph-Theoretic Topology Engines:** Implement automated
   closed-loop kinematic path detection using topological graph theory
   to eliminate manual loop configurations.
-  ☐ **Generalized Surface Joint Parsers:** Abstract boundary interface
   mechanics to support direct geometric contact extraction across
   arbitrary matching surface normals.
-  ☐ **Intrinsic Dimension Variability:** Support intrinsic component
   variance tracking (e.g., individual part radius changes) as
   stochastically independent parameters alongside rigid-body
   displacement profiles.
-  ☐ **Part-Internal Joint Tracking:** Enable the generation of isolated
   clearance loops for interdependent matching surfaces residing
   entirely inside a single complex physical part.
-  ☐ **CAD Ecosystem Integration:** Develop open APIs to directly
   extract geometric coordinates and surface topology bounds from modern
   open CAD formats.

.. raw:: html

   <p align="right"><a href="#readme-top">back to top</a></p>

Contributing
------------

Contributions to refine algorithmic stability or expand analytical
capabilities are welcome.

1. Fork the Project
2. Create your Feature Branch
   (``git checkout -b feature/AmazingFeature``)
3. Commit your Changes (``git commit -m 'Add some AmazingFeature'``)
4. Push to the Branch (``git push origin feature/AmazingFeature``)
5. Open a Pull Request

License
-------

Distributed under the **MIT License**. See ``LICENSE`` for more
information.

Contact
-------

**Kristof A. S.** (`@Kramer84 <https://github.com/Kramer84>`__) For bug
reports, feature requests, or scientific inquiries, please open an issue
on the repository tracking system.

**Project Link:** https://github.com/Kramer84/otaf

.. raw:: html

   <p align="right"><a href="#readme-top">back to top</a></p>

Acknowledgments
---------------

This structural research framework was funded and supported by the
**French National Research Agency (ANR)** under project **TRIP:
ToleRance analysis with Imprecise Probabilities** (Grant Number:
`ANR-21-CE46-0009 <https://anr.fr/Projet-ANR-21-CE46-0009>`__). The
framework aims to implement innovative formalisms for statistical
tolerancing, helping bridge the gap between abstract epistemic
uncertainty theory and industrial mechanical applications.

## Mathematical & Literature References

The theoretical, mathematical, and optimization concepts executed inside
OTAF are founded on advanced tolerance research and epistemic
uncertainty frameworks. To study the underlying principles of
statistical tolerance analysis, statistical process control, and
imprecise probabilities, explore the following foundation references:

-  **Over-Constrained System Foundations**

   -  *Citation:* Dumas, A. (2014). *Développement de méthodes
      probabilistes pour l’analyse des tolérances des systèmes
      mécaniques sur-contraints* (Doctoral dissertation, École nationale
      supérieure d’arts et métiers - ENSAM).
   -  *Links:* `[HAL Open Access
      Archive] <https://tel.archives-ouvertes.fr/tel-01177079>`__ \|
      Copy ID: ``tel-01177079``

-  **Quantified Constraint Satisfaction & Convex Hulls**

   -  *Citation:* Dantan, J.-Y., & Qureshi, A.-J. (2009). Worst-case and
      statistical tolerance analysis based on quantified constraint
      satisfaction problems and Monte Carlo simulation. *Computer-Aided
      Design*, 41(1), 1–12.
   -  *Links:* `[Resolve via
      DOI] <https://doi.org/10.1016/j.cad.2008.11.003>`__ \| Copy DOI:
      ``10.1016/j.cad.2008.11.003``

-  **Advanced Probability-Based Tolerance Analysis (APTA Framework)**

   -  *Citation:* Gayton, N., Beaucaire, P., Bourinet, J.-M., Duc, E.,
      Lemaire, M., & Gauvrit, L. (2011). APTA: advanced
      probability-based tolerance analysis of products. *Mécanique &
      Industries*, 12(2), 71–85.
   -  *Links:* `[Resolve via
      DOI] <https://doi.org/10.1051/meca/2011014>`__ \| Copy DOI:
      ``10.1051/meca/2011014``

-  **Imprecise Probabilities in Structural Mechanics**

   -  *Citation:* Beer, M., Ferson, S., & Kreinovich, V. (2013).
      Imprecise probabilities in engineering analyses. *Mechanical
      Systems and Signal Processing*, 37(1-2), 4–29.
   -  *Links:* `[Resolve via
      DOI] <https://doi.org/10.1016/j.ymssp.2013.01.024>`__ \| Copy DOI:
      ``10.1016/j.ymssp.2013.01.024``

-  **Epistemic Tolerance Representation Under Indeterminacy**

   -  *Citation:* Simády, K. A., Beaurepaire, P., & Gayton, N. (2023).
      Imprecise Probabilities as an Answer to the Indeterminacy Inherent
      to Mechanical Tolerances. *Proceedings of EURODYN 2023*, 377–390.
   -  *Links:* `[Resolve via
      DOI] <https://doi.org/10.7712/120223.10344.19792>`__ \| Copy DOI:
      ``10.7712/120223.10344.19792``

--------------

.. raw:: html

   <p align="right"><a href="#readme-top">back to top</a></p>

.. |Contributors| image:: https://img.shields.io/github/contributors/Kramer84/otaf.svg?style=for-the-badge
   :target: https://github.com/Kramer84/otaf/graphs/contributors
.. |Forks| image:: https://img.shields.io/github/forks/Kramer84/otaf.svg?style=for-the-badge
   :target: https://github.com/Kramer84/otaf/network/members
.. |Stargazers| image:: https://img.shields.io/github/stars/Kramer84/otaf.svg?style=for-the-badge
   :target: https://github.com/Kramer84/otaf/stargazers
.. |Issues| image:: https://img.shields.io/github/issues/Kramer84/otaf.svg?style=for-the-badge
   :target: https://github.com/Kramer84/otaf/issues
.. |MIT License| image:: https://img.shields.io/github/license/Kramer84/otaf.svg?style=for-the-badge
   :target: https://github.com/Kramer84/otaf/blob/main/LICENSE
