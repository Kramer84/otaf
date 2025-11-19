OTAF: Open Tolerance Analysis Framework
=======================================

📖 Table of Contents
--------------------

1. `Introduction <#introduction>`__
2. `Installation <#installation>`__
3. `Getting Started <#getting-started>`__
4. `Modules <#modules>`__
5. `Examples <#examples>`__
6. `Contributing <#contributing>`__
7. `License <#license>`__
8. `Acknowledgments <#acknowledgments>`__

--------------

📚 Introduction
---------------

**OTAF** (Open Tolerance Analysis Framework) is a research oriented
library designed to perform statistical tolerance analysis of mechanical
assemblies. It can be used for iso- and over- constrained rigid
mechanical assemblies, without large displacements. It allows for the
semi-automatic construction a System of Constraints (SOC)-based assembly
model, given a minimal representation of the assembly and its components
as well as hypotheses.

This model can then be used to perform statistical tolerance analysis,
by modeling defects as random variables, and estimating the probability
of non assembly using different estimation techniques.

These codes have been developed during a PhD called TRIP (ToleRance
analysis with Imprecise Probabilities), where the defect distributions
are modelled using imprecise probabilities, and the notebooks, (once
currated) will be from the examples in the paper.

Key use cases include: - **Construction of a SOC Model**: Automatically
construct the linear problem modeling the assembly. - **Modeling
Manufacturing Deviations**: Analyze the effects of geometric and
dimensional variations. - **Probability of Failure Estimation**: Use
advanced statistical tools to quantify and mitigate risks (Monte Carlo,
FORM, SORM, GLD). - **Imprecise Probability of Failure Estimation**:
Apply methods developed in an upcoming paper [COMING SOON] to model the
space of possible defects.

Limitations
~~~~~~~~~~~

-  **Only Rigid Transformations**: Only rigid defects (translations and
   rotations) are modeled.
-  **Limited Feature Types**: Currently supports only planar and
   cylindrical features.
-  **No Tools for Tolerance Allocation**: Users must implement this on
   their own after generating the SOC model.
-  **Not Production-Ready**: Apart from SOC model construction, the rest
   of the program is in an alpha stage and may remain so. Use at your
   discretion.

--------------

🚀 Installation
---------------

From Source
~~~~~~~~~~~

Do this inside of a virtual environment or conda :

.. code:: bash

   # Clone the repository
   git clone https://github.com/Kramer84/otaf.git

   # Navigate to the project directory
   cd otaf

   # Install the package
   pip install .

   # Install the package with documentation build dependencies
   pip install .[docs]

Dependencies
~~~~~~~~~~~~

The following Python libraries are required: - **Base**: ``joblib``,
``beartype`` - **Scientific**: ``numpy``, ``scipy``, ``sympy`` -
**Uncertainty Quantification**: ``openturns`` - **Visualization**:
``matplotlib``, ``trimesh`` - **Machine Learning**: ``torch``,
``torcheval``, ``scikit-learn`` - **Geometry**: ``triangle``,
``pytransform3d[all]``

--------------

🛠️ Getting Started
------------------

General Form Of The Assembly Data Dictionary To Pass to the ``AssemblyDataProcessor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``AssemblyDataProcessor`` takes as an input a minimal representation
of the assembly in the form of a dictionary, that needs to be
constructed manually. Some specific notation choices have been made on
how to represent features and assemblies. Parts are always represented
by a sequence of numbers, surfaces always by a sequence of lowercase
letter, and points on the surface always as the same sequence as for the
surface in uppercase and an identifying sequence of numbers after it.

So for Part number 123 on surface abc and point ABC321 can be written
like this : P123abcABC321

The point with index 0 is always the origin point of the feature! This
notation is kinda arbitrary, and may be updated if something better is
proposed.

The system data dictionary may look like this :

.. code:: python

   system_data = {
       "PARTS" : {
           '1' : {
               "a" : {
                   "FRAME": RP1a, # Array of size 3x3 with each component being the x,y,z unit vectors for the local frame of the surface (x positive outwards for planar features)
                   "POINTS": {'A0' : P1A0, 'A1' : P1A1, 'A2' : P1A2}, # Dictionary mapping point names to the coordinate as numpy array
                   "TYPE": "plane", # Type of the feature
                   "INTERACTIONS": ['P2a'], # The other feature this feature is interacting with
                   "CONSTRAINTS_D": ["PERFECT"], # We can constrain the defect vector D to be 0 so there is no defect modelled on this feature
                   "CONSTRAINTS_G": ["FLOATING"], # We can constrain components on the gap vector G at the feature level
               },
           ...
           ...
           },
           '2' : {
               "a" : {
                   "FRAME": RP2a,
                   "POINTS": {'A0' : P2A0, 'A1' : P2A1, 'A2' : P2A2},
                   "TYPE": "plane",
                   "INTERACTIONS": ['P1a'],
                   "CONSTRAINTS_D": ["PERFECT"], # In this modelization, only defects on the right side
                   "CONSTRAINTS_G": ["FLOATING"],
               },
           ...
           ...
           }
       },
       "LOOPS": {
           "COMPATIBILITY": {
               "L0": "P1cC0 -> P2cC0 -> P2aA0 -> P1aA0", #Minimal representation of the compatibility loops
               "L1": "P1cC0 -> P2cC0 -> P2bB0 -> P1bB0",
           },
       },
       "GLOBAL_CONSTRAINTS": "2D_NZ",
   }

Information about the different values that the keys can take refer to
the source code in ``constants.py``.

Documentation
~~~~~~~~~~~~~

Visit the `OTAF Documentation <https://kramer84.github.io/otaf/>`__ for
detailed guides, examples, and API references.

**TO COMPILE THE DOCUMENTATION LOCALLY INSTALL PACKAGE WITH [docs] FLAG,
THEN:**

.. code:: bash

   sudo apt install pandoc

   sphinx-apidoc -o docs/source src/otaf

   cd docs/

   make clean
   make html

--------------

🗂️ Modules
----------

OTAF is modular and extensible, with the following key components:

-  **``assembly_modeling``**: Base classes for mechanical assemblies.
   **MOST IMPORTANT MODULE**
-  **``geometry``**: Geometric functions for validating geometry, point
   clouds, etc.
-  **``optimization``**: Tools for solving SOC optimization with diverse
   approaches.
-  **``sampling``**: Sampling distributions of defects or defect
   parameters.
-  **``distribution``**: Modeling distributions based on
   ``assembly_modeling`` objects and other specialized tools.
-  **``surrogate``**: Experimental methods to create surrogate models
   for the SOC.
-  **``uncertainty``**: Methods for reliability analyses and failure
   probability estimation.

Explore the `source code <https://github.com/Kramer84/otaf/src/otaf/>`__
for a complete overview.

--------------

📦 Examples
-----------

Explore the ``NOTEBOOKS/`` directory for scripts demonstrating OTAF’s
capabilities. **THE NOTEBOOKS HAVE YET TO BE CLEANED UP**

--------------

🤝 Contributing
---------------

We welcome contributions! Whether you’re reporting a bug, suggesting a
feature, or submitting a pull request, your help is appreciated. There
is still significant work to do to support a broader variety of cases.
As the product of three years of PhD research, some parts of the code
may not yet be fully optimized for adding new feature types or handling
form defects. Legacy code for experimental tools, such as neural
network-based surrogate models, may also require further refinement.

--------------

📜 License
----------

This project is licensed under the **MIT License**. For details, see the
`LICENSE <https://github.com/Kramer84/otaf/blob/main/LICENSE>`__ file.

--------------

🌟 Acknowledgments
------------------

This work is supported by the `French National Research Agency
(ANR) <https://anr.fr/Projet-ANR-21-CE46-0009>`__ under the project
“Analyse des tolérances avec les probabilités imprécises”
(ANR-21-CE46-0009). The goal is to develop new formalisms for tolerance
analysis based on imprecise probabilities, bridging the gap between
theory and industrial applications.
