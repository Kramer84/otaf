# OTAF: Open Tolerance Analysis Framework

<div align="left">
    <img src="logo/logo.png" alt="OTAF Logo" width="400px">
</div>

## 📖 Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
# OTAF: Open Tolerance Analysis Framework

<div align="left">
    <img src="logo/logo.png" alt="OTAF Logo" width="400px">
</div>

## 📖 Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Modules](#modules)
5. [Examples](#examples)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

---

## 📚 Introduction

**OTAF** (Open Tolerance Analysis Framework) is an open-source Python library designed for statistical tolerance analysis and modeling of overconstrained 3D mechanical assemblies. It provides tools to model defects at the feature level, perform reliability analysis, and estimate the statistical effect of tolerance choices on the variability of quantities of interest (primarily the probability of non-assembly). Additionally, OTAF offers a foundation for constructing a System of Constraints (SOC)-based assembly model automatically, given a minimal representation of the assembly and its components.

Key use cases include:
- **Construction of a SOC Model**: Automatically construct the linear problem modeling the assembly.
- **Modeling Manufacturing Deviations**: Analyze the effects of geometric and dimensional variations.
- **Probability of Failure Estimation**: Use advanced statistical tools to quantify and mitigate risks (Monte Carlo, FORM, SORM, GLD).
- **Imprecise Probability of Failure Estimation**: Apply methods developed in an upcoming paper [COMING SOON] to model the space of possible defects.

### Limitations
- **Only Rigid Transformations**: Only rigid defects (translations and rotations) are modeled.
- **Limited Feature Types**: Currently supports only planar and cylindrical features.
- **No Tools for Tolerance Allocation**: Users must implement this on their own after generating the SOC model.
- **Not Production-Ready**: Apart from SOC model construction, the rest of the program is in an alpha stage and may remain so. Use at your discretion.

This package was developed to explore experimental approaches based on imprecise probabilities. For more details, refer to the associated paper.

---

## 🚀 Installation

### From Source
```bash
# Clone the repository
git clone https://github.com/Kramer84/otaf.git

# Navigate to the project directory
cd otaf

# Install the package
pip install .

# Install the package with documentation build dependencies
pip install .[docs]
```

### Dependencies
The following Python libraries are required:
- **Core**: `numpy`, `scipy`, `sympy`, `joblib`, `beartype`
- **Uncertainty Tools**: `openturns`
- **Visualization**: `matplotlib`, `trimesh`
- **Machine Learning**: `torch`, `torcheval`, `scikit-learn`
- **Geometry**: `triangle`, `pytransform3d[all]`

Refer to [requirements.txt](requirements.txt) for a complete list.

---

## 🛠️ Getting Started

### Importing the Library
After installation, you can use OTAF as follows:
```python
import otaf

# Example: Running a tolerance analysis
results = otaf.run_analysis(input_data)
```

### Documentation
Visit the [OTAF Documentation](https://github.com/Kramer84/otaf/wiki) for detailed guides, examples, and API references.
[View Documentation](https://kramer84.github.io/otaf/)

---

## 🗂️ Modules

OTAF is modular and extensible, with the following key components:

- **`assembly_modeling`**: Base classes for mechanical assemblies.
- **`geometry`**: Geometric functions for validating geometry, point clouds, etc.
- **`optimization`**: Tools for solving SOC optimization with diverse approaches.
- **`sampling`**: Sampling distributions of defects or defect parameters.
- **`distribution`**: Modeling distributions based on `assembly_modeling` objects and other specialized tools.
- **`surrogate`**: Experimental methods to create surrogate models for the SOC.
- **`uncertainty`**: Methods for reliability analyses and failure probability estimation.

Explore the [source code](https://github.com/Kramer84/otaf/src/otaf/) for a complete overview.

---

## 📦 Examples

Explore the `NOTEBOOKS/` directory for scripts demonstrating OTAF's capabilities.

---

## 🤝 Contributing

We welcome contributions! Whether you're reporting a bug, suggesting a feature, or submitting a pull request, your help is appreciated. There is still significant work to do to support a broader variety of cases. As the product of three years of PhD research, some parts of the code may not yet be fully optimized for adding new feature types or handling form defects. Legacy code for experimental tools, such as neural network-based surrogate models, may also require further refinement.

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. For details, see the [LICENSE](https://github.com/Kramer84/otaf/blob/main/LICENSE) file.

---

## 🌟 Acknowledgments

This work is supported by the [French National Research Agency (ANR)](https://anr.fr/Projet-ANR-21-CE46-0009) under the project "Analyse des tolérances avec les probabilités imprécises" (ANR-21-CE46-0009). The goal is to develop new formalisms for tolerance analysis based on imprecise probabilities, bridging the gap between theory and industrial applications.
3. [Getting Started](#getting-started)
4. [Modules](#modules)
5. [Examples](#examples)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

---

## 📚 Introduction

**OTAF** (Open Tolerance Analysis Framework) is an open-source Python library designed for statistical tolerance analysis and modeling of overconstrained 3D mechanical assemblies. It provides tools to model defects at the feature level, perform reliability analysis, and estimate the statistical effect of tolerance choices on the variability of quantities of interest (primarily the probability of non-assembly). Additionally, OTAF offers a foundation for constructing a System of Constraints (SOC)-based assembly model automatically, given a minimal representation of the assembly and its components.

Key use cases include:
- **Construction of a SOC Model**: Automatically construct the linear problem modeling the assembly.
- **Modeling Manufacturing Deviations**: Analyze the effects of geometric and dimensional variations.
- **Probability of Failure Estimation**: Use advanced statistical tools to quantify and mitigate risks (Monte Carlo, FORM, SORM, GLD).
- **Imprecise Probability of Failure Estimation**: Apply methods developed in an upcoming paper [COMING SOON] to model the space of possible defects.

### Limitations
- **Only Rigid Transformations**: Only rigid defects (translations and rotations) are modeled.
- **Limited Feature Types**: Currently supports only planar and cylindrical features.
- **No Tools for Tolerance Allocation**: Users must implement this on their own after generating the SOC model.
- **Not Production-Ready**: Apart from SOC model construction, the rest of the program is in an alpha stage and may remain so. Use at your discretion.

This package was developed to explore experimental approaches based on imprecise probabilities. For more details, refer to the associated paper.

---

## 🚀 Installation

### From Source
```bash
# Clone the repository
git clone https://github.com/Kramer84/otaf.git

# Navigate to the project directory
cd otaf

# Install the package
pip install .

# Install the package with documentation build dependencies
pip install .[docs]
```

### Dependencies
The following Python libraries are required:
- **Core**: `numpy`, `scipy`, `sympy`, `joblib`, `beartype`
- **Uncertainty Tools**: `openturns`
- **Visualization**: `matplotlib`, `trimesh`
- **Machine Learning**: `torch`, `torcheval`, `scikit-learn`
- **Geometry**: `triangle`, `pytransform3d[all]`

Refer to [requirements.txt](requirements.txt) for a complete list.

---

## 🛠️ Getting Started

### Documentation
Visit the [OTAF Documentation](https://github.com/Kramer84/otaf/wiki) for detailed guides, examples, and API references.
[View Documentation](https://kramer84.github.io/otaf/)

---

## 🗂️ Modules

OTAF is modular and extensible, with the following key components:

- **`assembly_modeling`**: Base classes for mechanical assemblies.
- **`geometry`**: Geometric functions for validating geometry, point clouds, etc.
- **`optimization`**: Tools for solving SOC optimization with diverse approaches.
- **`sampling`**: Sampling distributions of defects or defect parameters.
- **`distribution`**: Modeling distributions based on `assembly_modeling` objects and other specialized tools.
- **`surrogate`**: Experimental methods to create surrogate models for the SOC.
- **`uncertainty`**: Methods for reliability analyses and failure probability estimation.

Explore the [source code](https://github.com/Kramer84/otaf/src/otaf/) for a complete overview.

---

## 📦 Examples

Explore the `NOTEBOOKS/` directory for scripts demonstrating OTAF's capabilities.

---

## 🤝 Contributing

We welcome contributions! Whether you're reporting a bug, suggesting a feature, or submitting a pull request, your help is appreciated. There is still significant work to do to support a broader variety of cases. As the product of three years of PhD research, some parts of the code may not yet be fully optimized for adding new feature types or handling form defects. Legacy code for experimental tools, such as neural network-based surrogate models, may also require further refinement.

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. For details, see the [LICENSE](https://github.com/Kramer84/otaf/blob/main/LICENSE) file.

---

## 🌟 Acknowledgments

This work is supported by the [French National Research Agency (ANR)](https://anr.fr/Projet-ANR-21-CE46-0009) under the project "Analyse des tolérances avec les probabilités imprécises" (ANR-21-CE46-0009). The goal is to develop new formalisms for tolerance analysis based on imprecise probabilities, bridging the gap between theory and industrial applications.
