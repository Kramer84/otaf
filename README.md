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

**OTAF** (Open Tolerance Analysis Framework) is an open-source Python library designed for statistical tolerance analysis and modeling of overconstrained 3D mechanical assemblies. It provides tools to model defects at the feature level, perform reliability analysis, and estimate the statistical effect of a tolerance choice on the variaiblity of some quantity of interest (mainly the probability of non assembly here). It also provides a basis for constructing a System Of Constraint (SOC) based assembly model, given a minimal representation of the assembly and its constituting parts, automatically.

Key use cases include:
- **Construction of a SOC model**: Constrution of the linear problem modelling the assembly
- **Modeling Manufacturing Deviations**: Analyze the effects of geometric and dimensional variations.
- **Probability of Failure Estimation**: Leverage advanced statistical tools to quantify and mitigate risk. (Monte-Carlo / FORM / SORM / GLD)
- **Imprecise Probability of Failure Esitmation**: Application of the method developed in the paper [COMING SOON] to model the space of possible space of defects.

Limitations :
- **Only rigid transformations**: Only rigid defects (translation + rotations) are modelled
- **Only planar and cylindrical features**: The methods have only been developed for these two geometric objects for now.
- **No tools for tolerance allocation**: But can be easily made on your side once you have the SOC model
- **Not production ready**: Apart from the construction of the SOC model, the rest of the program is in its alpha stage and may stay there, so use at your own discretion.

This package has been written to explore experimental approaches based on imprecise probabilities, read the associated paper. 

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

# Install the package wit documentation build dependencies
pip install .[docs]
```

### Dependencies
The following Python libraries are required:
- **Core**: `numpy`, `scipy`, `sympy`, `joblib`, `beartype`
- **Uncertainty Tools**: `openturns`,
- **Visualization**: `matplotlib`, `trimesh`
- **Machine Learning**: `torch`, `torcheval`, `scikit-learn`
- **Geometry**: `triangle`, `pytransform3d[all]`

Check [requirements.txt](requirements.txt) for a complete list.

---

## 🛠️ Getting Started

### Importing the Library
Once installed, you can start using OTAF as follows:
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
- **`geometry`**: Geometric definitions and operations.
- **`optimization`**: Tools for constrained tolerance optimization.
- **`sampling`**: Low-discrepancy sampling methods.
- **`uncertainty`**: Probabilistic modeling and failure estimation.

Explore the [source code](https://github.com/Kramer84/otaf/src/otaf/) for a complete overview.

---

## 📦 Examples

Explore the `NOTEBOOKS/` directory for scripts demonstrating OTAF's capabilities.

---

## 🤝 Contributing

We welcome contributions! Whether you're reporting a bug, suggesting a feature, or submitting a pull request, your help is appreciated. There is still a lot of work to do to handle a broader variety of cases, and being the product of 3 years of PhD, the style may not be the most adapted yet for the easy inclusion of new feature types and/or form defects, and legacy code (for experimental stuff like neural-network based surrogate models).

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. For details, see the [LICENSE](https://github.com/Kramer84/otaf/blob/main/LICENSE) file.

---

## 🌟 Acknowledgments

This work is supported by the [French National Research Agency (ANR)](https://anr.fr/Projet-ANR-21-CE46-0009) under the project "Analyse des tolérances avec les probabilités imprécises" (ANR-21-CE46-0009). The goal is to develop new formalisms for tolerance analysis based on imprecise probabilities, bridging the gap between theory and industrial applications.
