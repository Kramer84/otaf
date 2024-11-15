import sys
import os
from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

setup(
    name="otaf",  # Updated package name
    version="0.1",
    description="Open (mechanical) tolerance analysis framework (OTAF) - A comprehensive framework for modeling overconstrained mechanical assemblies and doing statistical tolerance analysis.",
    author="Kristof Attila Simady",
    author_email="ksimady@sigma-clermont.fr",
    url="https://github.com/Kramer84/ANR_TRIP",
    license="MIT",
    # define packages which can be imported
    packages=find_packages(), #["otaf"],
    #package_dir={"otaf": "otaf"},
    long_description=readme + "\n\n" + """
In mechanical design, tolerances are defined to constrain the acceptable deviations of a systems features, and to satisfy a set of quality and functional requirements. This is generally done statistically, by estimating probabilities of failure. In regard to the approaches developed for the modeling of mechanical assemblies and the standardized ways of representing tolerances, there is a lack in approaches to simulate and estimate the full scope of effects of a given tolerance choice. This is due to the fact that tolerances are used to define the acceptable geometric space for the defects, but leave an indeterminacy concerning their distribution and nature. In practice, designers make assumptions, which can lead to either over-, or under-estimate the probability of failure. In this paper, work has been done to put in evidence the effect of the ambiguity in the graphical language of tolerances on the results obtained in tolerance analysis, and proposes the usage of the imprecise probability framework to complement these studies and obtain more realistic results.
""",
    install_requires=[
        "numpy",
        "sympy",
        "openturns",
        "scipy",
        "matplotlib",
        "joblib",
        "beartype",
        "trimesh",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",  # Updated development status
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
    ],
)
