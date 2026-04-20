import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="otaf.example_models.model_dumas_cython.plate_2_pins",
        sources=["src/otaf/example_models/model_dumas_cython/plate_2_pins.pyx"],
        include_dirs=[numpy.get_include()] # Required if your .pyx file uses cimport numpy
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)
