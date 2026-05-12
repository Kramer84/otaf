import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="otaf.example_models.models_3_D._model_3D_30_dof_cython",
        sources=["src/otaf/example_models/models_3_D/_model_3D_30_dof_cython.pyx"],
        include_dirs=[numpy.get_include()] # Required if your .pyx file uses cimport numpy
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)
