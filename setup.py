from setuptools import setup, Extension
import numpy
from Cython.Build import cythonize

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

def get_extensions():
    extensions = []
    extensions = cythonize(
        [
            Extension(
                "orbitize._kepler",
                ["orbitize/_kepler.pyx"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                "orbitize._kepler2",
                ["orbitize/_kepler2.pyx"],
                include_dirs=[numpy.get_include()],
            )
        ],
        annotate=True
    )
    return extensions


setup(
    ext_modules=get_extensions(),
)
