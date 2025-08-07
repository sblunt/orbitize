from setuptools import setup, Extension
import numpy
from Cython.Build import cythonize


def get_extensions():
    extensions = []
    extensions = cythonize(
        [
            Extension(
                "orbitize._kepler",
                ["orbitize/_kepler.pyx"],
                include_dirs=[numpy.get_include()],
            )
        ]
    )
    return extensions


setup(
    ext_modules=get_extensions(),
)
