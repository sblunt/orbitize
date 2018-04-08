from setuptools import setup, find_packages
import re
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

# auto-updating version code stolen from RadVel
def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)

ext_modules = [Extension(
    name="kepler",
    sources=["orbitize/kepler.pyx", "orbitize/_kepler.cc"],
        # extra_objects=["fc.o"],  # if you compile fc.cpp separately
    include_dirs = [numpy.get_include()],  # .../site-packages/numpy/core/include
    language="c++",
        # libraries=
        # extra_compile_args = "...".split(),
        # extra_link_args = "...".split()
    )]

setup(
    name='orbitize',
    version=get_property('__version__', 'orbitize'),
    description='orbitize! Turns imaaging data into orbits',
    url='https://github.com/sblunt/orbitize',
    author='',
    author_email='',
    license='BSD',
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        ],
    keywords='Orbits Astronomy Astrometry',
    install_requires=['numpy', 'scipy', 'astropy', 'emcee'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
    )