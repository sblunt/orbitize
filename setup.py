from setuptools import setup, find_packages
import re
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import build_ext
import numpy
import sys




# auto-updating version code stolen from RadVel
def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)

def get_ext_modules():
    return [Extension(
        name="orbitize._kepler",
        sources=["orbitize/_kepler.pyx", "orbitize/kepler.cc"],
            # extra_objects=["fc.o"],  # if you compile fc.cpp separately
        include_dirs = [numpy.get_include()],  # .../site-packages/numpy/core/include
        language="c++"
        )]

setup(
    name='orbitize',
    version=get_property('__version__', 'orbitize'),
    description='orbitize! Turns imaging data into orbits',
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
    install_requires=['numpy', 'scipy', 'astropy', 'emcee','cython'],
    setup_requires=['cython>=0.x',],
    ext_modules = get_ext_modules(),
    cmdclass = {'build_ext': build_ext}
    )