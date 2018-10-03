from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy, sys
import re

# auto-updating version code stolen from RadVel
def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)

extensions = [Extension("orbitize._kepler", ["orbitize/_kepler.pyx"])]

setup(
    name='orbitize',
    version=get_property('__version__', 'orbitize'),
    description='orbitize! Turns imaging data into orbits',
    url='https://github.com/sblunt/orbitize',
    author='',
    author_email='',
    license='BSD',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
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
    install_requires=['numpy', 'scipy', 'astropy', 'emcee', 'ptemcee', 'cython', 'corner', 'pytest>=3.0.0', 'h5py']
    )
