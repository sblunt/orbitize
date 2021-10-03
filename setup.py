from setuptools import setup, find_packages, Extension
import numpy, sys
import re

USE_C_KEPLER_MODULE = 1
if ("--disable-cython" in sys.argv):
    sys.argv.remove("--disable-cython")
    USE_C_KEPLER_MODULE = 0
else:
    try:
        from Cython.Build import cythonize
    except:
        print("Error: Importing cython build environment failed")
        USE_C_KEPLER_MODULE = 0

# auto-updating version code stolen from RadVel
def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)

def get_requires():
    reqs = []
    for line in open('requirements.txt', 'r').readlines():
        reqs.append(line)
    return reqs

def get_extensions():
    extensions = []
    if(USE_C_KEPLER_MODULE):
        extensions = cythonize([Extension("orbitize._kepler", ["orbitize/_kepler.pyx"], include_dirs=[numpy.get_include()])])
    return extensions

setup(
    name='orbitize',
    version=get_property('__version__', 'orbitize'),
    description='orbitize! Turns imaging data into orbits',
    url='https://github.com/sblunt/orbitize',
    author='',
    author_email='',
    license='BSD',
    packages=find_packages(),
    package_data={"":["kernels/*.cu"]},
    ext_modules=get_extensions(),
    include_dirs=[numpy.get_include()],
    include_package_data = True,
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        ],
    keywords='Orbits Astronomy Astrometry',
    install_requires=get_requires()
    )
