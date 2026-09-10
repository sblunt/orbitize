import os

__version__ = "3.4.0"

# set Python env variable to keep track of example data dir
orbitize_dir = os.path.dirname(__file__)
DATADIR = os.path.join(orbitize_dir, "example_data/")

try:
    from . import _kepler

    cext = True
except ImportError:
    cext = False
