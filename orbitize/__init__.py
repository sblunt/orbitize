import os

__version__ = '2.0b1'

# set Python env variable to keep track of example data dir
orbitize_dir = os.path.dirname(__file__)
DATADIR = os.path.join(orbitize_dir, 'example_data/')