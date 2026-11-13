import importlib.util
import os

__version__ = "4.0.0"

# set Python env variable to keep track of example data dir
orbitize_dir = os.path.dirname(__file__)
DATADIR = os.path.join(orbitize_dir, "example_data/")


spec = importlib.util.find_spec("._kepler", package=__name__)

if spec is None:
    cext = False
else:
    cext = True
