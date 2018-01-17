import os
from orbitize.read_input import read_csv


def test_read_csv():
    """
    Test the read_csv function using the test_val.csv file
    """
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')
    print(read_csv(input_file))
