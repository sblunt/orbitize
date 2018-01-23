import os
from orbitize.read_input import read_csv


def test_read_formatted_file():
    """
    Test the read_formatted_file function using the test_val.csv file
    """
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')
    print(read_csv(input_file))
