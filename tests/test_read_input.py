import os
from orbitize.read_input import read_formatted_file


def test_read_formatted_file():
    """
    Test the read_formatted_file function using the test_val.csv file
    """
    testdir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(testdir, 'test_val.csv')
    print(read_formatted_file(input_file))

def test_read_write_orbitize_input():
    """
    Test the read_orbitize_input and the write_orbitize_input functions
    """
