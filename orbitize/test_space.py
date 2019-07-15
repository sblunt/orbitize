import numpy as np
import system as sys
import read_input

data = read_input.read_file('/Users/Helios/orbitize/tests/test_val.csv')
print(data)

sys.System(2, data, 3.0, 50)
