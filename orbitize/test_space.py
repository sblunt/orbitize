import numpy as np
import system as sys
import read_input

data = read_input.read_file('/Users/Helios/orbitize/tests/test_val.csv')
print(data)  # read_input will only read 'rv','rv_err' named columns

output_1 = sys.System(2, data, 3.0, 50)

print(output_1.rv0)
print(output_1.radec)
print(output_1.seppa)
