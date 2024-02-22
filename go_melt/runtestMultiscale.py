import os
import json
from createPath import *
from testMultiscale import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
try:
    # Run on single GPU
    DEVICE_ID = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    os.system("clear")
except:
    # Run on CPU
    import jax; jax.config.update('jax_platform_name','cpu')
    os.system("clear")
    print('No GPU found.')

# Load input file
input_file = "../examples/sample_input_1.json"
# input_file = "../examples/sample_input_2.json"

with open(input_file, "r") as read_file:
    solver_input = json.load(read_file)

# Create laser path text file
createPath(solver_input)

# Run GO-MELT
testMultiscale(solver_input)
