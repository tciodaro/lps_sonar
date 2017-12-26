

# Check if neural net is in the path
import os
lib = '_neuralnet.so'
found = False
for path in os.environ["PYTHONPATH"].split(os.pathsep):
    path = path.strip('"')
    lib_file = os.path.join(path, lib)
    if os.path.exists(lib_file):
       found = True
       break
if not found:
    raise Exception('NeuralNet not in PYTHONPATH')



