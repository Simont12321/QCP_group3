import numpy as np
from Sparse import SparseMatrix
from Dense import DenseMatrix
from Apply_File import Apply
from Gate_File import Gate
from Q_Register_File import Q_Register
from Tensor import TensorProduct


# initialize a 2 qbit register 
register1 = Q_Register(2)
#print(f"qubits are {register1}.")

# create a hadamard gate to act on register1
gate1 = Gate(DenseMatrix, register1, hadamard) 

# apply the hadamard gate to register1
register1.apply_gate(gate1) 