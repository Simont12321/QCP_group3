import numpy as np
from Sparse import SparseMatrix
from Dense import DenseMatrix
from Apply_File import Apply
from Gate_File import Gate
from Q_Register_File import Q_Register
from Tensor import TensorProduct


# initialize 2 qbit register 

test1 = Q_Register(2)
#print(f"qubits are {test1}.")

gate1 = Gate(DenseMatrix, test1, )