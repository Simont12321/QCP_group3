import numpy as np
# from Sparse import SparseMatrix
# #from Dense import DenseMatrix
# import Apply_File
#import Gate
from Q_Register_File import Q_Register
#import Tensor


# initialize 2 qbit register 

test1 = Q_Register(2)
#print(f"qubits are {test1}.")

gate1 = Gate(DenseMatrix, test1, )