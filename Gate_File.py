import numpy as np
from Sparse import SparseMatrix
from Dense import DenseMatrix
from LazyMatrix_File import LazyMatrix


class Gate(object):

    def __init__(self, matrixType, gateName=None, customInput=None):
        # TODO: let's say the matrixTypes are "Sparse", "Dense", ???"Lazy"???
        self.matrixType = matrixType
        self.gateName = gateName
        self.customInput = customInput
        self.GateMatrix = self.gateMethod()


    def gateMethod(self):    
        if self.gateName == "hadamard":
            gate = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
        elif self.gateName == "cNot":
            gate = np.array([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0])
        elif self.gateName == "cV":
            gate = np.array([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j])
        elif self.gateName == "spinX":
            gate = np.array([0, 1], [1, 0])
        elif self.gateName == "spinY":
            gate = np.array([0, -1j], [1j, 0])
        elif self.gateName == "spinZ":
            gate = np.array([1, 1], [0, -1])
        elif self.gateName == "custom":
            gate = self.customInput

        if self.matrixType == "Dense":
            return DenseMatrix(gate)
        if self.matrixType == "Sparse":
            return DenseMatrix(gate).Sparse()



