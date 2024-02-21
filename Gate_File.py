import numpy as np
from Sparse import SparseMatrix
from Dense import DenseMatrix
from LazyMatrix_File import LazyMatrix


class Gate(object):

    def __init__(self, matrixType, quBits, gateName=None, customInput=None):
        # TODO: let's say the matrixTypes are "Sparse", "Dense", ???"Lazy"???

        # Store the quBits we're acting on
        self.quBits = quBits
        self.customInput = customInput

        self.matrixType = matrixType

        self.getGate(gateName)

        if self.customInput != None:
            self.gateMethod = self.custom

        self.Gate = self.gateMethod()

    def getGate(self, gateName):

        if gateName == "hadamard":
            self.gateMethod = self.hadamard
        elif gateName == "cNot":
            self.gateMethod = self.cNot
        elif gateName == "cV":
            self.gateMethod = self.cV
        elif gateName == "spinX":
            self.gateMethod = self.spinX
        elif gateName == "spinY":
            self.gateMethod = self.spinY
        elif gateName == "spinZ":
            self.gateMethod = self.spinZ

    def hadamard(self):

        hadamard = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])

        return self.makeGate(hadamard)

    def cNot(self):

        cNot = np.array([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0])

        return self.makeGate(cNot)

    def cV(self):

        cV = np.array([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j])

        return self.makeGate(cV)

    def custom(self):

        custom = np.array(self.customInput)

        return self.makeGate(custom)

    def spinX(self):

        spinX = np.array([0, 1], [1, 0])

        return self.makeGate(spinX)

    def spinY(self):

        spinY = np.array([0, -1j], [1j, 0])

        return self.makeGate(spinY)

    def spinZ(self):

        spinZ = np.array([1, 1], [0, -1])

        return self.makeGate(spinZ)

    def makeGate(self, matrix):

        self.Gate = DenseMatrix(matrix)

        if self.matrixType == "Sparse":
            self.Gate.Sparse()
