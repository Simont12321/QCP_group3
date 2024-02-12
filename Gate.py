import numpy as np
#import SparseMatrix
import DenseMatrix
#import LazyMatrix

class Gate(object):
    
    def __init__(self, matrixType, quBitOne, gateName, quBitTwo = None, customInput = None):
    
        #Store the one (or two) quBits we're acting on
        self.quBitOne = quBitOne
        self.quBitTwo = quBitTwo
        
        self.gateName = gateName
        self.customInput = customInput
        
        self.Gate = self.gateName()
        
        if matrixType == "Dense":
            self.matrixType = DenseMatrix
       # elif matrixType == "Sparse":
       #     self.matrixType = SparseMatrix
       # elif matrixType == "Lazy":
       #     self.matrixType = LazyMatrix
        
    
    def hadamard(self):
        
        hadamard = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
        
        return self.matrixType(hadamard)
    
    def cNot(self):
        
        cNot = np.array([1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0])
        
        return self.matrixType(cNot)
    
    def cV(self):
        
        cV = np.array([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1j])
        
        return self.matrixType(cV)
    
    def custom(self):
        
        custom = np.array(self.customInput)
        
        return self.matrixType(custom)
    
    def spinX(self):
        
        spinX = np.array([0,1],[1,0])
        
        return self.matrixType(spinX)
    
    def spinY(self):
        
        spinY = np.array([0,-1j],[1j,0])
        
        return self.matrixType(spinY)
    
    def spinZ(self):
        
        spinZ = np.array([1,1],[0,-1])
        
        return self.matrixType(spinZ)
