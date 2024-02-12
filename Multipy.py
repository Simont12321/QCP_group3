# import LazyMatrix
# import DenseMatrix
# import SparseMatrix

import numpy as np

class Multiply():

    def __init__(self, matrix1, matrix2, matrixType = None):  # documentation not finished
        
        # Check correct format
        # if matrix1 == numpy.ndarray and matrix2 == numpy.ndarray:
        #     if matrix1.shape[0] == matrix1.shape[1] and matrix3.shape[0] == matrix3.shape[1]:
        #         self.matrixType = "denseMatrix"
        #     else:
        #         self.matrixType = "lazyMatrix"
        # elif matrix1 == numpy.matrix and matrix2 == numpy.matrix:
        #    self.matrixType = "npMatrix"
        # elif matrix1 == list and matrix2 == list:  # still need to check if every element is a tripple
        #    self.matrixType = "sparceMatrix"
        # else:
        #    raise Exception("Unexpected inputs.")
        
        self.matrix1 = matrix1
        self.matrix2 = matrix2
        self.matrixType = matrixType

    def denseMultiply(self):
        """
        Input
        ------
        Not finished documenting 

        Returns
        -------
      

        """

        product = np.asmatrix(self.matrix1) * np.asmatrix(self.matrix2)
        return np.asarray(product) 

    def sparceMultiply(self):
        
        result = []
        
        for i in range(len(self.matrix1)):
            for j in range(len(self.matrix2)):
                if self.matrix1[i][0] == self.matrix2[j][0] and self.matrix1[i][1] == self.matrix2[j][1]:
                    product = self.matrix1[i][2]*self.matrix2[j][2]
                    result.append((self.matrix1[i][0],self.matrix1[i][1],product)) 
        
        return result
        
        

a = np.array([[3,0,1],[1,0,0],[0,0,6]])
b = np.array([[1,0,1],[0,2,0],[1,0,3]])

sparceMatrix1 = [(1,2,3), (4,5,4)]
sparceMatrix2 = [(7,5,6), (1,2,2), (5,2,8)]

AMultiplyB = Multiply(a,b)
denseResult = AMultiplyB.denseMultiply() 
print(f"the dense result is {denseResult}")

lazy1Multiply2 = Multiply(sparceMatrix1,sparceMatrix2)
lazyResult = lazy1Multiply2.sparceMultiply()
print(f"The lazy result is {lazyResult}")
