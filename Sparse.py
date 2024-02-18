import numpy as np
from Dense import DenseMatrix
class SparseMatrix(object):

    def __init__(self, n, elements):
        self.Dimension = n
        self.Elements = elements

    def Multiply(self, matrix2):
        """
        Input
        ------
        Sparse matrix. Each sparse matrix is a list of tuples, and every tuple has three 
        elements (i,j,Mij) giving the row, column, and value of each entry. All other matrix entries are 0. 

        Returns
        -------

        The product of the matrix with the input in sparse matrix format. 

        """
        
        result = []
        
        for i in range(len(self.Elements)):
            for j in range(len(matrix2.Elements)):
                if self.Elements[i][1] == matrix2.Elements[j][0]:
                    product = self.Elements[i][2]*matrix2.Elements[j][2]
                    result.append((self.Elements[i][0],matrix2.Elements[j][1],product)) 
        return SparseMatrix(self.Dimension, result)
    
    def SparseApply(self, u):
        """
        Input
        ------
        vector u

        Implementation for sparse matrix is different from the dense matrix. Sparse matrix is a list of elements of the form [i,j,Mij], corresponding to rows, columns and values. 
        The multiplication is done value by value. Elements in the first row of the matrix (indexed by row number), are multiplied with vector elements to calculate the first element
        of NewVector. This is repeated for each row. Speed should be faster than dense matrix if the matrix has lots of 0s.

        Returns
        ------
        NewVector : The matrix applied to the target vector
        The output will be a vector itself
        """
    # Start product
        NewVector = np.zeros_like(u, dtype = complex)

    # Loop over matrix elements and vector elements to add to NewVector

        for elem in self.Elements:
            NewVector[elem[0]] += (elem[2])*(u[elem[1]])
        return(NewVector)
    
    def Dense(self):
        """
        Output
        ------
        The dense matrix representation of the sparse matrxi
        """
        Dense = np.zeros((self.Dimension, self.Dimension), dtype = complex)
        for i in self.Elements:
            Dense[i[0]][i[1]] = i[2]
        return DenseMatrix(Dense)

    def __getitem__(self,index):
        """
        Gets the value of a given index.
        """
        row, column = index
        for i in self.Elements:
            if i[0] == row and i[1] == column:
                return(i[2])
        return(complex(0))

Matrix = SparseMatrix(3,[(0,0,2),(1,2,3),(2,2,1)])
NewMatrix = Matrix.Multiply(Matrix)
print(NewMatrix)
print(NewMatrix.Dense())

    