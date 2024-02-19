# import LazyMatrix
# import DenseMatrix
# import SparseMatrix
from time import time
import numpy as np


class Apply(object):
    def __init__(self, VectorApply, ApplyType=None):
        # Check correct format
        assert isinstance(VectorApply, list)
        self.VectorApply = VectorApply
        self.ApplyType = ApplyType

    def DenseApply(self):
        """
        Input
        ------
        List of dense matrix and vector [M,u]

        Implementation for dense matrices is row multiplication over the vector elements, as usual.
        Function assumes the matrix is square, which should be fine as all matrices we use will be square.
        For matrix M applied to vector u, the element v_1 of the new vector is given by: v_1 = M_11u_1+M_12u_2+...+M_1nu_n.

        Returns
        -------
        NewVector : The matrix applied to the target vector of the list VectorApply.
        The output will be a vector itself.

        """
        # Start product

        NewVector = []
        Matrix = self.VectorApply[0]
        Vector = self.VectorApply[1]

        # Loop over matrix elements and vector elements to find new vector elements and append them

        for i in range(0, len(Matrix)):
            NewVectorelem = 0
            for j in range(0, len(Matrix[i])):
                NewVectorelem += Matrix[i][j]*Vector[j]
            NewVector.append(NewVectorelem)
        NewVector = np.asarray(NewVector, dtype=complex)
        return (NewVector)

    def SparseApply(self):
        """
        Input
        ------
        List of sparse matrix and vector [M,u]

        Implementation for sparse matrix is different from the dense matrix. Sparse matrix is a list of elements of the form [i,j,Mij], corresponding to rows, columns and values. 
        The multiplication is done value by value. Elements in the first row of the matrix (indexed by row number), are multiplied with vector elements to calculate the first element
        of NewVector. This is repeated for each row. Speed should be faster than dense matrix if the matrix has lots of 0s.

        Returns
        ------
        NewVector : The matrix applied to the target vector of the list VectorApply
        The output will be a vector itself
        """
    # Start product

        Matrix = self.VectorApply[0]
        Vector = self.VectorApply[1]
        NewVector = np.zeros_like(Vector, dtype=complex)

    # Loop over matrix elements and vector elements to add to NewVector

        for elem in Matrix:
            NewVector[elem[0]] += (elem[2])*(Vector[elem[1]])
        return (NewVector)


"""MDense = np.array([[3,0,0],[0,0,0],[0,0,6]])
MSparse = np.array([[0,0,3],[2,2,6]])
u = np.array([1,2,3])
MDenseApplyu = Apply([MDense,u])
MSparseApplyu = Apply([MSparse,u])
print(MDenseApplyu.DenseApply())
print(MSparseApplyu.SparseApply())
"""
