import numpy as np 
from abc import ABC, abstractmethod 
from Sparse import SparseMatrix  


class LinearOperator(ABC):
    '''
    Class to represent linear operators i.e. quantum gates. Linear operators will either take 
    the form of dense-, sparse-, or lazy matrices. Class must fullfill requirements for linearity, 
    i.e. contain a scalar multiplication and addition, as well as a multiplication. 
    Furthermore it must be clear how each Linear Operator is applied to a quantum register. 
    Being able to print out a matrix representation of a linear operator is an advantage. 

    '''

    @abstractmethod
    def scale(self, factor):
        pass

    @abstractmethod
    def multiply(self, other_operator):
        pass

    @abstractmethod
    def __str__(self):
        pass

    

class LazyMatrix(LinearOperator):   #or LazyMatrix(LinearOperator):
    
    def __init__(self, dimension, apply):
        '''
        Constructor. 

        Input:
        ------ 
        dimension [int]   - the dimension of the operator
        apply [func]      - function dictating the effect of applying LazyMatrix
        '''

        assert isinstance(dimension, int), f"Dimension must be an integer, received type {type(dimension)}."
        self.Dimension = dimension
        self.Apply = apply
        self.Cache = None

    
    def multiply(self, otherOperator):
        '''
        Function to determine effect of multiplying two LazyMatrices. 

        Input:
        ------ 
        otherOperator [LazyMatrix]   - the operator to be multiplied with. Must be of type LazyMatrix and 
                                                of same dimensionality as self. 

        Returns:
        ------ 
        productOperator [LazyMatrix] - new LazyMatrix with same dimensionality as self and updated Apply
                                                operation corresponding to the product of the two LazyMatrices. 
        '''

        assert isinstance(otherOperator, LazyMatrix) , f'''Lazy Matrix multiplication requires two Lazy Matrices
                                                        but was given a {type(otherOperator)}.'''
        assert self.Dimension == otherOperator.Dimension, "Incompatible dimensions."

        updatedApply = lambda v : self.Apply(otherOperator.Apply(v))
        productOperator = LazyMatrix(self.Dimension, updatedApply)

        return productOperator


    def scale(self, factor):
        '''
        Function to scale Apply operation of self by given factor. 

        Input: 
        ------
        factor [int, float]         - scaling factor to be applied

        Returns:
        ------ 
        scaledOperator [LazyMatrix] - new LazyMatrix with scaled Apply operation. 
        '''

        assert isinstance(factor, (int, float)) , f'''Lazy Matrix scaling requires a float or int
                                                        but was given a {type(factor)}.'''
        
        updatedApply = lambda v : factor * self.Apply(v)
        scaledOperator = LazyMatrix(self.Dimension, updatedApply)

        return scaledOperator
    

    def __SparseRepresentation(self):
        '''
        Function to convert LazyMatrix to a SparseMatrix. Conversion performed by application of 
        Apply operation to each basis element. 

        Returns:
        ------ 
        sparseRepresentation [SparseMatrix]   - a sparse representation of the more abstract LazyMatrix self
        '''
 
        sparseRepresentation = SparseMatrix(self.Dimension)
        basisElement = np.zeros(self.Dimension, dtype = complex) 

        for col in range(self.Dimension):  
            basisElement[col] += 1                      
            column = self.Apply(basisElement)
            

            for row in range(self.Dimension):
                sparseRepresentation[row, col] = column[row] 

            basisElement[col] -= 1

        self.Cache = sparseRepresentation  # will this work as intended? And not take up memory until __str__ called?   


    def __getitem__(self, index):
        if self.Cache is None : self.__SparseRepresentation()
        
        return self.Cache[index]


    def __str__(self):
        if self.Cache is None : self.__SparseRepresentation()
            
        return str(self.Cache)




#________________________________________________________________________________________________

