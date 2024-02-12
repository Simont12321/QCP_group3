import numpy as np

class DenseMatrix():

    def __init__(self, inputArray, shape = None):
        """
        Input
        ------
        A numpy array
        """

        self.inputArray = inputArray 
        self.shape = np.shape(inputArray) 

    def scale(self, factor):
        """
        Input
        ------
        Scalar as an integer or floating point number. 

        Returns
        -------
        Nothing, modifies the original matrix. 
        """

        self.inputArray = np.asmatrix(self.inputArray) * factor

    def multiply(self, matrix2):
        """
        Input
        ------
        Another DenseMatrix 

        Returns
        -------
        The product of the two input matrices as a DenseMatrix. 
        """

        product = np.asmatrix(self.inputArray) * np.asmatrix(matrix2)
        array = np.asarray(product) 
        return DenseMatrix(array)

    def __str__(self):
        return str(np.asmatrix(self.inputArray))


if __name__ == "__main__":
    a = np.array([[3,0,1],[1,0,0],[0,0,6]])
    b = 2

    matrixA = DenseMatrix(a) 
    print(matrixA) 

    print(matrixA.scale(b))