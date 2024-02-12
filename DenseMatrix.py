import numpy as np

class DenseMatrix():

    def __init__(self, inputArray, shape = None):
        """
        Input
        ------
        A numpy array 
        """
        if not isinstance(inputArray, np.ndarray):
            print(f"Warning, had to convert DenseMatrix primary input from a {type(inputArray)} into a numpy array.") 
            inputArray = np.array(inputArray)

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

        assert isinstance(factor, (int, float)), "DenseMatrix scale method expects an int or flaot input scalar."

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

        assert isinstance(matrix2, DenseMatrix), "DenseMatrix multiply method expects DenseMatrix input."

        product = np.asmatrix(self.inputArray) * np.asmatrix(matrix2.inputArray)
        array = np.asarray(product) 
        return DenseMatrix(array)

    def __str__(self):
        return str(np.asmatrix(self.inputArray))


if __name__ == "__main__":
    a = np.array([[3,0,1],[1,0,0],[0,0,6]])
    b = 2
    c = [[3,0,1],[1,0,0],[0,0,6]] 

    matrixA = DenseMatrix(a) 
    matrixA.scale(b)
    print(matrixA)

    matrixC = DenseMatrix(c)

    product = matrixA.multiply(matrixC)
    print(product)