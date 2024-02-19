from Sparse import Sparse
from Dense import Dense
from Q_Register_file import Q_Register
from Gate import Gate
import numpy as np

class TensorProduct(object):
    
    def __init__(self, thingsToBeTensored):
        
        assert isinstance(thingsToBeTensored, list), "The primary input for the tensor product method should be passed as a list."
        
        #Check that we're inputting only vectors or only matrices
        if(all(isinstance(vector, Q_Register) for vector in self.thingsToBeTensored)):
           self.productType = "Vectors"
        elif(all(isinstance(operator, Gate) for operator in self.thingsToBeTensored)):
           self.productType = "Operators"
        else:
           raise Exception("The inputs for a tensor products should ALL be either vectors (states) or matrices (operators).")
        
        if self.productType == "Operators" and thingsToBeTensored[0].matrixType == SparseMatrix:
            self.tensorProduct = self.sparseTensorProduct
        else:
            self.tensorProduct = self.denseTensorProduct
        
        self.thingsToBeTensored = thingsToBeTensored
        
    def denseTensorProduct(self):
        """
        Initial naive implementation
        Based on the definition that for 2 matrices, A (n x m) and B (p x q), the tensor product is:
        
        [[A_11 B, A_12 B, ... , A_1m B],
         [A_21 B, A_22 B, ... , A_2m B],
          ...
         [A_n1 B, A_n2 B, ... , A_nm B]]
        
        Thus the output is a np x mq matrix.
        
        The ij'th component of the product is therefore:
        A_kl B_xy
        With k/l = floor(i/j / p/q) and x/y = i/j mod p/q
        
        Note for vectors, which have dimension n x 1, m x 1, the tensor product has dimension nm x 1.
        Strictly, for 2 vectors, one has the tensor product of dimension n x m.
        The elements relate as the n x m, ij'th entry, is the i*n +j'th element of the nm x 1 vector.
        That is, this is just a representation of the same thing.
        I based this code on the kronecker product, which is essentially a tensor product specialised to matrices.

        Returns
        -------
        Product : The tensor product of the list thingsToBeTensored. 
        
        Output is of type operator or vector depending what is being tensored.

        """
        
        #Initialise the product 
        Product = self.thingsToBeTensored[0].inputArray
        
        for productNumber in range(1, len(self.thingsToBeTensored)):
            
            xLengthA = Product.shape[0]
            yLengthA = Product.shape[1]
            
            xLengthB = self.thingsToBeTensored[productNumber].shape[0]
            yLengthB = self.thingsToBeTensored[productNumber].shape[1]
            
            #Following the notation of the docstring, Product is A, and self.thingsToBeTensored[productNumber] is B
            BMatrix = self.thingsToBeTensored[productNumber].inputArray
            
            #Find the shape of the product
            newShape = (xLengthA * xLengthB, yLengthA * yLengthB)
            
            newProduct = np.zeros(newShape)
            
            #Loop over entries of the tensor product
            for i in range(newShape[0]):
                for j in range(newShape[1]):
                    
                    n = int(i / xLengthB)
                    m = int(j / yLengthB)
                    
                    p = i % xLengthB
                    q = j % yLengthB
                    
                    newProduct[i][j] = Product[n][m] * BMatrix[p][q]
                    
            
            Product = newProduct
            
        #Give the Product the correct type and return it
        if self.productType == "Vectors":
            
            nQubits = np.log2(Product.shape[0])
            
            return Q_Register(nQubits, Product)
        else:
            
            quBits = []
            
            for Gate in self.thingsToBeTensored:
                
                quBits.append(Gate.quBits)
            
            return Gate("Dense", quBits, customInput = Product)
        
        
        
        #np.kron seems good for calculating tensor products, but I think prof probaby wouldn't like us using it        
        #np.kron(A,B) gives the kronecker product which is what the naive implementation does.
        # NB: since we use matrices, the kronecker product should be identical to the tensor product, with the caveat listed above for vectors.
        
        return Product
    
    def sparseTensorProduct(self):
        """
        Implemenetation using sparse matrices.
        Still based on the kronecker product.
        
        Assumes sparse matrices are input as lists of tuples [(i,j,M_ij)]

        Returns
        -------
        Product : The tensor product of the list thingsToBeTensored. 
        
        Output is of type operator or vector depending what is being tensored.
        

        """
        
        #Initialise the product 
        Product = self.thingsToBeTensored[0]
        
        for productNumber in range(1, len(self.thingsToBeTensored)):
            
            BMatrix = self.thingsToBeTensored[productNumber]
            
            newShape = (Product.Dimension[0] * BMatrix.Dimension[0], Product.Dimension[1] * BMatrix.Dimension[1])
            
            newProduct = []
            
            for elementA in Product.elements:                
                for elementB in BMatrix.elements:
                    
                    i = BMatrix.dimension[0] * elementA[0] + elementB[0]
                    j = BMatrix.dimension[1] * elementA[1] + elementB[1]
                    
                    val = elementA * elementB
                    
                    newProduct.append((i,j,val))
            
            Product = newProduct
          
        quBits = []
            
        for Gate in self.thingsToBeTensored:
                
            quBits.append(Gate.quBits)
          
        return Gate("Sparse", quBits, customInput = Product) 
    
A = np.array([[1,1],[1,-1]])
B = np.array([[1,1],[1,-1]]) 
C = np.eye(2)

ATensorB = Tensor([A,C,B])

print(ATensorB.denseTensorProduct())
print(np.kron(np.kron(A,C),B))
