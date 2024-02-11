#import State
#import Gate
import numpy as np

class Tensor(object):
    
    def __init__(self, thingsToBeTensored, tensorType = None):
        
        assert isinstance(thingsToBeTensored, list), "The primary input for the tensor product method should be passed as a list."
        
        #Check that we're inputting only vectors or only matrices
        #if(all(isinstance(vector, State.vector) for vector in self.thingsToBeTensored)):
        #   self.productType = "Vectors"
        #elif(all(isinstance(operator, Gate.operator) for operator in self.thingsToBeTensored)):
        #   self.productType = "Operators"
        #else:
        #   raise Exception("The inputs for a tensor products should ALL be either vectors (states) or matrices (operators).")
        
        #I think I'll need an extra method for sparse matrices and maybe lazy ones too, hence the tensorType
        self.tensorType = tensorType
        self.thingsToBeTensored = thingsToBeTensored
        
    def tensorProduct(self):
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
        Product = self.thingsToBeTensored[0]
        
        for productNumber in range(1, len(self.thingsToBeTensored)):
            
            xLengthA = Product.shape[0]
            yLengthA = Product.shape[1]
            
            xLengthB = self.thingsToBeTensored[productNumber].shape[0]
            yLengthB = self.thingsToBeTensored[productNumber].shape[1]
            
            #Following the notation of the docstring, Product is A, and self.thingsToBeTensored[productNumber] is B
            BMatrix = self.thingsToBeTensored[productNumber]
            
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
        #if self.productType == "Vectors":
        #   return State(Product)
        #else:
        #   return Gate(Product)
        
        
        
        #np.kron seems good for calculating tensor products, but I think prof probaby wouldn't like us using it        
        #np.kron(A,B) gives the kronecker product which is what the naive implementation does.
        # NB: since we use matrices, the kronecker product should be identical to the tensor product, with the caveat listed above for vectors.
        
        return Product
    
A = np.array([[1,-2],[3,6]])
B = np.array([[3,5],[7,4]])

ATensorB = Tensor([A,B])

print(ATensorB.tensorProduct())
print(np.kron(A,B))