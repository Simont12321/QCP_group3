import numpy as np

H_gate = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])


class Q_Register:

    def __init__(self, n: int) -> None:
        """
        Initializes the Q_Register with n |0> state Qubits

        Args:
        n (int) : number of qubits
        """
        self.n = n
        # default |0> state
        qubit = np.array([1, 0])
        # initialize n qubits
        self.states = np.tile(qubit, 2**(n-1)).reshape((2,)*n)

    def apply_gate(self, gate, index):
        """
        Applies gate to index^th qubit and return the new state of the Q_Register

        Arg:
        gate (ND squaere matrix) : matrix representation of the gate
        index (int) : qubit index that the gate should be applied to

        Returns:
        ND matrix where N is the initial number of qubits
        """
        pass

    def measure(self):
        """
        Measurement of the Q_Register, all the qubits collapse into classical bit (0 or 1)

        Returns:
        result (binary reresentation of int) : the current state of register
        """
        probabilities = [abs(q)**2 for q in self.states.flatten()]
        result = np.random.choice(range(self.n), 1, p=probabilities)
        return bin(result)

    def __str__(self) -> str:
        # prints the Q_Register as 1D array
        print(f"{self.states.flatten()}")
