import numpy as np
from Apply_File import Apply
from Gate_File import Gate
from Tensor import TensorProduct

H_gate = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])


class Qubit:
    def __init__(self, state=np.array([1, 0], dtype=complex)) -> None:

        self.state = np.array(self.normalize(state), dtype=complex)

    def normalize(self, vec_to_norm):
        """
        Normalizes a complex vector to magnitude 1

        Args:
        vec_to_norm (1D array) : 2*1 vector holding the linear coefficient of "unormalized" state
        """

        assert (isinstance(vec_to_norm, np.ndarray) and len(
            vec_to_norm) == 2), "Supply complex vector (2*1 numpy array)"

        factor = np.sqrt((vec_to_norm*np.conj(vec_to_norm)).sum())
        return vec_to_norm / factor

    def apply(self, gate):
        """
        Applies a gate to a qubit -> performs inner profuct on the 
        matrix and vector representation of the state of the qubit

        Args:
        gate (2D array) : square matrix repesentig a quantum gate (lin operator)
        """

        temp = Apply([gate, self.state])
        self.state = temp.DenseApply()

    def measure(self):
        """
        A qubit is of form |state> = a|+> + b|->, so when measured,
        based on its amplitudes (a,b) it will collapse either to |+> or |->
        """

        P = np.abs(self.state) ** 2
        pos = np.random.choice([0, 1], p=P)
        self.state[pos] = 1
        self.state[pos-1] = 0

    def __str__(self) -> str:
        out = f"state = | {self.state[0]}|+> + {self.state[1]}|-> >"
        return out


class Q_Register:

    def __init__(self, n: int, states=None) -> None:
        """
        Initializes the Q_Register with n |0> state Qubits

        Args:
        n (int) : number of qubits
        """
        self.n = n
        self.state = np.zeros(2**n, dtype=complex)
        temp = []

        if np.all(states) == None:
            for i in range(n):

                temp.append(Qubit())

            self.state[0] = 1

        else:
            # semiProd = np.array([1])
            to_tens_prod = []
            for i in range(n):

                temp.append(Qubit(states[2*i: 2*(i+1)]))
                to_tens_prod.append(temp[i].state)
                # semiProd = np.kron(semiProd, temp[i].state)

            self.state = TensorProduct(to_tens_prod).denseTensorProduct()
            self.state = np.squeeze(self.state)
        self.qubits = np.array(temp)

    def apply_gate(self, gate: Gate, index):
        """
        Applies gate to index^th qubit and return the new state of the Q_Register

        Arg:
        gate (ND squaere matrix) : matrix representation of the gate
        index (int) : qubit index that the gate should be applied to

        Returns:
        ND matrix where N is the initial number of qubits
        """

        # TODO: we assume the gate is compatible with the register
        # -> qRegState is of size 2**n * 1 and gate 2**n * 2**n

        if gate.matrixType == "Sparse":
            self.state = gate.GateMatrix.SparseApply(self.state)
        elif gate.matrixType == "Dense":
            self.state = gate.GateMatrix.DenseApply(self.state)
        else:  # Lazy ?????
            pass
        pass

    def measure(self):
        """
        Measurement of the Q_Register, all the qubits collapse into |+> or |->        
        """

        for q in self.qubits:
            q.measure()

    def __str__(self) -> str:  # basically converts self.qubits to self.state
        # prints the Q_Register as 1D array
        out = f""
        for x in self.qubits:
            out += f"{x.state}"

        return out.replace("][", "] [")


a = np.array([1+1j, 2+2j], dtype=complex)
b = np.array([3+3j, 4+4j], dtype=complex)
q = Q_Register(4, np.array([1+0j, 0j, 1+0j, 0j, 1+0j, 0j, 1+0j, 0j]))
print(q)
print(q.state)
q.measure()
print(q)
