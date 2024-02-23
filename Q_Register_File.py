import numpy as np
from Apply_File import Apply
from Gate_File import Gate, SwapMatrix1a
from Tensor import TensorProduct
from Dense import DenseMatrix
from Sparse import SparseMatrix
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

        # TODO: is it a problem that the quibits are normalized individually when they are initialized?

        if np.all(states) == None:
            for i in range(n):

                temp.append(Qubit())

            self.state[0] = 1
            self.state = DenseMatrix(self.state)

        else:
            to_tens_prod = []
            for i in range(n):

                temp.append(Qubit(states[2*i: 2*(i+1)]))
                to_tens_prod.append(DenseMatrix(temp[i].state))

            self.state = np.squeeze(TensorProduct(
                to_tens_prod).denseTensorProduct().inputArray)
        self.qubits = np.array(temp)

    def apply_gate(self, gate: Gate, index):
        """
        Applies gate to qubit/qubits in index and return the new state of the Q_Register

        Arg:
        gate : matrix representation of the gate
        index (list) : qubit index that the gate should be applied to (NOTE If gate is cNot/cV, first index in list is control, second is target)

        Returns:
        The register with the modified state
        """

        # TODO: we assume the gate is compatible with the register
        # -> qRegState is of size 2**n * 1 and gate 2**n * 2**n

        QubitNum = self.n
        State = self.state
        TensorList = []

        if gate.gateName != "cNot" and gate.gateName != "cV":
            if gate.matrixType == "Sparse":
                Identity = SparseMatrix(2, [[0, 0, 1], [1, 1, 1]])
                for i in range(QubitNum):
                    TensorList.append(Identity)
                for num in index:
                    TensorList[num] = gate.GateMatrix
                TensorGate = TensorProduct(TensorList).sparseTensorProduct()

                NewState = TensorGate.SparseApply(State)
                self.state = NewState
                return NewState

            elif gate.matrixType == "Dense":
                Identity = DenseMatrix(np.array([[1, 0], [0, 1]]))
                for i in range(QubitNum):
                    TensorList.append(Identity)
                for num in index:
                    TensorList[num] = gate.GateMatrix
                TensorGate = TensorProduct(TensorList).denseTensorProduct()

                NewState = TensorGate.DenseApply(State)
                self.state = NewState
                return NewState

            else:  # Lazy ?????
                pass
        else:
            Control = index[0]
            Target = index[1]
            Identity = SparseMatrix(2, [[0,0,1],[1,1,1]])
            if Control == 0:
                SwapMatrixControl = DenseMatrix(np.eye(2**QubitNum)).Sparse()
            else:
                SwapMatrixControl = SwapMatrix1a(QubitNum, Control)
            if Target == 1:
                SwapMatrixTarget = DenseMatrix(np.eye(2**QubitNum)).Sparse()
            else:
                SwapMatrixTarget = TensorProduct([Identity,SwapMatrix1a(QubitNum-1, Target-1)]).sparseTensorProduct()
            SwapMatrixForward = SwapMatrixTarget.Multiply(SwapMatrixControl)
            SwapMatrixBackward = SwapMatrixControl.Multiply(SwapMatrixTarget)

            if gate.matrixType == "Sparse":
                for i in range(QubitNum-1):
                    TensorList.append(Identity)
                TensorList[0] = gate.GateMatrix
                TensorGate = TensorProduct(TensorList).sparseTensorProduct()

                NewState1 = SwapMatrixForward.SparseApply(State)
                NewState2 = TensorGate.SparseApply(NewState1)
                NewState = SwapMatrixBackward.SparseApply(NewState2)
                self.state = NewState
                return NewState
            
            elif gate.matrixType == "Dense":
                DenseSwapForward = DenseMatrix(SwapMatrixForward.Dense())
                DenseSwapBackward = DenseMatrix(SwapMatrixBackward.Dense())
                Identity = DenseMatrix(np.array([[1, 0], [0, 1]]))
                for i in range(QubitNum-1):
                    TensorList.append(Identity)
                TensorList[0] = gate.GateMatrix
                TensorGate = TensorProduct(TensorList).denseTensorProduct()

                NewState1 = DenseSwapForward.DenseApply(State)
                NewState2 = TensorGate.DenseApply(NewState1)
                NewState = DenseSwapBackward.DenseApply(NewState2)
                self.state = NewState
                return NewState
            
            else:  # Lazy ?????
                pass





    def measure(self):
        """
        Measurement of the possibly entagled state of Q_Register,
        according to the amplitudes, leaving the register in a state
        that is binary representation of a number between 0 and (2**n)-1       
        """

        P = np.array([abs(qb)**2 for qb in self.state.inputArray])
        result = np.random.choice(np.arange(len(self.state.inputArray)), p=P)
        collapsed = format(result, "0"+str(len(self.state.inputArray))+"b")
        for i in range(len(collapsed)):
            self.state.inputArray[i] = collapsed[i]

    def __str__(self) -> str:
        # prints the Q_Register as 1D array
        out = f""
        for x in self.qubits:
            out += f"{x.state}"

        return out.replace("][", "] [")


a = np.array([1+1j, 2+2j], dtype=complex)
b = np.array([3+3j, 4+4j], dtype=complex)
q = Q_Register(3, np.array([1+0j, 0j, 0j, 1+0j, 1+0j, 0j]))


print(q.state)


HGate = Gate("Dense", "cNot")
q.apply_gate(HGate, [2,1])

print(q.state)

