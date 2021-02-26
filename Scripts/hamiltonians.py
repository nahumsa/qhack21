import numpy as np
import pennylane as qml

def hamiltonian_XXZ(n_qubits: int, delta: float, eta: float) -> qml.Hamiltonian:
    """ Creates the XXZ hamiltonian, which is given by:

    $$
    \mathcal{H} = \sum_{i=1}^N \big( X_i X_{i+1} + Y_i Y_{i+1} 
    + \Delta Z_i Z_{i+1} \big) + \eta \sum_{i=1}^N Z_i
    $$

    Args:
        n_qubits(int): number of spins in the chain.
        delta(float): delta parameter.
        eta(float): eta parameter.
    """
    hamiltonian = []
    coeffs = []
    
    # Periodic Boundary Conditions
    for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
        hamiltonian.append(op(n_qubits-1)@op(0))
        if op != qml.PauliZ :
            coeffs.append(1.)
        else:
            coeffs.append(delta)
    
    hamiltonian.append(qml.PauliZ(n_qubits-1))
    coeffs.append(eta)

    for qubits in range(n_qubits - 1):
        for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
            
            hamiltonian.append(op(qubits)@op(qubits+1))
            
            if op != qml.PauliZ :
                coeffs.append(1.)
            else:
                coeffs.append(delta)
        
        hamiltonian.append(qml.PauliZ(qubits))
        coeffs.append(eta)

    H = qml.Hamiltonian(coeffs, hamiltonian, simplify=True)
    return H

def hamiltonian_to_matrix(H: qml.Hamiltonian) -> np.array:
    """ Converts a pennylane Hamiltonian object into a matrix.

    Args:
        H(qml.Hamiltonian): Hamiltonian.

    Output:
        np.array: Outputs the matrix representation of the Hamiltonian.
    """
    n_qubits = len(H.wires)
    mat = np.zeros((2**n_qubits, 2**n_qubits), np.complex128)
    for coef, op in zip(*H.terms):
        mat += coef*qml.utils.expand(op.matrix, op.wires, n_qubits)
    return mat

def exact_gs(H: qml.Hamiltonian) -> float:
    """ Calculates the Ground State energy of the Hamiltonian.

    Args:
        H(qml.Hamiltonian): Hamiltonian.

    Output:
        float: outputs the ground state energy of the Hamiltonian.
    """
    matrix = hamiltonian_to_matrix(H)
    energies = np.linalg.eigvals(matrix)
    return np.real(min(energies))