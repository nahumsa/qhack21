import pennylane as qml
import numpy as np

from hamiltonians import hamiltonian_XXZ
from Ansatz import variational_ansatz

def ExpvalH(H: qml.Hamiltonian, device: qml.device):
    coeffs, observables = H.terms
    qnodes = qml.map(
            variational_ansatz, observables, device
            )
    cost = qml.dot(coeffs, qnodes)
    return cost

def m_vqe_cost(train_deltas: np.array, eta: float, dev: qml.device , params: np.array) -> float:
    # cost function value
    c = 0.
    n_qubits = dev.num_wires

    for delta in train_deltas:
        H = hamiltonian_XXZ(n_qubits, delta, eta)
        cost = ExpvalH(H, dev)
        c += cost(params, delta=delta)
    
    return c