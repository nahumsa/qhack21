import numpy as np
import pennylane as qml

def variational_ansatz(params: np.array, delta: float , wires: qml.wires, H=None):
    """ Variational ansatz with linear encoding.


    """
    n_layers = params.shape[0]
    n_qubits = params.shape[1]
    for L in range(n_layers):
        # Encoding Layer
        if L % 2 == 0:
            for qubit in range(n_qubits):
                qml.RZ(params[L][qubit][0] * delta + params[L][qubit][1], wires=qubit)
                qml.RY(params[L][qubit][2] * delta + params[L][qubit][3], wires=qubit)
            
            for ent in range(0, n_qubits - 1, 2):
                qml.CNOT(wires= [ent, ent+1])
        
        # Processing Layer
        else:
            for qubit in range(n_qubits):
                qml.RZ(params[L][qubit][0] , wires=qubit)
                qml.RY(params[L][qubit][2] , wires=qubit)
            
            for ent in range(0, n_qubits - 1, 2):
                qml.CNOT(wires= [ent, ent+1])