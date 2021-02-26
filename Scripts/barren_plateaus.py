import pennylane as qml
from functools import partial
import numpy as np
from tqdm import tqdm

from cost_functions import m_vqe_cost, ExpvalH
from Ansatz import variational_ansatz
from hamiltonians import hamiltonian_XXZ

# Creating training data
train_deltas = np.random.uniform(low=-1.1, high=1.1, size=5)

# Hyperparameters
eta = 0.75
n_layers = 2 # One encoding and one processing
L = 2*n_layers

# Training Parameters
epochs = 50
optimizer = qml.AdagradOptimizer()

samples = 100

qubits = [2, 3, 4, 5]

var = []
mean = []

for n_qubits in qubits:
    dev = qml.device("default.qubit", wires=n_qubits)
    
    v_gradient = []
    
    for _ in tqdm(range(samples)):
        # Applyies train_deltas for the Meta-VQE cost function
        cost_fn = partial(m_vqe_cost, train_deltas, eta, dev)
        params = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(L, n_qubits, 4))
        optimizer = qml.AdagradOptimizer()

        # Optimization step
        for i in range(epochs):
            params, val = optimizer.step_and_cost(cost_fn, params)
        # Copy optimal parameters
        params_mvqe = params.copy()

        # Compute the gradient
        qcircuit = qml.QNode(variational_ansatz, dev)
        n_qubits = dev.num_wires

        H = hamiltonian_XXZ(n_qubits, delta=0., eta=eta)

        cost_fn = ExpvalH(H, dev)
        grad = qml.grad(cost_fn, argnum=0)
        gradient = grad(params_mvqe, delta=0.)
        v_gradient.append(gradient[0][0][-1])
    
    var.append(np.var(v_gradient))
    mean.append(np.mean(v_gradient))
    
import os

try:
    os.mkdir('BP')

except:
    pass

np.savetxt('BP/var.txt', var)
np.savetxt('BP/mean.txt', mean)