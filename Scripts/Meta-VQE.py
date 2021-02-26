import sys
import pennylane as qml
import numpy as np
from cost_functions import m_vqe_cost

from tqdm import tqdm

# Setup the device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Creating training data
train_deltas = np.random.uniform(low=-1, high=1, size=5)

# Hyperparameters
eta = 0.75
n_layers = 4 # One encoding and one processing
L = 2*n_layers

# initializing parameters
params = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(L, n_qubits, 4))

# Training Parameters
epochs = 10
optimizer = qml.AdagradOptimizer()

from functools import partial

# Applyies train_deltas for the Meta-VQE cost function
cost_fn = partial(m_vqe_cost, train_deltas, eta, dev)

pbar = tqdm(range(epochs), desc='Energy', leave=True)

for i in pbar:
    params, val = optimizer.step_and_cost(cost_fn, params)
    pbar.set_description(f"Loss: {val:.3f}")

params_mvqe = params.copy()

np.save('params_vqe', params_mvqe)