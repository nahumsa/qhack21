# Can Meta-VQE beat Barren Plateaus?

One of the main problems on Quantum Neural Networks (QNN) is the problem of Barren Plateaus, that is as the system grows in size (more qubits) the gradient of the loss function becomes exponentially smaller, this leads to untrainable circuits. Barren Plateaus have various origins, for instance the ansatz expressiveness or even the presence of noise.
In this project I plan to analyze if the Meta-VQE initialization can surpass the problem of barren plateaus. 
This project has two parts:

    - The first step of the project was to implement the Meta-VQE for pennylane and apply for solving the XXZ Hamiltonian;
    - The second part is to compare random initalization and Meta-VQE initialization.



## Results

Due to lack of time, I analyzed the performance of VQE using random initialization (orange) and Meta-VQE initialization (blue) and we see that it appears that the Meta-VQE initialization is able to avoid barren plateaus for this gradient as becomes constant according to the number of qubits while random initialization keeps decreasing.
![grad](https://github.com/nahumsa/qhack21/blob/master/images/grad.png)