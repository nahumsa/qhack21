# Can Meta-VQE beat Barren Plateaus?

One of the main problems on Quantum Neural Networks (QNN) is the problem of Barren Plateaus, that is as the system grows in size (more qubits) the gradient of the loss function becomes exponentially smaller, this leads to untrainable circuits. Barren Plateaus have various origins, for instance the ansatz expressiveness [4] or even the presence of noise [5].

It has been shown in [3] that a clever initialization of parameters can avoid barren plateaus, thus in this project I plan to analyze if the Meta-VQE [1] initialization can surpass the problem of barren plateaus. This project has two parts:

    - The first step of the project was to implement the Meta-VQE for pennylane and apply for solving the XXZ Hamiltonian;
    - The second part is to compare random initalization and Meta-VQE initialization.



## Results
All results are on this [notebook](https://nbviewer.jupyter.org/github/nahumsa/qhack21/blob/master/Meta_VQE_Pennylane.ipynb).

### Meta-VQE
I sucessfuly implemented the Meta-VQE algorithm [1] according to the [paper](https://arxiv.org/abs/2009.13545), and got similar results as the paper for the XXZ hamiltonian.

![XXZ](https://github.com/nahumsa/qhack21/blob/master/images/XXZ.png)

### Barren Plateaus

Due to lack of time, I analyzed the performance of VQE using random initialization (orange) and Meta-VQE initialization (blue) and we see that it appears that the Meta-VQE initialization is able to avoid barren plateaus for this gradient as becomes constant according to the number of qubits while random initialization keeps decreasing.
![grad](https://github.com/nahumsa/qhack21/blob/master/images/grad.png)


# References

[1] [Cervera-Lierta, Alba, Jakob S. Kottmann, and Alán Aspuru-Guzik. "The meta-variational quantum eigensolver (meta-vqe): Learning energy profiles of parameterized hamiltonians for quantum simulation." arXiv preprint arXiv:2009.13545 (2020)](https://arxiv.org/abs/2009.13545).

[2] [Barren Plateaus Pennylane Demo](https://pennylane.ai/qml/demos/tutorial_barren_plateaus.html)

[3] [Grant, Edward, et al. An initialization strategy for addressing barren plateaus in parametrized quantum circuits. arXiv preprint arXiv:1903.05076 (2019)](https://arxiv.org/abs/1903.05076)

[4] [Wang, Samson, et al. "Noise-induced barren plateaus in variational quantum algorithms." arXiv preprint arXiv:2007.14384 (2020).](https://arxiv.org/abs/2007.14384)

[5] [Holmes, Zoë, et al. "Connecting ansatz expressibility to gradient magnitudes and barren plateaus." arXiv preprint arXiv:2101.02138 (2021).](https://arxiv.org/abs/2101.02138)