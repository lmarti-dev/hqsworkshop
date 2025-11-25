import numpy as np

import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from merqury.algorithms.sweep import get_sweep_function
from merqury.hamiltonians.pauliops import (
    load_ham,
    get_diagonal_ham,
)


def get_sweep_spectrum(
    ham_0: SparsePauliOp,
    ham_1: SparsePauliOp,
    n_steps: int,
    sweep_function: str = "simple",
):
    spectrum = np.zeros((2**ham_0.num_qubits, n_steps))
    sf = get_sweep_function(sweep_function)
    for n in range(n_steps):
        alpha = sf(n / (n_steps - 1))
        sweep_ham = ham_0 * (1 - alpha) + ham_1 * alpha
        v = np.linalg.eigvalsh(sweep_ham.to_matrix())
        spectrum[:, n] = v

    return spectrum


filename = "pauli_hamiltonian_6qubits.npy"


ham1 = load_ham(filename)

ham0 = get_diagonal_ham(ham1)

n_steps = 100

spectrum = get_sweep_spectrum(ham0, ham1, n_steps)


fig, ax = plt.subplots()

n_levels = 2**ham1.num_qubits

for n in range(n_levels):
    ax.plot(np.linspace(0, 1, n_steps), spectrum[n, :])


plt.show()
