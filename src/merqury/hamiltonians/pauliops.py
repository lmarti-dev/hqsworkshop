from qiskit.quantum_info import SparsePauliOp
import numpy as np
from scipy.sparse.linalg import eigsh
from pathlib import Path


def load_ham(fpath: Path) -> SparsePauliOp:
    with open(Path(fpath), "rb") as fd:
        hamiltonian = SparsePauliOp.from_list(np.load(fd))
    return hamiltonian


def get_ground_energy(hamiltonian: SparsePauliOp) -> float:
    mat = hamiltonian.to_matrix(sparse=True)
    ground_energy, _ = eigsh(mat, k=1, which="SA")
    return ground_energy[0]


def get_min_gap(hamiltonian: SparsePauliOp) -> float:
    mat = hamiltonian.to_matrix(sparse=True)
    min_gap = 0
    k = 2
    while np.isclose(min_gap, 0):
        energies, _ = eigsh(mat, k=k, which="SA")
        min_gap = np.abs(energies[-1] - energies[0])
        k += 1
    return min_gap


def TFIM(
    n_qubits: int, h: float, J: float, periodic: bool = False, split_parts: bool = False
) -> SparsePauliOp:

    hamiltonian_J = sum(
        [
            -J * SparsePauliOp(s * "I" + "XX" + (n_qubits - 2 - s) * "I")
            for s in range(n_qubits - 1)
        ]
    )
    if periodic:
        hamiltonian_J += SparsePauliOp("X" + "I" * (n_qubits - 2) + "X")
    hamiltonian_h = sum(
        [
            -h * SparsePauliOp(s * "I" + "Z" + (n_qubits - 1 - s) * "I")
            for s in range(n_qubits)
        ]
    )

    if split_parts:
        # J: XX h: Z
        return hamiltonian_J, hamiltonian_h

    hamiltonian = hamiltonian_J + hamiltonian_h

    return hamiltonian
