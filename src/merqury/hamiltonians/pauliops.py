from qiskit.quantum_info import SparsePauliOp
import numpy as np
from scipy.sparse.linalg import eigsh

from pathlib import Path
from merqury.utils.hardware import to_bitstring, index_bits
from qiskit import QuantumCircuit


def get_nth_bitstring_v2(
    hamiltonian: SparsePauliOp, nth: int = 0, left_to_right: bool = True
) -> str:
    mat = hamiltonian.to_matrix(True).diagonal()
    ind = int(np.argsort(mat)[nth])
    bs = to_bitstring(ind, hamiltonian.num_qubits, left_to_right)
    return bs


def get_nth_bitstring(hamiltonian: SparsePauliOp, nth: int = 0) -> str:
    mat = hamiltonian.to_matrix(True)
    v, eigvecs = eigsh(mat, k=4, which="SA")
    ind = int(np.argsort(-np.abs(eigvecs[:, nth]))[0])
    bs = to_bitstring(ind, hamiltonian.num_qubits, True)
    return bs


def prep_circ_bitstring(bs: str) -> QuantumCircuit:
    inds = index_bits(bs, len(bs), right_to_left=True)
    circ = QuantumCircuit(len(bs))
    for ind in inds:
        circ.x(ind)
    return circ


def kill_diagonal(hamiltonian: SparsePauliOp) -> SparsePauliOp:
    out = []
    for term in hamiltonian:
        s = str(term.paulis[0])
        if ["I"] == list(set(s)):
            pass
        else:
            out.append(term)
    return sum(out)


def get_diagonal_ham(hamiltonian: SparsePauliOp) -> SparsePauliOp:
    out = []
    for term in hamiltonian:
        s = str(term.paulis[0])
        if "X" in s or "Y" in s:
            pass
        else:
            out.append(term)
    return sum(out)


def load_ham(filename: Path) -> SparsePauliOp:
    dirname = Path(Path(__file__).parent, "../files/")
    with open(Path(dirname, filename), "rb") as fd:
        hamiltonian = SparsePauliOp.from_list(np.load(fd))
    return hamiltonian


def get_nth_energy(hamiltonian: SparsePauliOp, nth: int = 0) -> float:
    mat = hamiltonian.to_matrix(sparse=True)
    vals, _ = eigsh(mat, k=4, which="SA")
    return vals[nth]


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


def singlet_triplet_splitting(mat: np.ndarray) -> float:
    eigenvalues, _ = eigsh(mat, k=2, which="SA")
    return (eigenvalues[1] - eigenvalues[0]) * 27.2114


def diagonalize_hamiltonian_splitting(hamiltonian: SparsePauliOp):
    mat = hamiltonian.to_matrix(True)

    return singlet_triplet_splitting(mat)
