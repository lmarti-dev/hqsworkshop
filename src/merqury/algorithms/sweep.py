from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter, LieTrotter
import numpy as np
from scipy.linalg import expm


def get_sweep_function(kind: str):
    if kind == "sin":

        def sweep_function(t):
            return np.sin(np.pi / 2 * np.sin(np.pi * t / 2) ** 2) ** 2

    elif kind == "simple":

        def sweep_function(t):
            return t

    else:
        raise NotImplementedError(f"Unknown kind: {kind}")

    return sweep_function


def get_sweep_circuit(
    ham_0: SparsePauliOp,
    ham_1: SparsePauliOp,
    total_time: float,
    n_steps: int,
    sweep_function: str = "simple",
    synthesis: str = "suzuki",
) -> QuantumCircuit:

    sf = get_sweep_function(sweep_function)
    if synthesis == "lie":
        synth_fn = LieTrotter()
    elif synthesis == "suzuki":
        synth_fn = SuzukiTrotter()
    elif synthesis is None:
        synth_fn = lambda x: x
    circuit = QuantumCircuit((ham_0 + ham_1).num_qubits)
    for n in range(n_steps):
        alpha = sf(n / (n_steps - 1))
        circuit = circuit.compose(
            synth_fn.synthesize(
                PauliEvolutionGate(
                    ham_0 * (1 - alpha) + ham_1 * alpha,
                    total_time / n_steps,
                )
            ),
            range((ham_0 + ham_1).num_qubits),
        )
    return circuit
