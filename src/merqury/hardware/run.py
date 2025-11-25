import numpy as np


from merqury.utils.hardware import measure_ground_state_energy_subspace_sampling

from merqury.algorithms.sweep import get_sweep_circuit
from merqury.hamiltonians.pauliops import TFIM, get_min_gap, get_ground_energy

from qiskit.quantum_info import SparsePauliOp


def compute_ground_energy_sweep(
    ham_start: SparsePauliOp,
    hamiltonian: SparsePauliOp,
    total_time: float,
    n_steps: int,
    n_shots: int,
    device_name: str = "garnet",
) -> dict:

    ground_energy = get_ground_energy(hamiltonian)
    sweep_circuit = get_sweep_circuit(
        ham_start, hamiltonian, total_time=total_time, n_steps=n_steps
    )
    energy_dict = measure_ground_state_energy_subspace_sampling(
        hamiltonian,
        ground_energy,
        sweep_circuit,
        run_on_hardware=True,
        device_name=device_name,
        n_shots=n_shots,
    )

    return energy_dict


if __name__ == "__main__":
    ham0, ham1 = TFIM(10, 1, 1, split_parts=True)

    mult = 1

    total_time = mult / (get_min_gap(ham1) ** 2)

    n_steps = mult * ham0.num_qubits**2

    n_shots = 1000

    energy_dict = compute_ground_energy_sweep(ham0, ham1, total_time, n_steps, n_shots)
