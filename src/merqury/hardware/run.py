import numpy as np

from qupiter.utils.hardware import (
    backend_sample,
    subspace_diagonalize,
    get_unique_bitstrings,
    sv_estimate_observables,
    get_appropriate_fake_backend,
    get_tfim_ratio,
    get_counts_from_job,
    get_backend,
    add_meas_layer,
    encode_sampler_result,
)

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

def measure_ground_state_energy_subspace_sampling(
    hamiltonian: SparsePauliOp,
    ground_energy: float,
    circuit: QuantumCircuit,
    n_shots: int = 1e3,
    run_on_hardware: bool = False,
    backend_name: str = "AQT",
) -> dict:
    fake_backend = get_appropriate_fake_backend(circuit.num_qubits, backend_name)
    ratio = get_tfim_ratio(hamiltonian)

    print(f"n_qubits: {hamiltonian.num_qubits}, j/h {ratio}")

    circuit_to_sample = add_meas_layer(circuit)

    def relerr(e):
        return np.abs(e - ground_energy) / np.abs(ground_energy)

    fake_job = backend_sample(
        backend=fake_backend,
        circuit=circuit_to_sample,
        parameters=[],
        n_shots=n_shots,
    )
    fake_counts = get_counts_from_job(fake_job)
    fake_unique_bitstring = get_unique_bitstrings(fake_counts)
    fake_backend_bitstring_prop = len(fake_unique_bitstring) / (2**circuit.num_qubits)
    fake_backend_energy = subspace_diagonalize(
        unique_bitstrings=fake_unique_bitstring, ham=hamiltonian
    )

    statevector_energy = sv_estimate_observables(
        circuit=circuit, observables=[hamiltonian], param_sweep=[]
    ).data.evs[0]

    if run_on_hardware:
        real_backend = get_backend(circuit.num_qubits, backend_name)
        real_job = backend_sample(
            backend=real_backend,
            circuit=circuit_to_sample,
            parameters=[],
            n_shots=n_shots,
        )
        real_counts = get_counts_from_job(real_job)
        real_unique_bitstring = get_unique_bitstrings(real_counts)
        real_bitstring_prop = len(real_unique_bitstring) / (2**circuit.num_qubits)
        real_backend_energy = subspace_diagonalize(
            unique_bitstrings=real_unique_bitstring, ham=hamiltonian
        )

        print(
            f"RealBackend ({real_backend.name}) {relerr(real_backend_energy):.5e} sub. mat.: {len(real_unique_bitstring)} × {len(real_unique_bitstring)}: {real_bitstring_prop*100:.3f}%"
        )

    print(f"Statevector {relerr(statevector_energy):.5e}")

    print(
        f"FakeBackend ({fake_backend.name}) {relerr(fake_backend_energy):.5e} sub. mat.: {len(fake_unique_bitstring)} × {len(fake_unique_bitstring)}: {fake_backend_bitstring_prop*100:.3f}%"
    )

    var_dict = {
        "fake_backend": {
            "name": fake_backend.name,
            "error": relerr(fake_backend_energy),
            "energy": fake_backend_energy,
            "unique_bitstrings": fake_unique_bitstring,
            "bitstring_prop": fake_backend_bitstring_prop,
            "n_shots": n_shots,
            "job": encode_sampler_result(fake_job),
        },
        "statevector": {
            "error": relerr(statevector_energy),
            "energy": statevector_energy,
        },
        "system": {
            "ground_energy": ground_energy,
            "n_qubits": circuit.num_qubits,
            "hamiltonian": hamiltonian,
            "ratio": ratio,
        },
    }
    if run_on_hardware:
        var_dict["real_backend"] = {
            "name": real_backend.name,
            "error": relerr(real_backend_energy),
            "energy": real_backend_energy,
            "unique_bitstrings": real_unique_bitstring,
            "bitstring_prop": real_bitstring_prop,
            "n_shots": n_shots,
            "job": encode_sampler_result(real_job),
        }

    return var_dict
