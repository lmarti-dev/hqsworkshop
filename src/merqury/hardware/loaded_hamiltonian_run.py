from pathlib import Path

from merqury.utils.hardware import measure_ground_state_energy_subspace_sampling
from merqury.utils.json_utils import save_json

from merqury.algorithms.sweep import get_sweep_circuit
from merqury.hamiltonians.pauliops import (
    get_min_gap,
    get_nth_energy,
    load_ham,
    get_diagonal_ham,
    get_nth_bitstring,
    get_nth_bitstring_v2,
    prep_circ_bitstring,
    singlet_triplet_splitting,
    diagonalize_hamiltonian_splitting,
)


def compute_splittings(d: dict):
    splits = {}
    for k in d:
        if "sub_ham" in d[k].keys():
            v = singlet_triplet_splitting(d[k]["sub_ham"])
            print(f"Splitting: {v:.5e} for {k}")
            splits[k] = v
    return splits


def compute_splittings_e(e1: float, e0: float):

    return (e1 - e0) * 27.2114


def compute_ground_energy_sweep(
    filename: str,
    n_shots: int,
    device_name: str = "garnet",
    run_on_hardware: bool = False,
    run_sv: bool = True,
    run_fake: bool = True,
    solve_ham: bool = True,
) -> dict:

    d_out = {}

    for nth in (0, 1):

        print(f"==== nth: {nth} ====")

        ham1 = load_ham(filename)

        ham0 = get_diagonal_ham(ham1)
        bitstring = get_nth_bitstring_v2(ham0, nth)
        prep_circ = prep_circ_bitstring(bitstring)

        mult = 1

        total_time = mult / (get_min_gap(ham1) ** 2)

        n_steps = mult * ham0.num_qubits**2

        n_shots = 1000

        true_energy = get_nth_energy(ham1, nth)
        sweep_circuit = get_sweep_circuit(
            ham0, ham1, total_time=total_time, n_steps=n_steps
        )

        prep_circ.compose(sweep_circuit, inplace=True)
        energy_dict = measure_ground_state_energy_subspace_sampling(
            ham1,
            true_energy,
            prep_circ,
            run_fake=run_fake,
            run_sv=run_sv,
            run_on_hardware=run_on_hardware,
            device_name=device_name,
            n_shots=n_shots,
        )
        d_out[nth] = energy_dict

    splits = {}
    for k in d_out[0].keys():
        if "energy" in d_out[0][k].keys() and "energy" in d_out[1][k].keys():
            splits[k] = compute_splittings_e(
                d_out[1][k]["energy"], d_out[0][k]["energy"]
            )
            print(f"Splitting: {splits[k]:.5e} for {k}")

    for k in splits:
        energy_dict[k]["splitting"] = splits[k]

    if solve_ham:
        s_true = diagonalize_hamiltonian_splitting(ham1)
        energy_dict["system"]["splitting"] = s_true
        print(f"True Splitting: {s_true:.5e}")

    return d_out


if __name__ == "__main__":
    filename = "pauli_hamiltonian_6qubits.npy"

    energy_dict = compute_ground_energy_sweep(
        filename, run_on_hardware=False, run_fake=True, run_sv=True, n_shots=1000
    )

    save_json(Path(filename).stem + ".json", energy_dict)
