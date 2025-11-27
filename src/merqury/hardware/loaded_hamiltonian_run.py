from pathlib import Path

from merqury.utils.hardware import (
    measure_ground_state_energy_subspace_sampling,
    loop_measure_ground_state_energy_subspace_sampling_v2,
    subspace_diagonalize,
    subspace_diagonalize_qiskit,
)
from merqury.utils.json_utils import save_json
import os
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
    kill_diagonal,
)


from qiskit.quantum_info import (
    SparsePauliOp,
    Statevector,
    state_fidelity,
)
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from qiskit.primitives import StatevectorSampler


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


def compare_statevectors(
    filename: str,
):

    sv = {}
    sampler = StatevectorSampler()
    for nth in (0, 1):

        print(f"==== nth: {nth} ====")

        ham1 = load_ham(filename)
        ham1 = kill_diagonal(ham1)

        ham0 = get_diagonal_ham(ham1)
        bitstring = get_nth_bitstring_v2(ham0, nth, False)
        prep_circ = prep_circ_bitstring(bitstring)

        mult = 10

        total_time = mult / (get_min_gap(ham1) ** 2)

        n_steps = mult * ham0.num_qubits**2

        true_energy = get_nth_energy(ham1, nth)
        sweep_circuit = get_sweep_circuit(
            ham0, ham1, total_time=total_time, n_steps=n_steps
        )

        prep_circ.compose(sweep_circuit, inplace=True)

        sv[nth] = Statevector(prep_circ).data

    mat = ham1.to_matrix(True)
    eigenvalues, eigvecs = eigsh(mat, k=4, which="SA")

    print(np.round(eigvecs[:, 0], 4))
    print(np.round(sv[0], 4))
    print(np.round(eigvecs[:, 1], 4))
    print(np.round(sv[1], 4))

    for jj in range(2):
        for kk in range(2):
            fid = state_fidelity(sv[jj], eigvecs[:, kk])
            print(f"fidelity ({jj},{kk}) {fid:.5e}")

    fig, ax = plt.subplots()

    for kk in range(2):
        ax.plot(range(len(eigvecs[:, kk])), np.abs(eigvecs[:, kk]), label=f"eig {kk}")
        ax.plot(range(len(eigvecs[:, kk])), np.abs(sv[kk]), label=f"sweep {kk}")
    ax.set_yscale("log")
    ax.legend()
    plt.show()


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
        bitstring = get_nth_bitstring_v2(ham0, nth, False)
        prep_circ = prep_circ_bitstring(bitstring)

        mult = 1

        total_time = mult / (get_min_gap(ham1) ** 2)

        n_steps = int(mult * ham0.num_qubits)

        if run_sv:
            true_energy = get_nth_energy(ham1, nth)
        else:
            true_energy = 1000
        sweep_circuit = get_sweep_circuit(
            ham0, ham1, total_time=total_time, n_steps=n_steps
        )
        print("sweep circuit ready")
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

    if solve_ham:
        s_true = diagonalize_hamiltonian_splitting(ham1)
        energy_dict["system"]["splitting"] = s_true
        print(f"True Splitting: {s_true:.5e}")

    for k in d_out[0].keys():
        if (
            "unique_bitstrings" in d_out[0][k].keys()
            and "unique_bitstrings" in d_out[1][k].keys()
        ):
            bs = list(
                set(
                    [
                        *d_out[1][k]["unique_bitstrings"],
                        *d_out[0][k]["unique_bitstrings"],
                    ]
                )
            )
            _, sub_ham = subspace_diagonalize(bs, ham1)
            sp = singlet_triplet_splitting(sub_ham)
            print(f"Splitting (e1-e0): {sp:.5e} for {k}")
            energy_dict[k]["splitting"] = sp
    for k in d_out[0].keys():
        try:
            del d_out[1][k]["unique_bitstrings"]
            del d_out[0][k]["unique_bitstrings"]
            del d_out[1][k]["sub_ham"]
            del d_out[0][k]["sub_ham"]
        except Exception:
            pass

    return d_out


def merge_counts(d_out: dict, which_backend: str) -> dict:
    all_counts = d_out[1][which_backend]["counts"]
    for k in d_out[0][which_backend]["counts"]:
        if k not in all_counts.keys():
            all_counts[k] = d_out[0][which_backend]["counts"][k]
        else:
            all_counts[k] += d_out[0][which_backend]["counts"][k]
    bs_len = len(all_counts[k])
    hilb_size = 2**bs_len
    n_bs = len(all_counts)
    print(f"bitstring subspace: {n_bs}/{hilb_size}, {n_bs/hilb_size:.5e}%")
    return all_counts


def compute_ground_energy_sweep_v2(
    filename: str, n_shots: int, which_backends: list, solve: bool
) -> dict:

    d_out = {}

    for nth in (0, 1):

        print(f"==== nth: {nth} ====")

        ham1 = load_ham(filename)

        ham0 = get_diagonal_ham(ham1)
        bitstring = get_nth_bitstring_v2(ham0, nth, False)
        prep_circ = prep_circ_bitstring(bitstring)

        mult = 1

        total_time = mult / (get_min_gap(ham1) ** 2)

        n_steps = int(mult * ham0.num_qubits)

        if solve:
            true_energy = get_nth_energy(ham1, nth)
        else:
            true_energy = 1000
        sweep_circuit = get_sweep_circuit(
            ham0, ham1, total_time=total_time, n_steps=n_steps
        )
        print("sweep circuit ready")
        prep_circ.compose(sweep_circuit, inplace=True)
        energy_dict = loop_measure_ground_state_energy_subspace_sampling_v2(
            ham1, prep_circ, which_backends, true_energy, n_shots=n_shots
        )

        d_out[nth] = energy_dict

    if solve:
        s_true = diagonalize_hamiltonian_splitting(ham1)
        energy_dict["system"]["splitting"] = s_true
        print(f"True Splitting: {s_true:.5e}")

    assert d_out[0].keys() == d_out[1].keys()
    keys = d_out[0].keys()

    for k in keys:
        if k != "system":
            all_counts = merge_counts(d_out, k)
            eig_energies, eig_vecs = subspace_diagonalize_qiskit(all_counts, ham1)
            sp = compute_splittings_e(eig_energies[1], eig_energies[0])
            print(f"Splitting (e1-e0): {sp:.5e} for {k}")
            energy_dict[k]["splitting"] = sp
    for k in keys:
        try:

            del d_out[1][k]["counts"]
            del d_out[0][k]["counts"]
        except Exception:
            pass

    return d_out


def get_n_qubits(fn: str) -> int:
    fn = fn.replace("pauli_hamiltonian_", "")
    fn = fn.replace("qubits.npy", "")
    return int(fn)


def out_exists(filename):
    p = Path(Path(__file__).parent, "../output/", Path(filename).stem + ".json")
    print(f"path {p} exists: {p.exists()}")
    return p.exists()


def run_v1():
    filenames = os.listdir(Path(Path(__file__).parent, "../files"))

    filenames = sorted(filenames, key=lambda x: get_n_qubits(x))

    dry_run = True

    for filename in filenames:

        if not out_exists(filename) or dry_run:
            if get_n_qubits(filename) < 14:
                solve_ham = True
                run_sv = True
                run_fake = True
                run_on_hardware = True
            else:
                solve_ham = False
                run_sv = False
                run_fake = False
                run_on_hardware = True

            d_out = compute_ground_energy_sweep(
                filename,
                run_on_hardware=run_on_hardware,
                run_fake=run_fake,
                run_sv=run_sv,
                solve_ham=solve_ham,
                n_shots=1000,
                device_name="emerald",
            )
            if not dry_run:
                save_json(Path(filename).stem + ".json", d_out)


def run_v2():
    filenames = os.listdir(Path(Path(__file__).parent, "../files"))

    filenames = sorted(filenames, key=lambda x: get_n_qubits(x))

    dry_run = False

    for filename in filenames:
        n_qubits = get_n_qubits(filename)
        if not out_exists(filename) or dry_run:
            if n_qubits < 16:
                solve = True
                which_backends = ["statevector", "fake_backend"]
            else:
                solve = False
                which_backends = ["real_backend"]

            d_out = compute_ground_energy_sweep_v2(
                filename,
                which_backends=which_backends,
                solve=solve,
                n_shots=1000,
            )
            if not dry_run:
                save_json(Path(filename).stem + ".json", d_out)

            if n_qubits == 14:
                break


if __name__ == "__main__":
    run_v2()
