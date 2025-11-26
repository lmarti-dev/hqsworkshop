from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, RuntimeJobV2, SamplerV2

from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
from iqm.qiskit_iqm.fake_backends.fake_aphrodite import IQMFakeAphrodite
from iqm.qiskit_iqm.fake_backends.fake_adonis import IQMFakeAdonis
from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMFakeBackend
from iqm.qiskit_iqm import IQMProvider, IQMBackend, IQMJob


from qiskit import transpile

import io
import json

from typing import Tuple, Union

from qiskit.primitives.base.sampler_result_v1 import SamplerResult
from qiskit.circuit.parametertable import ParameterView

from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend


from qiskit import ClassicalRegister

from qiskit.circuit.parametertable import ParameterView
from qiskit.primitives import StatevectorSampler, PrimitiveJob, StatevectorEstimator


from qiskit_aer import AerJob


def get_counts_from_job(job: Union[RuntimeJobV2, PrimitiveJob], ind: int = 0) -> dict:
    if isinstance(job, RuntimeJobV2):
        key = list(job.result()[ind].data.keys())[0]
        counts = job.result()[ind].data[key].get_counts()
    elif isinstance(job, PrimitiveJob):
        res = job.result()
        try:
            qd = res.quasi_dists[ind]
            n_shots = res.metadata[ind]["shots"]
            counts = {k: int(qd[k] * n_shots) for k in qd.keys()}
        except AttributeError:
            counts = res[ind].join_data().get_counts()
    elif isinstance(job, (AerJob, IQMJob)):
        counts = job.result().get_counts(ind)
    else:
        raise ValueError(f"Unexpected job type: type {type(job)}")
    return counts


def sampler_result_to_json(sampler: SamplerResult):
    d = {}

    d["quasi_dists"] = [
        dict(sampler.quasi_dists[m]) for m in range(len(sampler.quasi_dists))
    ]
    d["metadata"] = sampler.metadata

    return d


def encode_sampler_result(job):
    res = job.result()
    if isinstance(res, SamplerResult):
        return sampler_result_to_json(res)
    else:
        return res


def get_token():
    HOME = Path(__file__).parent
    with io.open(Path(HOME, "token.dat"), "r", encoding="utf8") as f:
        token = f.read()
    return token


def get_ibm_backend(n_qubits: int) -> IBMBackend:
    token = get_token("IBM")
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    real_backend = service.least_busy(
        simulator=False, operational=True, min_num_qubits=n_qubits
    )
    return real_backend


def get_tfim_ratio(ham: SparsePauliOp) -> float:
    c = ham.coeffs
    u = np.unique_counts(c)
    idx = np.argsort(u.counts)
    j = u.values[idx[0]]
    h = u.values[idx[-1]]
    return np.real(j / h).astype(float)


def get_backend(device_name: str):
    url = f"https://cocos.resonance.meetiqm.com/{device_name}"
    backend = IQMProvider(url, token=get_token()).get_backend()
    return backend


def add_meas_layer(circuit: QuantumCircuit) -> QuantumCircuit:
    circuit = circuit.copy()
    meas_register = ClassicalRegister(circuit.num_qubits)
    circuit.add_register(meas_register)
    for q, c in zip(circuit.qubits, meas_register):
        circuit.measure(q, c)
    return circuit


def get_unique_bitstrings(counts: dict) -> list[str]:
    return np.unique(list(counts.keys()))


def binleftpad(integer: int, left_pad: int):
    b = format(integer, "0{lp}b".format(lp=left_pad))
    return b


def index_bits(
    a: Union[str, int], n_qubits: int = None, ones=True, right_to_left: bool = False
) -> list:
    """Takes a binary number and returns a list of indices where the bit is one (or zero)
    Args:
        a (binary number): The binary number whose ones or zeroes will be indexed
        ones (bool): If true, index ones. If false, index zeroes
    Returns:
        list: List of indices where a is one (or zero)
    """
    if isinstance(a, int):
        if n_qubits is None:
            b = bin(a)
        else:
            b = binleftpad(integer=a, left_pad=n_qubits)
    else:
        b = a
    if "b" in b:
        b = b.split("b")[1]
    if right_to_left:
        b = list(reversed(b))
    if ones:
        return [idx for idx, v in enumerate(b) if int(v)]
    elif not ones:
        return [idx for idx, v in enumerate(b) if not int(v)]


def to_bitstring(ind: int, n_qubits: int, right_to_left: bool = False) -> str:
    if isinstance(ind, (int, np.integer)):
        if n_qubits is None:
            b = bin(ind)
        else:
            b = binleftpad(integer=ind, left_pad=n_qubits)
    elif isinstance(ind, str):
        b = ind
    else:
        raise ValueError(f"Expected ind to be int or str, got: {type(ind)}")
    if "b" in b:
        b = b.split("b")[1]
    if right_to_left:
        b = "".join(list(reversed(b)))

    return b


def from_bitstring(b: str, n_qubits: int, right_to_left: bool = False) -> int:
    if "b" in b:
        b = b.split("b")[1]
    idx = index_bits(a=b, n_qubits=n_qubits, right_to_left=True)
    if right_to_left:
        return sum([2 ** (n_qubits - 1 - iii) for iii in idx])
    else:
        return sum([2**iii for iii in idx])


def subspace_diagonalize(
    unique_bitstrings: Union[list[int], list[str]], ham: SparsePauliOp
) -> Tuple[float, np.ndarray]:
    if isinstance(unique_bitstrings[0], str):
        ind_list = [
            from_bitstring(bx, len(bx), right_to_left=True) for bx in unique_bitstrings
        ]
    else:
        ind_list = [int(x) for x in unique_bitstrings]

    sparse_ham = ham.to_matrix(sparse=True)
    n_samples = len(unique_bitstrings)
    sub_ham = np.zeros((n_samples, n_samples), dtype=complex)
    for x in range(n_samples):
        for y in range(x, len(unique_bitstrings)):
            ind_x = ind_list[x]
            ind_y = ind_list[y]
            sub_ham[x, y] = sparse_ham[ind_x, ind_y]
    sub_ham += np.triu(sub_ham, k=1).T
    ground_energy = np.min(np.linalg.eigvalsh(sub_ham))
    return ground_energy, sub_ham


def isa_circuit_param_dict(
    new_circuit: QuantumCircuit, old_circuit: QuantumCircuit, parameters: np.ndarray
) -> dict:
    param_dict = parameters_dict(old_circuit, parameters)
    return {k: param_dict[k] for k in new_circuit.parameters}


def parameters_dict(
    circuit: Union[QuantumCircuit, list[QuantumCircuit]], parameters: np.ndarray
) -> dict:
    if not parameters_exist(circuit.parameters) or not parameters_exist(parameters):
        return {}
    return {k: v for k, v in zip(circuit.parameters, parameters)}


def parameters_exist(parameters) -> bool:
    x = (
        (parameters is None)
        or (isinstance(parameters, ParameterView) and parameters == ParameterView([]))
        or (len(parameters) == 0)
    )
    return not x


def backend_sample(
    backend,
    circuit,
    parameters: np.ndarray,
    n_shots: int,
) -> RuntimeJobV2:
    if isinstance(backend, (IBMBackend, FakeBackendV2)):
        sampler = SamplerV2(mode=backend)

        payload = circuits_to_payload(backend, circuit, parameters, None)
        job = sampler.run(
            payload,
            shots=int(n_shots),
        )

    elif isinstance(backend, (IQMFakeBackend, IQMBackend)):
        circuit = circuit.assign_parameters(parameters_dict(circuit, parameters))

        initial_layout = list(range(circuit.num_qubits))
        isa_circuits = transpile(
            circuit,
            backend=backend,
            initial_layout=initial_layout,
            optimization_level=1,
            coupling_map=backend.coupling_map,
        )
        payload = circuits_to_payload(backend, circuit, parameters, None)
        sampler = SamplerV2(mode=backend)
        job = backend.run(isa_circuits, shots=n_shots)
    else:
        raise ValueError(f"Unexpected backend: type {type(backend)}")
    return job


def circuits_to_payload(
    backend,
    circuit: Union[QuantumCircuit, list[QuantumCircuit]],
    parameters: np.ndarray,
    observables: Union[list[SparsePauliOp], SparsePauliOp],
    which: str = "v2",
) -> list[tuple[QuantumCircuit, dict]]:
    if isinstance(circuit, QuantumCircuit):
        circuit = [circuit]
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(circuit)
    if which == "v1":
        payload = [[], [], []]
        # assume that for each circuit you want to measure each parameters
        if isinstance(isa_circuit, QuantumCircuit):
            isa_circuit = [isa_circuit]
        if isinstance(observables, SparsePauliOp):
            observables = [observables]
        for c in isa_circuit:
            for o in observables:
                payload[0].append(c)
                isa_o = o.apply_layout(c.layout)
                payload[1].append(isa_o)
                if parameters_exist(parameters):
                    payload[2].append(parameters)
        return payload

    elif which == "v2":
        payload = []

        if isinstance(isa_circuit, QuantumCircuit):
            isa_circuit = [isa_circuit]

        for idx in range(len(isa_circuit)):
            toadd = []
            if observables is not None:
                toadd.append(
                    [o.apply_layout(isa_circuit[idx].layout) for o in observables]
                )
            if parameters_exist(parameters):
                isa_params = isa_circuit_param_dict(
                    isa_circuit[idx], circuit[idx], parameters
                )
                toadd.append(isa_params)
            payload.append((isa_circuit[idx], *toadd))

    return payload


def get_appropriate_fake_backend(n_qubits: int) -> Union[FakeBackendV2]:
    if n_qubits <= 5:
        return IQMFakeAdonis()
    elif n_qubits <= 20:
        return IQMFakeGarnet()
    else:
        return IQMFakeAphrodite()


def sv_estimate_observables(
    circuit: QuantumCircuit, observables: list[SparsePauliOp], param_sweep: np.ndarray
):
    estimator = StatevectorEstimator()
    job_observables = []
    for ind, o in enumerate(observables):
        if o.num_qubits < circuit.num_qubits:
            job_observables.append(
                SparsePauliOp("I" * (circuit.num_qubits - o.num_qubits)).expand(o)
            )
        else:
            job_observables.append(o)
    pub = (circuit, job_observables, param_sweep)
    job = estimator.run([pub])
    result = job.result()[0]
    return result


def measure_ground_state_energy_subspace_sampling(
    hamiltonian: SparsePauliOp,
    target_energy: float,
    circuit: QuantumCircuit,
    device_name: str,
    n_shots: int = 1e3,
    run_fake: bool = True,
    run_sv: bool = True,
    run_on_hardware: bool = False,
) -> dict:

    print(f"n_qubits: {hamiltonian.num_qubits}")

    circuit_to_sample = add_meas_layer(circuit)

    def relerr(e):
        return np.abs(e - target_energy) / np.abs(target_energy)

    if run_fake:
        fake_backend = get_appropriate_fake_backend(circuit.num_qubits)
        fake_job = backend_sample(
            backend=fake_backend,
            circuit=circuit_to_sample,
            parameters=[],
            n_shots=n_shots,
        )
        fake_counts = get_counts_from_job(fake_job)
        fake_unique_bitstring = get_unique_bitstrings(fake_counts)
        fake_backend_bitstring_prop = len(fake_unique_bitstring) / (
            2**circuit.num_qubits
        )
        fake_backend_energy, fake_sub_ham = subspace_diagonalize(
            unique_bitstrings=fake_unique_bitstring, ham=hamiltonian
        )

    if run_sv:
        statevector_energy = sv_estimate_observables(
            circuit=circuit, observables=[hamiltonian], param_sweep=[]
        ).data.evs[0]
        print(f"Statevector {relerr(statevector_energy):.5e}")

    if run_on_hardware:
        real_backend = get_backend(device_name)
        real_job = backend_sample(
            backend=real_backend,
            circuit=circuit_to_sample,
            parameters=[],
            n_shots=n_shots,
        )
        real_counts = get_counts_from_job(real_job)
        real_unique_bitstring = get_unique_bitstrings(real_counts)
        real_bitstring_prop = len(real_unique_bitstring) / (2**circuit.num_qubits)
        real_backend_energy, real_sub_ham = subspace_diagonalize(
            unique_bitstrings=real_unique_bitstring, ham=hamiltonian
        )

        print(
            f"RealBackend ({real_backend.name}) {relerr(real_backend_energy):.5e} sub. mat.: {len(real_unique_bitstring)} × {len(real_unique_bitstring)}: {real_bitstring_prop*100:.3f}%"
        )

    var_dict = {}
    if run_fake:
        print(
            f"FakeBackend ({fake_backend.name}) {relerr(fake_backend_energy):.5e} sub. mat.: {len(fake_unique_bitstring)} × {len(fake_unique_bitstring)}: {fake_backend_bitstring_prop*100:.3f}%"
        )

        var_dict["fake_backend"] = {
            "name": fake_backend.name,
            "error": relerr(fake_backend_energy),
            "energy": fake_backend_energy,
            "unique_bitstrings": fake_unique_bitstring,
            "bitstring_prop": fake_backend_bitstring_prop,
            "n_shots": n_shots,
            "sub_ham": fake_sub_ham,
        }

    if run_sv:
        var_dict["statevector"] = {
            "error": relerr(statevector_energy),
            "energy": statevector_energy,
        }
    var_dict["system"] = {
        "ground_energy": target_energy,
        "n_qubits": circuit.num_qubits,
        "hamiltonian": hamiltonian,
    }
    if run_on_hardware:
        var_dict["real_backend"] = {
            "name": real_backend.name,
            "error": relerr(real_backend_energy),
            "energy": real_backend_energy,
            "unique_bitstrings": real_unique_bitstring,
            "bitstring_prop": real_bitstring_prop,
            "n_shots": n_shots,
            "sub_ham": real_sub_ham,
        }

    return var_dict


def prep_nup_ndown_circ(n_up: int, n_down: int, n_qubits: int) -> QuantumCircuit:
    circ = QuantumCircuit(n_qubits)
    for ind_up in range(n_up):
        circ.x(2 * ind_up)
    for ind_down in range(n_down):
        circ.x(2 * ind_down + 1)
    return circ
