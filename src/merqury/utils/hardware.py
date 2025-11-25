from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, RuntimeJob, SamplerV2

try:
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    from iqm.qiskit_iqm.fake_backends.fake_aphrodite import IQMFakeAphrodite
    from iqm.qiskit_iqm.fake_backends.fake_adonis import IQMFakeAdonis
except ModuleNotFoundError:
    pass


import io

from qiskit_ibm_runtime.fake_provider import (
    FakeLagosV2,
    FakeWashingtonV2,
    FakeMelbourneV2,
    FakeKolkataV2,
)
from typing import Literal, Union

from qiskit.primitives.base.sampler_result import SamplerResult
from qiskit.circuit.parametertable import ParameterView
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTSampler, AQTEstimator
from qiskit_aqt_provider.aqt_resource import AQTResource
from mqp.qiskit_provider import MQPProvider, MQPBackend

from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend


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
) -> RuntimeJob:
    if isinstance(backend, (IBMBackend, FakeBackendV2)):
        sampler = SamplerV2(mode=backend)

        payload = circuits_to_payload(backend, circuit, parameters, None)
        job = sampler.run(
            payload,
            shots=int(n_shots),
        )

    elif isinstance(backend, AQTResource):
        circuit = circuit.assign_parameters(parameters_dict(circuit, parameters))
        sampler = AQTSampler(backend)
        job = sampler.run([circuit], parameter_values=None, shots=n_shots)
    elif isinstance(backend, MQPBackend):
        circuit = circuit.assign_parameters(parameters_dict(circuit, parameters))
        job = backend.run(circuit, shots=n_shots, qasm3=True, queued=True)
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

