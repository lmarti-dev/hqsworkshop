from merqury.utils.json_utils import load_json
import os
from pathlib import Path
import matplotlib.pyplot as plt

BACKENDS = ("real_backend", "fake_backend", "statevector")


def get_n_qubits(fn: str) -> int:
    fn = fn.replace("pauli_hamiltonian_", "")
    fn = fn.replace("qubits.json", "")
    return int(fn)


def get_backend_splits(d: dict, backend: str):
    y = []
    x = []

    for ind, k in enumerate(sorted(d.keys())):
        print(k, d[k].keys())
        if not backend in d[k]["1"].keys():
            break
        subd = d[k]["1"][backend]
        if "splitting" not in subd.keys():
            break
        else:
            x.append(k)
            y.append(subd["splitting"])
    return x, y


filenames = os.listdir(Path(Path(__file__).parent, "../output"))
d = {}

for filename in filenames:
    d[get_n_qubits(filename)] = load_json(filename)


fig, ax = plt.subplots()

mark = "xosd"
for ind, b in enumerate(BACKENDS):
    x, y = get_backend_splits(d, b)
    ax.plot(x, y, f"{mark[ind]}-", label=b)


ax.legend()

ax.set_ylabel("Energy splitting")
ax.set_xlabel("Qubits")

plt.savefig(Path(Path(__file__).parent, "../plots/main.png"))

plt.show()
