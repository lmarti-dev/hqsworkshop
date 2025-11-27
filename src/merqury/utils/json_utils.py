from json import JSONDecoder, JSONEncoder
from typing import Any

import numpy as np
from pathlib import Path
import io
import json

# don't need these libraries but they're useful for me.
try:
    import cirq

    CIRQ_TYPES = (
        cirq.Qid,
        cirq.Gate,
        cirq.Operation,
        cirq.Moment,
        cirq.AbstractCircuit,
        cirq.PauliSum,
        cirq.PauliString,
    )
    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False


try:
    import openfermion as of

    OF_TYPES = (of.SymbolicOperator,)
    HAS_OF = True
except ImportError:
    HAS_OF = False


try:
    import sympy

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


try:
    import qiskit
    from qiskit_ibm_runtime import RuntimeEncoder, RuntimeDecoder

    HAS_QISKIT_JSON = True
except ImportError:
    HAS_QISKIT_JSON = False

TYPE_FLAG = "type"
ARGS_FLAG = "args"
KWARGS_FLAG = "kwargs"


OUTPUT_DIRNAME = "output_v2"


class ExtendedJSONEncoder(JSONEncoder):
    def default(self, obj: Any) -> dict:
        if isinstance(obj, complex):
            return {
                TYPE_FLAG: "complex",
                KWARGS_FLAG: {"real": obj.real, "imag": obj.imag},
            }
        elif isinstance(obj, Path):
            return {
                TYPE_FLAG: "path",
                ARGS_FLAG: str(obj),
            }
        elif isinstance(obj, np.ndarray):
            return {
                TYPE_FLAG: obj.__class__.__name__,
                ARGS_FLAG: obj.tolist(),
                KWARGS_FLAG: {"dtype": str(obj.dtype)},
            }
        elif obj.__class__.__module__ == np.__name__:
            # longdouble.item() casts to longdouble. What's the point?
            if type(obj.item()) is type(obj):
                return {
                    TYPE_FLAG: obj.__class__.__name__,
                    ARGS_FLAG: obj.astype(float).item(),
                }
            return {
                TYPE_FLAG: obj.__class__.__name__,
                ARGS_FLAG: obj.tolist(),
            }
        elif HAS_CIRQ and isinstance(obj, CIRQ_TYPES):
            return {
                TYPE_FLAG: obj.__class__.__name__,
                ARGS_FLAG: cirq.to_json(obj, indent=4),
            }

        elif (HAS_OF and isinstance(obj, OF_TYPES)) or (
            HAS_SYMPY and isinstance(obj, sympy.Symbol)
        ):
            return {TYPE_FLAG: obj.__class__.__name__, ARGS_FLAG: str(obj)}

        elif HAS_QISKIT_JSON:
            try:
                return RuntimeEncoder().default(obj)
            except Exception:
                pass

        return super().default(obj)


class ExtendedJSONDecoder(JSONDecoder):
    def __init__(self):
        JSONDecoder.__init__(self, object_hook=self.object_hook)

    def object_hook(self, dct: dict) -> Any:
        if TYPE_FLAG in dct:
            t = get_type(dct[TYPE_FLAG])
            args = []
            kwargs = {}
            if ARGS_FLAG in dct:
                args.append(dct[ARGS_FLAG])
            if KWARGS_FLAG in dct:
                kwargs.update(dct[KWARGS_FLAG])
            return t(*args, **kwargs)
        elif HAS_QISKIT_JSON:
            return RuntimeDecoder().object_hook(dct)
        return dct


# A+ type hinting
def try_get_attr(cls: np.__class__, obj: Any):
    try:
        return getattr(cls, obj)
    except Exception:
        return None


def get_type(s: str) -> Any:
    try:
        # make it fail fast if needed
        # somehow getattr(__builtins__, "complex") raises an error. why?
        assert s == "complex"
        return complex
    except Exception:
        pass
    try:
        # hate this
        assert getattr(np, s).__module__ == np.__name__
        if s == "ndarray":
            return np.array
        return getattr(np, s)
    except Exception:
        pass
    try:
        assert s == "path"
        return Path
    except Exception:
        pass
    # the attr is the class with desired constructor
    # would be nice to do a for loop without including the non-imported module
    # names explicitly
    if HAS_CIRQ:
        try:
            getattr(cirq, s)
            return lambda x: cirq.read_json(json_text=x)
        # TODO: figure out which errors would show up
        except Exception:
            pass
    if HAS_SYMPY:
        x = try_get_attr(sympy, s)
        if x is not None:
            return x
    if HAS_OF:
        x = try_get_attr(of, s)
        if x is not None:
            return x
    if HAS_QISKIT_JSON:
        x = try_get_attr(qiskit, s)
        if x is not None:
            return x
    raise TypeError("{} is an unknown type".format(s))


def save_json(filename: Path, jobj: dict) -> None:
    dirname = Path(Path(__file__).parent, f"../{OUTPUT_DIRNAME}")
    with io.open(Path(dirname, filename), "w+", encoding="utf8") as f:
        f.write(json.dumps(jobj, ensure_ascii=False, indent=4, cls=ExtendedJSONEncoder))


def load_json(filename: Path) -> dict:
    dirname = Path(Path(__file__).parent, f"../{OUTPUT_DIRNAME}")
    with io.open(Path(dirname, filename), "r", encoding="utf8") as f:
        jobj = json.loads(f.read(), cls=ExtendedJSONDecoder)

    return jobj
