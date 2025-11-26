from merqury.utils.json_utils import load_json, save_json
import os
from pathlib import Path


filenames = os.listdir(Path(Path(__file__).parent, "../output"))
for filename in filenames:
    d_out = load_json(filename)
    print(filename)

    for k in d_out["0"].keys():
        print(k)
        try:
            del d_out["1"][k]["unique_bitstrings"]
            del d_out["0"][k]["unique_bitstrings"]
            del d_out["1"][k]["sub_ham"]
            del d_out["0"][k]["sub_ham"]
        except Exception:
            pass
    save_json(filename, d_out)
