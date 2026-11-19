"""Verify dataset SHA-256 hashes and the loss-array shapes; requires only NumPy."""
import hashlib
import json
from pathlib import Path
import numpy as np
from dataset import regroup_losses


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "DATA_MANIFEST.json").read_text())
    for name, record in manifest["files"].items():
        path = root / name
        assert path.stat().st_size == record["bytes"], f"Size mismatch: {name}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], f"Hash mismatch: {name}"
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                assert set(data.files) == set(record["arrays"])
                for key, spec in record["arrays"].items():
                    array = data[key]
                    assert list(array.shape) == spec["shape"] and str(array.dtype) == spec["dtype"]
                loss = data["loss"]
                assert np.all(np.isfinite(loss)) and np.all(loss >= 0)
                flat = loss.reshape(-1, loss.shape[-1])
                for arms in [8, 16, 32, 64]:
                    grouped = regroup_losses(loss, arms)
                    assert np.array_equal(grouped.reshape(flat.shape), flat)
                    assert not grouped.flags.writeable
        print("OK", name)
    print("All corpus files match the published manifest; regrouping preserves every tape.")


if __name__ == "__main__":
    main()
