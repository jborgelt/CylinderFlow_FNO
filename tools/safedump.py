#!/usr/bin/env python3
"""safedump: ncdump-style console viewer for safetensors weight files.

Needs only numpy. No torch, no safetensors package, no project build.

Usage:
  safedump.py FILE                  table of all tensors + metadata
  safedump.py FILE TENSOR           dump one tensor's values (head/tail)
  safedump.py FILE layers.0.R       R_real/R_imag auto-paired incl. |R| stats

Complex weights (stored as R_real + R_imag, since safetensors has no
complex dtype) are paired automatically when you name the R_ stem.
"""
import json
import sys

import numpy as np

DTYPES = {"F64": np.float64, "F32": np.float32}


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 8:
        sys.exit(f"{path}: file too small")
    n = int.from_bytes(raw[:8], "little")
    if n > len(raw) - 8:
        sys.exit(f"{path}: header length exceeds file")
    hdr = json.loads(raw[8 : 8 + n])
    buf = raw[8 + n :]
    meta = hdr.pop("__metadata__", {})
    tensors = {}
    for name, info in hdr.items():
        try:
            dt = DTYPES[info["dtype"]]
            a, b = info["data_offsets"]
        except (KeyError, TypeError):
            sys.exit(f"{path}: malformed entry '{name}'")
        arr = np.frombuffer(buf[a:b], dtype=dt).reshape(info["shape"])
        tensors[name] = (info["dtype"], arr)
    return meta, tensors


def stat_line(name, dtype, a):
    flat = a.ravel()
    extra = ""
    if np.isnan(flat).any() or np.isinf(flat).any():
        extra = "  <-- HAS NaN/Inf!"
    print(
        f"{name:28s} {str(list(a.shape)):24s} {dtype:3s} "
        f"min={flat.min():+.4e} max={flat.max():+.4e} "
        f"mean={flat.mean():+.4e}{extra}"
    )


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    meta, tensors = load(sys.argv[1])
    print(f"# {sys.argv[1]}: {len(tensors)} tensors")
    for k, v in meta.items():
        print(f"#   {k} = {v}")
    if len(sys.argv) == 2:
        # Table view; pair R_real/R_imag for magnitude stats.
        shown = set()
        for name in tensors:
            if name in shown:
                continue
            if name.endswith(".R_real") and name[:-5] + "_imag" in tensors:
                stem = name[:-7]
                _, re = tensors[name]
                _, im = tensors[name[:-5] + "_imag"]
                mag = np.abs(re.astype(np.float64) + 1j * im.astype(np.float64))
                stat_line(name, tensors[name][0], re)
                stat_line(name[:-5] + "_imag", tensors[name[:-5] + "_imag"][0], im)
                stat_line(stem + ".R_abs", "mag", mag)
                shown |= {name, name[:-5] + "_imag"}
            else:
                stat_line(name, *tensors[name])
                shown.add(name)
        return
    # Single-tensor view.
    want = sys.argv[2]
    if want in tensors:
        _, a = tensors[want]
        print(f"# {want} shape={list(a.shape)} dtype={tensors[want][0]}")
        flat = a.ravel()
        n = min(len(flat), 12)
        print("head:", np.array2string(flat[:n], precision=5, separator=", "))
        if len(flat) > n:
            print("tail:", np.array2string(flat[-n:], precision=5, separator=", "))
        return
    if want + ".R_real" in tensors or (
        want.endswith(".R") and want[:-2] + ".R_real" in tensors
    ):
        stem = want if want.endswith(".R") else want + ".R"
        _, re = tensors[stem + "_real"]
        _, im = tensors[stem + "_imag"]
        mag = np.abs(re.astype(np.float64) + 1j * im.astype(np.float64))
        print(f"# {want} (complex, shape={list(re.shape)})")
        print(f"# |R|: min={mag.min():.4e} max={mag.max():.4e} mean={mag.mean():.4e}")
        print("R[0,0,:4,:4] abs:")
        print(np.array2string(mag[0, 0, :4, :4], precision=3, suppress_small=True))
        return
    sys.exit(f"tensor '{want}' not found")


if __name__ == "__main__":
    main()
