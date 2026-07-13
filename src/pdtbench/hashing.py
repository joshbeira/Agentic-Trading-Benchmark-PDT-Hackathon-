"""Canonical hashing.

Parquet file bytes are not reproducible across pyarrow versions or platforms
(metadata, compression, row-group layout all drift), so nothing here hashes a
file. Every hash is taken over a canonical serialization of the *values*, in a
fixed column and row order. That way `dataset_sha256` means "this exact data"
rather than "this exact file", which is what the tick log actually needs to pin.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

_SEP = b"\x1f"


def hash_bytes(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
        h.update(_SEP)
    return h.hexdigest()


def hash_frame(df: pd.DataFrame, cols: Sequence[str]) -> str:
    """Hash a frame's values over `cols`, in the frame's current row order.

    Floats go in as raw little-endian float64 so the digest is exact rather than
    format-dependent. Everything else goes in as its string repr.
    """
    h = hashlib.sha256()
    for col in cols:
        s = df[col]
        h.update(col.encode())
        h.update(_SEP)
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            arr = np.ascontiguousarray(s.to_numpy(dtype=np.float64))
            if arr.dtype.byteorder == ">":  # pragma: no cover - big-endian hosts
                arr = arr.astype("<f8")
            h.update(arr.tobytes())
        else:
            h.update(_SEP.join(str(v).encode() for v in s))
        h.update(_SEP)
    return h.hexdigest()


def hash_json(obj: Any) -> str:
    """Hash a JSON-able object canonically (sorted keys, no incidental whitespace)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
