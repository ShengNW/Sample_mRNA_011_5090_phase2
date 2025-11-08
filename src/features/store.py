
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
FeatureStore with organ_id type auto-detection (auto/int/str).
"""
import os, json, csv
from typing import Dict, Tuple, Optional, Any, Literal
import numpy as np

def _load_kv_matrix(path: str) -> Tuple[Dict[Any, int], np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"feature file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        obj = np.load(path, allow_pickle=True).item()
        keys = obj["keys"]; data = np.asarray(obj["data"], dtype=np.float32)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        keys = obj["keys"]; data = np.asarray(obj["data"], dtype=np.float32)
    elif ext == ".csv":
        keys = []
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                keys.append(row[0])
                rows.append([float(x) for x in row[1:]])
        data = np.asarray(rows, dtype=np.float32)
    else:
        raise ValueError(f"unsupported feature file ext: {ext}")
    key2row = {k:i for i,k in enumerate(keys)}
    return key2row, data

def _try_all_int(keys) -> bool:
    try:
        for k in keys:
            int(k)
        return True
    except Exception:
        return False

class FeatureStore:
    def __init__(self, tissue_path: Optional[str]=None, rbp_path: Optional[str]=None,
                 struct_path: Optional[str]=None, organ_id_type: Literal["auto","int","str"]="auto"):
        self.tissue_key2row, self.tissue = ({}, None)
        self.rbp_key2row, self.rbp = ({}, None)
        self.struct_key2row, self.struct = ({}, None)
        self._organ_id_type = organ_id_type
        if tissue_path:
            self.tissue_key2row, self.tissue = _load_kv_matrix(tissue_path)
        if rbp_path:
            self.rbp_key2row, self.rbp = _load_kv_matrix(rbp_path)
        if struct_path:
            self.struct_key2row, self.struct = _load_kv_matrix(struct_path)
        # auto decide
        if self._organ_id_type == "auto":
            sample_keys = []
            if self.tissue_key2row: sample_keys = list(self.tissue_key2row.keys())[:5]
            elif self.rbp_key2row: sample_keys = list(self.rbp_key2row.keys())[:5]
            self._organ_id_type = "int" if (sample_keys and _try_all_int(sample_keys)) else "str"

    def _norm_organ_key(self, organ_id):
        if organ_id is None: return None
        if self._organ_id_type == "int":
            try: return int(organ_id)
            except Exception: return str(organ_id)
        return str(organ_id)

    def get_tissue(self, organ_id):
        if self.tissue is None: return None
        key = self._norm_organ_key(organ_id)
        ridx = self.tissue_key2row.get(key, None)
        if ridx is None: return None
        return self.tissue[ridx]

    def get_rbp(self, organ_id):
        if self.rbp is None: return None
        key = self._norm_organ_key(organ_id)
        ridx = self.rbp_key2row.get(key, None)
        if ridx is None: return None
        return self.rbp[ridx]

    def get_struct(self, transcript_id):
        if self.struct is None: return None
        key = str(transcript_id)
        ridx = self.struct_key2row.get(key, None)
        if ridx is None: return None
        return self.struct[ridx]

    @property
    def dims(self):
        t = self.tissue.shape[1] if isinstance(self.tissue, np.ndarray) else 0
        r = self.rbp.shape[1] if isinstance(self.rbp, np.ndarray) else 0
        s = self.struct.shape[1] if isinstance(self.struct, np.ndarray) else 0
        return dict(tissue=t, rbp=r, struct=s)
