import numpy as np
from pathlib import Path
import os
import shutil, os

class CorrespondanceCache:
    def __init__(self,cache_root):
        self.root = Path(cache_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, scene_id):
        return self.root / f"{scene_id}.npz"

    def exists(self, scene_id):
        return self._path(scene_id).exists()

    def write_scene(self, scene_id, records):
        X0, X1, xy0, xy1, conf = [], [], [], [], []
        K0, K1, pair_ids = [], [], []
        offsets = [0]

        for r in records:
            n = len(r["X0"])
            X0.append(np.asarray(r["X0"], dtype=np.float32).reshape(n, 3))
            X1.append(np.asarray(r["X1"], dtype=np.float32).reshape(n, 3))
            xy0.append(np.asarray(r["xy0"], dtype=np.float32).reshape(n, 2))
            xy1.append(np.asarray(r["xy1"], dtype=np.float32).reshape(n, 2))
            conf.append(np.asarray(r["conf"], dtype=np.float32).reshape(n))
            K0.append(np.asarray(r["K0"], dtype=np.float32).reshape(3, 3))
            K1.append(np.asarray(r["K1"], dtype=np.float32).reshape(3, 3))
            pair_ids.append(int(r["pair_id"]))
            offsets.append(offsets[-1] + n)

        def cat(chunks, width):
            if chunks:
                return np.concatenate(chunks, axis=0)
            return np.zeros((0, width) if width else (0,), dtype=np.float32)

        arrays = dict(
            X0=cat(X0, 3), X1=cat(X1, 3),
            xy0=cat(xy0, 2), xy1=cat(xy1, 2),
            conf=cat(conf, 0),
            K0=np.stack(K0) if K0 else np.zeros((0, 3, 3), np.float32),
            K1=np.stack(K1) if K1 else np.zeros((0, 3, 3), np.float32),
            pair_ids=np.asarray(pair_ids, dtype=np.int64),
            offsets=np.asarray(offsets, dtype=np.int64),
        )

        final = self._path(scene_id)
        local_tmp = f"/content/{scene_id}.npz"      # local disk — reliable write
        np.savez_compressed(local_tmp, **arrays)
        shutil.copy(local_tmp, final)               # copy to Drive
        os.remove(local_tmp)

    def read_scene(self, scene_id):
        with np.load(self._path(scene_id)) as data:
            offsets = data["offsets"]
            pair_ids = data["pair_ids"]
            X0, X1 = data["X0"], data["X1"]
            xy0, xy1 = data["xy0"], data["xy1"]
            conf = data["conf"]
            K0, K1 = data["K0"], data["K1"]

            records = []
            for k in range(len(pair_ids)):
                a, b = offsets[k], offsets[k + 1]
                records.append(dict(
                    pair_id=int(pair_ids[k]),
                    X0=X0[a:b], X1=X1[a:b],
                    xy0=xy0[a:b], xy1=xy1[a:b],
                    conf=conf[a:b],
                    K0=K0[k], K1=K1[k],
                ))
        return records




