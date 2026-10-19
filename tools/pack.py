"""Pack a .npz (or a single .npy, stored as 'array') into a float32 safetensors file, which MLX Swift can load.
  pack.py <in.npz|in.npy> <out.safetensors>"""
import sys
import numpy as np
from safetensors.numpy import save_file

src, dst = sys.argv[1:3]
data = np.load(src)
arrays = {k: data[k] for k in data.files} if src.endswith(".npz") else {"array": data}
save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in arrays.items()}, dst)
print(f"{len(arrays)} arrays -> {dst}")
