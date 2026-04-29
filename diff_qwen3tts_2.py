import numpy as np
from gguf import GGUFReader

ref = GGUFReader("/tmp/qwen3-tts-ref.gguf")

def get(name):
      for t in ref.tensors:
          if t.name == name:
              return np.array(t.data, copy=False)
      raise KeyError(name)

a = get("talker_inputs_embeds").astype(np.float32).reshape(-1)
b = np.fromfile("/tmp/qwen3-icl-dump/icl_prefill.bin", dtype=np.float32)
n = min(a.size, b.size)
a = a[:n]
b = b[:n]
d = np.abs(a - b)
print("icl_prefill", "n=", n, "max_abs=", float(d.max()), "mean_abs=", float(d.mean()))
