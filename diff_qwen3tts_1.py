import numpy as np
from gguf import GGUFReader

ref = GGUFReader("/tmp/qwen3-tts-ref.gguf")

def get(name):
      for t in ref.tensors:
          if t.name == name:
              return np.array(t.data, copy=False)
      raise KeyError(name)

pairs = [
      ("icl_role", "/tmp/qwen3-icl-dump/icl_role.bin"),
      ("icl_bridge", "/tmp/qwen3-icl-dump/icl_bridge.bin"),
      ("icl_codec_input", "/tmp/qwen3-icl-dump/icl_codec_input.bin"),
      ("icl_text_embed", "/tmp/qwen3-icl-dump/icl_text_embed.bin"),
      ("icl_codec_embed", "/tmp/qwen3-icl-dump/icl_codec_embed.bin"),
      ("icl_input", "/tmp/qwen3-icl-dump/icl_input.bin"),
  ]

for name, path in pairs:
      a = get(name).astype(np.float32).reshape(-1)
      b = np.fromfile(path, dtype=np.float32)
      n = min(a.size, b.size)
      a = a[:n]
      b = b[:n]
      d = np.abs(a - b)
      print(name, "n=", n, "max_abs=", float(d.max()), "mean_abs=", float(d.mean()))

