# -*- coding: utf-8 -*-
"""Convert the bnb-nf4 merged Gemma (models/gemma-merged) to GGUF without writing a bf16 copy to disk.

Each 4-bit weight is dequantized on the GPU on demand, then llama.cpp's converter writes it.
Usage (.venv python):  python tools/bnb_to_gguf.py <out.gguf> [outtype=q8_0]
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
import bitsandbytes.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "gemma-merged"
LLAMA = ROOT / "tools" / "llama-src"
CFG_DIR = ROOT / "cache" / "tmp" / "gemma-cfg"   # config without quantization_config, tokenizer files

sys.path.insert(0, str(LLAMA))
sys.path.insert(0, str(LLAMA / "gguf-py"))

AUX = (".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map", ".quant_state.bitsandbytes__nf4")


def prepare_cfg_dir():
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    for f in SRC.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and not f.name.startswith("."):
            shutil.copy2(f, CFG_DIR / f.name)
    cfg = json.loads((SRC / "config.json").read_text(encoding="utf-8"))
    cfg.pop("quantization_config", None)
    (CFG_DIR / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def main():
    out = sys.argv[1]
    outtype = sys.argv[2] if len(sys.argv) > 2 else "q8_0"
    prepare_cfg_dir()

    from conversion.base import ModelBase, LazyTorchTensor

    st = safe_open(str(SRC / "model.safetensors"), framework="pt", device="cpu")
    keys = set(st.keys())

    def dequant(name):
        qs = {k[len(name) + 1:]: st.get_tensor(k) for k in keys if k.startswith(name + ".")}
        state = F.QuantState.from_dict(qs, device=torch.device("cuda"))
        w = st.get_tensor(name).cuda()
        return F.dequantize_4bit(w, state).to(torch.bfloat16).cpu(), state.shape

    def index_local_tensors(self, prefix):
        tensors = {}
        for name in sorted(keys):
            if name.endswith(AUX):
                continue
            if name + ".quant_state.bitsandbytes__nf4" in keys:
                shape = F.QuantState.from_dict(
                    {k[len(name) + 1:]: st.get_tensor(k) for k in keys if k.startswith(name + ".")},
                    device=torch.device("cpu")).shape
                meta = LazyTorchTensor.meta_with_dtype_and_shape(torch.bfloat16, tuple(shape))
                gen = lambda n=name, m=meta: LazyTorchTensor(meta=m, args=(n,), func=lambda n: dequant(n)[0])  # noqa: E731
            else:
                t = st.get_slice(name)
                dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[t.get_dtype()]
                meta = LazyTorchTensor.meta_with_dtype_and_shape(dtype, tuple(t.get_shape()))
                gen = lambda n=name, m=meta: LazyTorchTensor(meta=m, args=(n,), func=lambda n: st.get_tensor(n))  # noqa: E731
            titem = self.filter_tensors((name, gen))
            self._source_tensor_names[name] = titem[0] if titem else None
            if titem:
                tensors[titem[0]] = titem[1]
        return tensors

    ModelBase.index_local_tensors = index_local_tensors
    import convert_hf_to_gguf
    sys.argv = ["convert_hf_to_gguf.py", str(CFG_DIR), "--outfile", out, "--outtype", outtype]
    convert_hf_to_gguf.main()


if __name__ == "__main__":
    main()
