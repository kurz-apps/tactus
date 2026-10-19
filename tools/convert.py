"""Convert the 8 allin1 fold checkpoints (.pth) to one safetensors file plus config.json. Needs torch and safetensors.
  convert.py <dir with harmonix-fold*.pth> <out_dir>
Tensor names are the PyTorch state_dict names prefixed with 'foldN.'; config.json has the shared config and per-fold thresholds."""
import json, sys
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file


def main(src, out):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tensors, folds, cfg = {}, [], None
    for n in range(8):
        path = next(src.glob(f"harmonix-fold{n}-*.pth"))
        ckpt = torch.load(path, map_location="cpu")
        c = dict(ckpt["config"])
        folds.append({"fold": n, "best_threshold_beat": c["best_threshold_beat"], "best_threshold_downbeat": c["best_threshold_downbeat"]})
        cfg = cfg or {k: v for k, v in c.items() if k not in ("fold", "best_threshold_beat", "best_threshold_downbeat")}
        for k, v in ckpt["state_dict"].items():
            tensors[f"fold{n}.{k}"] = v.contiguous()
    save_file(tensors, out / "allin1.safetensors")
    (out / "config.json").write_text(json.dumps({"config": cfg, "folds": folds}, indent=1))
    back = load_file(out / "allin1.safetensors")
    assert back.keys() == tensors.keys() and all(torch.equal(back[k], tensors[k]) for k in tensors)
    print(f"{len(tensors)} tensors, {sum(t.numel() for t in tensors.values())} parameters")


if __name__ == "__main__":
    main(*sys.argv[1:])
