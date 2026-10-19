"""Reference outputs from the original allin1 + NATTEN. Run with the python of an env that has allin1 (CPU natten), e.g.
  docker run --rm -v $PWD:/work -e HF_HOME=/work/fixtures/hf isolate-server /opt/allin1/bin/python /work/tools/reference.py ...

  reference.py song <stems_dir> <out_dir>   stems_dir has bass/drums/other/vocals.wav
      -> out_dir/spec.npy       (4, T, 81) madmom spectrogram
         out_dir/logits.npz     per-fold and ensemble logits (beat, downbeat, section, function)
         out_dir/dbn_input.npy  (T, 2) what madmom's DBN receives
         out_dir/result.json    allin1's final output
  reference.py na <out_dir>                 random neighborhood-attention inputs with NATTEN's outputs
"""
import json, sys
from pathlib import Path
import numpy as np
import torch


def song(stems, out):
    from allin1.spectrogram import extract_spectrograms
    from allin1.models import load_pretrained_model
    from allin1.models.ensemble import Ensemble
    from allin1.models.loaders import ENSEMBLE_MODELS
    from allin1.postprocessing import postprocess_metrical_structure, postprocess_functional_structure, estimate_tempo_from_beats
    from omegaconf import OmegaConf

    stems, out = Path(stems), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec_path = extract_spectrograms([stems], out / "spec_tmp", multiprocess=False)[0]
    spec = np.load(spec_path)
    np.save(out / "spec.npy", spec)
    spec_path.unlink()
    spec_path.parent.rmdir()
    x =torch.from_numpy(spec).unsqueeze(0)

    names = ENSEMBLE_MODELS["harmonix-all"]
    models = [load_pretrained_model(n, device="cpu") for n in names]
    saved = {}
    with torch.no_grad():
        for n, m in zip(names, models):
            o = m(x)
            for k in ("beat", "downbeat", "section", "function"):
                saved[f"{n}_{k}"] = getattr(o, f"logits_{k}")[0].numpy()
        ens = Ensemble(models).eval()
        o = ens(x)
        for k in ("beat", "downbeat", "section", "function"):
            saved[f"ensemble_{k}"] = getattr(o, f"logits_{k}")[0].numpy()
    np.savez(out / "logits.npz", **saved)

    # same arithmetic as allin1/postprocessing/metrical.py, to capture the DBN input
    b, d = torch.sigmoid(o.logits_beat[0]), torch.sigmoid(o.logits_downbeat[0])
    no = ((1. - b) + (1. - d)) / 2.
    c = torch.stack([torch.maximum(torch.tensor(1e-8), b - d), d, no], dim=-1)
    c /= c.sum(dim=-1, keepdim=True)
    np.save(out / "dbn_input.npy", c.numpy()[:, :2])

    metrical = postprocess_metrical_structure(o, ens.cfg)
    segments = postprocess_functional_structure(o, ens.cfg)
    (out / "result.json").write_text(json.dumps({
        "bpm": estimate_tempo_from_beats(metrical["beats"]),
        **metrical,
        "segments": [{"start": float(s.start), "end": float(s.end), "label": s.label} for s in segments],
        "best_threshold_downbeat": ens.cfg.best_threshold_downbeat,
        "best_threshold_beat": ens.cfg.best_threshold_beat,
        "thresholds_per_fold": [float(m.cfg.best_threshold_downbeat) for m in models],
        "config": OmegaConf.to_container(models[0].cfg, resolve=True),
    }, indent=1))
    print(f"frames {spec.shape[1]}, beats {len(metrical['beats'])}, segments {len(segments)}")


def na(out):
    from natten.functional import natten1dav, natten1dqkrpb, natten2dav, natten2dqkrpb

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    heads, dim, k = 2, 12, 5
    cases = {}
    # lengths must be >= k * dilation (the model pads shorter inputs); include lengths that are not a multiple of the dilation
    for T, d in [(5, 1), (12, 1), (37, 1), (10, 2), (31, 2), (45, 4), (100, 4), (123, 8), (300, 16), (1000, 64), (1031, 128), (5200, 1024)]:
        q, kk, v, rpb = r(2, heads, T, dim), r(2, heads, T, dim), r(2, heads, T, dim), r(heads, 2 * k - 1)
        s = natten1dqkrpb(q, kk, rpb, k, d)
        a = torch.softmax(s, dim=-1)
        cases[f"na1d_{T}_{d}"] = dict(q=q, k=kk, v=v, rpb=rpb, scores=s, out=natten1dav(a, v, k, d))
    for H, W in [(5, 5), (5, 17), (8, 23), (5, 100)]:
        q, kk, v, rpb = r(2, heads, H, W, dim), r(2, heads, H, W, dim), r(2, heads, H, W, dim), r(heads, 2 * k - 1, 2 * k - 1)
        s = natten2dqkrpb(q, kk, rpb, k, 1)
        a = torch.softmax(s, dim=-1)
        cases[f"na2d_{H}_{W}_1"] = dict(q=q, k=kk, v=v, rpb=rpb, scores=s, out=natten2dav(a, v, k, 1))
    np.savez(out / "na.npz", **{f"{c}__{n}": t.numpy() for c, ts in cases.items() for n, t in ts.items()})
    print(f"{len(cases)} cases")


if __name__ == "__main__":
    {"song": song, "na": na}[sys.argv[1]](*sys.argv[2:])
