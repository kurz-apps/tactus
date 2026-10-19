# Plan

Each step is done only when its check passes. Architecture is described in the [README](README.md).

## Steps

0. **Reference** — done. `tools/reference.py` runs the original `allin1` + NATTEN and writes, under `fixtures/` (git-ignored):
   - `na/na.npz`: random inputs and NATTEN's outputs for the four attention ops (1D lengths 5–5200 with dilations 1–1024, including lengths that are not a multiple of the dilation; 2D windows from 5×5 to 5×100).
   - `ref/<song>/`: `spec.npy` (madmom spectrogram), `logits.npz` (per fold and ensemble), `dbn_input.npy`, `result.json`.
   - First song: a 90 s excerpt of a 4-stem separation (9000 frames, 123 beats, 5 segments). **Baseline: 4 min 47 s on CPU** (8 folds, in a Linux container).
1. **Weights.** `tools/convert.py`: the 8 `.pth` checkpoints → `safetensors` + `config.json`. Check: reloading in PyTorch reproduces the reference logits.
2. **Spectrogram.** Swift + Accelerate. Check: max error < 1e-4 against `spec.npy`.
3. **Model.** First the four attention ops against `na.npz`; then one block, the whole network, and the ensemble. Check: logits within 1e-3 of `logits.npz`, in `float32`.
4. **Post-processing.** DBN and section picking. Check: same beats (±1 frame) and same sections as `result.json`.
5. **Service.** Local HTTP: `POST /analyze` with the stems → JSON.
6. **Measure.** Time per stage on Metal vs the baseline above, on the same excerpt and on a full song.

## Risks

- Porting madmom's DBN is the largest piece (a tempo × bar-position state space, Viterbi decoding). Keep madmom's BSD license notice when porting.
- NATTEN's border rules have subtle details; the step 3 op tests exist to catch them.
- MLX Swift's Metal shaders are not built by `swift build`; use `xcodebuild`.
- Not yet checked: that MLX Swift offers max-pooling and gather-style indexing (otherwise they are written by hand).

## Reproducing the reference

```
docker run --rm -v "$PWD:/work" -e HF_HOME=/work/fixtures/hf <image with allin1> \
  /opt/allin1/bin/python /work/tools/reference.py song /work/fixtures/<stems> /work/fixtures/ref/<song>
docker run ... reference.py na /work/fixtures/na
```
