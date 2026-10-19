# Plan

Each step is done only when its check passes. Architecture is described in the [README](README.md).

## Steps

0. **Reference** — done. `tools/reference.py` runs the original `allin1` + NATTEN and writes, under `fixtures/` (git-ignored):
   - `na/na.npz`: random inputs and NATTEN's outputs for the four attention ops (1D lengths 5–5200 with dilations 1–1024, including lengths that are not a multiple of the dilation; 2D windows from 5×5 to 5×100).
   - `ref/<song>/`: `spec.npy` (madmom spectrogram), `logits.npz` (per fold and ensemble), `dbn_input.npy`, `result.json`.
   - First song: a 90 s excerpt of a 4-stem separation (9000 frames, 123 beats, 5 segments). The whole reference script took **4 min 47 s on CPU** (8 folds, in a Linux container; includes loading, spectrogram and post-processing).
1. **Weights** — done. `tools/convert.py`: the 8 `.pth` checkpoints → one `safetensors` (3928 tensors, 2.4 M parameters, 10 MB) + `config.json` with the per-fold thresholds. Check: every tensor read back equals the checkpoint's.
2. **Spectrogram** — next. Swift + Accelerate. Check: max error < 1e-4 against `spec.npy`. Until then the model tests read `spec.npy` directly.
3. **Model** — done. `NeighborhoodAttention.swift` (the four ops) and `Model.swift` (embedding, 11 blocks, heads, ensemble), MLX on the Metal GPU, `float32`.
   - Ops vs NATTEN, 16 cases: scores identical, outputs within 1.2e-6. Breaking the border rule on purpose makes the lengths that are not a multiple of the dilation fail, so the test does catch it.
   - Network vs `logits.npz`: max error 1.4e-5 over the 8 folds and the ensemble (target 1e-3). 16 forward passes on the 90 s excerpt take 9.9 s.
4. **Post-processing.** DBN and section picking. Check: same beats (±1 frame) and same sections as `result.json`.
5. **Service.** Local HTTP: `POST /analyze` with the stems → JSON.
6. **Measure.** Time per stage on Metal vs the baseline above, on the same excerpt and on a full song.

## Risks

- Porting madmom's DBN is the largest remaining piece (a tempo × bar-position state space, Viterbi decoding). Keep madmom's BSD license notice when porting.
- The 2D layer gathers `5×5` windows (about 290 MB per tensor for a 4-minute song). Fine on this Mac; revisit if memory matters.
- Songs shorter than about 100 s are zero-padded inside the network, as in the reference; the 90 s fixture exercises that path, but a full-length song is still to be checked.

## Running the tests

Needs Xcode with the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`); `swift build` alone does not compile MLX's shaders.

```
python tools/convert.py <dir with harmonix-fold*.pth> weights      # needs torch, safetensors
python tools/pack.py fixtures/na/na.npz fixtures/na/na.safetensors  # and spec.npy / logits.npz of each song
xcodebuild test -scheme tactus -destination 'platform=macOS' -derivedDataPath .build/xcode
```

## Reproducing the reference

```
docker run --rm -v "$PWD:/work" -e HF_HOME=/work/fixtures/hf <image with allin1> \
  /opt/allin1/bin/python /work/tools/reference.py song /work/fixtures/<stems> /work/fixtures/ref/<song>
docker run ... reference.py na /work/fixtures/na
```
