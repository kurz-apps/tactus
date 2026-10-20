# Plan

Each step is done only when its check passes. Architecture is described in the [README](README.md).

## Steps

0. **Reference** — done. `tools/reference.py` runs the original `allin1` + NATTEN and writes, under `fixtures/` (git-ignored):
   - `na/na.npz`: random inputs and NATTEN's outputs for the four attention ops (1D lengths 5–5200 with dilations 1–1024, including lengths that are not a multiple of the dilation; 2D windows from 5×5 to 5×100).
   - `ref/<song>/`: `spec.npy` (madmom spectrogram), `logits.npz` (per fold and ensemble), `dbn_input.npy`, `result.json`.
   - First song: a 90 s excerpt of a 4-stem separation (9000 frames, 123 beats, 5 segments). The whole reference script took **4 min 47 s on CPU** (8 folds, in a Linux container; includes loading, spectrogram and post-processing).
1. **Weights** — done. `tools/convert.py`: the 8 `.pth` checkpoints → one `safetensors` (3928 tensors, 2.4 M parameters, 10 MB) + `config.json` with the per-fold thresholds. Check: every tensor read back equals the checkpoint's.
2. **Spectrogram** — done. `Spectrogram.swift` + `Audio.swift`: 16-bit mono mix as madmom does it, Hann STFT, the 81-band filterbank. Max error 4.1e-6 against `spec.npy` (target 1e-4), filterbank 1.5e-8.
3. **Model** — done. `NeighborhoodAttention.swift` (the four ops) and `Model.swift` (embedding, 11 blocks, heads, ensemble), MLX on the Metal GPU, `float32`.
   - Ops vs NATTEN, 16 cases: scores identical, outputs within 1.2e-6. Breaking the border rule on purpose makes the lengths that are not a multiple of the dilation fail, so the test does catch it.
   - Network vs `logits.npz`: max error 1.4e-5 over the 8 folds and the ensemble (target 1e-3). 16 forward passes on the 90 s excerpt take 9.9 s.
4. **Post-processing** — done. `Beats.swift` (madmom's DBN: 3/4-beat bars, 60 tempi, Viterbi; back pointers are stored only at beat starts, since every other transition has probability 1) and `Sections.swift` (local maxima, ±12 s peak picking, mean label probability); tempo and the JSON shape in `Analysis.swift`.
   - From allin1's own ensemble logits: beats, downbeats, bar positions, BPM (82) and the 5 sections are **identical** to `result.json` (max time difference 0.0 s). Through our network and spectrogram too (target ±1 frame).
5. **Service** — done. `tactus-cli serve` (Network.framework, 127.0.0.1 only): `POST /analyze`, `multipart/form-data` with the files `bass`, `drums`, `other`, `vocals` (WAV, 44.1 kHz) → the JSON of `allin1`. One song at a time. Errors are `{"error": ...}`: 400 missing file, 422 unreadable audio, 404 other paths. `tactus-cli analyze <dir>` does the same from the command line.
6. **Measure** — done. Apple M4 Max, 36 GB, Release build, second run on (the first run after a build also compiles the Metal kernels, about 3 s more):

   | | spectrogram | network (8 folds) | post-processing | total | peak memory |
   |---|---|---|---|---|---|
   | 90 s excerpt | 0.08 s | 1.14 s | 0.50 s | **1.7 s** | 3.5 GB |
   | 270 s (the excerpt three times) | 0.19 s | 3.55 s | 1.51 s | **5.3 s** | 4.7 GB |

   The reference took 287 s for the 90 s excerpt (CPU, Linux container), about 170× more. Over HTTP the 90 s excerpt takes 1.8 s including the 64 MB upload. The 270 s input is for time and memory only: it is not a real song and has no reference output.

## Risks

- The 2D layer gathers `5×5` windows. Evaluating one fold at a time keeps the peak at 4.7 GB for 270 s (it was 11.5 GB with the 8 folds in one graph); a 10-minute song would need roughly 10 GB (extrapolated, not measured).
- Songs shorter than about 100 s are zero-padded inside the network, as in the reference; the 90 s fixture exercises that path. A real full-length song has not been compared with `allin1`: that needs another run of `tools/reference.py`.
- Ties in the tempo histogram go to the faster tempo; numpy leaves them to its sorting algorithm.

## Running the tests

Needs Xcode with the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`); `swift build` alone does not compile MLX's shaders.

```
python tools/convert.py <dir with harmonix-fold*.pth> weights      # needs torch, safetensors
python tools/pack.py fixtures/na/na.npz fixtures/na/na.safetensors  # and spec.npy / logits.npz of each song
xcodebuild test -scheme tactus-Package -destination 'platform=macOS' -derivedDataPath .build/xcode SWIFT_OPTIMIZATION_LEVEL=-O
```

`SWIFT_OPTIMIZATION_LEVEL=-O` is optional but the Viterbi search takes about 40 s per song unoptimized (the suite runs in 4 s with it, 46 s without).

## Reproducing the reference

```
docker run --rm -v "$PWD:/work" -e HF_HOME=/work/fixtures/hf <image with allin1> \
  /opt/allin1/bin/python /work/tools/reference.py song /work/fixtures/<stems> /work/fixtures/ref/<song>
docker run ... reference.py na /work/fixtures/na
```
