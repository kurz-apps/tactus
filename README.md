# tactus

Beat, downbeat, tempo and section analysis in Swift, running on the Mac GPU (MLX / Metal).

*Tactus* is the music-theory term for the pulse a listener taps along to, which is what this models.

It implements the model from **All-In-One Metrical And Functional Structure Analysis With Neighborhood Attentions on Demixed Audio** (Taejun Kim and Juhan Nam, WASPAA 2023, [arXiv 2307.16425](https://arxiv.org/abs/2307.16425)), using the authors' pretrained weights ([`taejunkim/allinone`](https://huggingface.co/taejunkim/allinone) on Hugging Face, MIT) and the reference code [`mir-aidj/all-in-one`](https://github.com/mir-aidj/all-in-one) (MIT). It is not a different model: the goal is to reproduce the output of `allin1`.

> Status: plan only. Nothing is implemented yet.

## Why

`allin1` relies on neighborhood attention (NATTEN), which ships CPU and CUDA kernels but nothing for Metal. On a Mac it therefore runs on the CPU only. The rest of the model is small (24 embedding dims, 11 blocks, 8 folds), and NATTEN is used through just four functions (`natten{1,2}d{qkrpb,av}`), which can be written with `gather` + `softmax`.

## Scope

- **In:** the four Demucs stems (bass, drums, other, vocals) → JSON with `bpm`, `beats`, `downbeats`, `beat_positions`, `segments`, in the same format as `allin1`.
- **Out:** source separation itself. Run Demucs separately (it already works on Metal) and pass the stems in.

## The paper's pipeline, and how each step is ported

| Step | Reference (Python) | Here |
|---|---|---|
| Per-stem spectrogram: 2048-sample frames at 100 fps (hop 441 at 44.1 kHz), STFT, logarithmic filterbank with 12 bands per octave, 30–17000 Hz, normalized, `log(1+x)` → 81 bins | madmom | Swift + Accelerate (vDSP) |
| Model: three 2D convolutions with max-pooling over frequency only; 11 blocks, each with dilated double 1D neighborhood attention over time (dilation 1, 2, 4, …, 1024; kernel 5) and 2D neighborhood attention across the 4 instruments; beat, downbeat, section and function heads | PyTorch + NATTEN | MLX Swift (Metal) |
| Ensemble of the 8 folds (mean of the logits) | `Ensemble` | same |
| Beats and downbeats: DBN (Böck et al.), `beats_per_bar` [3, 4], 100 fps, 55–240 BPM | madmom `DBNDownBeatTrackingProcessor` | Swift port (CPU) |
| Sections and labels (start, end, intro, outro, break, bridge, inst, solo, verse, chorus) | `postprocessing/functional.py` | Swift port (CPU) |

### Neighborhood attention without NATTEN

For each position `i`, compute the indices of its `k` neighboring keys and the index of the relative position bias (`rpb`), using NATTEN's border rules: the window slides to stay inside the sequence, and with dilation `d` the sequence is treated as `d` interleaved subsequences. Then: gather keys and values → `q·k + rpb` → `softmax` → weighted sum. Memory is `O(T·k)`, never `O(T²)` (a 4-minute song is 24,000 frames at 100 fps). The 2D case does the same over `5×5` windows; the 4 instruments are zero-padded to 5, as in the original.

## Plan

Each step is done only when its check passes.

0. **Reference.** Run the original `allin1` (with `natten`) on 2–3 songs and keep, under `fixtures/`: the 4 stems, the spectrogram (`.npy`), the logits (`--activ`) and the final JSON.
1. **Weights.** `tools/convert.py` (the only Python file): the 8 `.pth` checkpoints → `safetensors` + `config.json`. Check: reloading in PyTorch reproduces the reference logits.
2. **Spectrogram.** Check: max error < 1e-4 against madmom's `.npy`.
3. **Model.** First a single block (1D and 2D attention) against NATTEN, over varied lengths, dilations and borders; then the whole network and the ensemble. Check: logits within 1e-3 of the reference, in `float32`.
4. **Post-processing.** Check: same beats (±1 frame) and same sections as the reference JSON.
5. **Service.** Local HTTP (`POST /analyze` with the stems → JSON).
6. **Measure.** Time per stage on Metal vs CPU, on a real song.

## Risks

- Porting madmom's DBN is the largest piece (a tempo × bar-position state space, Viterbi decoding). Keep madmom's BSD license notice when porting.
- NATTEN's border rules have subtle details; the step 3 block test exists to catch them.
- MLX Swift's Metal shaders are not built by `swift build`; use `xcodebuild`.

## License

To be decided. The upstream `allin1` code and weights are MIT.
