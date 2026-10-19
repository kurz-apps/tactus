import Foundation
import MLX
import MLXFast

/// Output of the network for one song: one logit per frame for beat, downbeat and section, and `labels` classes for function.
public struct Logits {
    public var beat: MLXArray  // [T]
    public var downbeat: MLXArray  // [T]
    public var section: MLXArray  // [T]
    public var function: MLXArray  // [labels, T]
}

/// The All-In-One network (Kim & Nam, WASPAA 2023), inference only, written to follow `allin1/models/allinone.py`
/// and `dinat.py`. All 8 released checkpoints share one configuration, which is fixed here.
public struct AllInOne {
    static let dim = 24, heads = 2, kernel = 5, depth = 11, instruments = 4, eps: Float = 1e-5

    let weights: [String: MLXArray]

    /// `allin1.safetensors` as written by tools/convert.py: tensors named `fold<N>.<state_dict name>`.
    public init(weights url: URL) throws {
        weights = try loadArrays(url: url)
    }

    public var folds: Int { 8 }

    /// Mean of the logits of all folds (the released `harmonix-all` ensemble).
    /// `spec`: [4, T, 81] spectrogram, one per stem (bass, drums, other, vocals).
    public func ensemble(_ spec: MLXArray) -> Logits {
        let all = (0..<folds).map { logits(spec, fold: $0) }
        func mean(_ f: (Logits) -> MLXArray) -> MLXArray { MLX.stacked(all.map(f)).mean(axis: 0) }
        return Logits(beat: mean(\.beat), downbeat: mean(\.downbeat), section: mean(\.section), function: mean(\.function))
    }

    public func logits(_ spec: MLXArray, fold: Int) -> Logits {
        let f = Fold(w: weights, prefix: "fold\(fold).")
        let t = spec.dim(1)
        var x = f.embed(spec.reshaped(Self.instruments, t, 81, 1))  // [K, T, C]
        for i in 0..<Self.depth {
            let pre = "encoder.layers.\(i)."
            x = f.timeLayer(x, pre + "timelayer.", dilation: min(1 << i, 3200))
            x = f.instLayer(x.reshaped(1, Self.instruments, t, Self.dim), pre + "instlayer.")
                .reshaped(Self.instruments, t, Self.dim)
        }
        let h = f.layerNorm(x, "norm").reshaped(1, Self.instruments, t, Self.dim)
            .transposed(0, 2, 1, 3).reshaped(1, t, Self.instruments * Self.dim)  // [N, T, K*C]
        func head(_ name: String) -> MLXArray { f.linear(h, name + ".classifier")[0].T }  // [classes, T]
        return Logits(beat: head("beat_classifier")[0], downbeat: head("downbeat_classifier")[0],
                      section: head("section_classifier")[0], function: head("function_classifier"))
    }
}

struct Fold {
    let w: [String: MLXArray]
    let prefix: String
    typealias M = AllInOne

    func p(_ name: String) -> MLXArray { w[prefix + name]! }

    func linear(_ x: MLXArray, _ name: String) -> MLXArray { matmul(x, p(name + ".weight").T) + p(name + ".bias") }

    func layerNorm(_ x: MLXArray, _ name: String) -> MLXArray {
        MLXFast.layerNorm(x, weight: p(name + ".weight"), bias: p(name + ".bias"), eps: M.eps)
    }

    func elu(_ x: MLXArray) -> MLXArray { MLX.where(x .> 0, x, exp(minimum(x, 0)) - 1) }
    func gelu(_ x: MLXArray) -> MLXArray { x * 0.5 * (1 + erf(x / Float(2).squareRoot())) }

    /// Three convolutions with max-pooling over frequency only. [K, T, 81, 1] -> [K, T, C].
    func embed(_ x: MLXArray) -> MLXArray {
        func conv(_ x: MLXArray, _ name: String, pad: Int) -> MLXArray {  // PyTorch NCHW weights -> MLX NHWC
            conv2d(x, p("embeddings." + name + ".weight").transposed(0, 2, 3, 1), padding: [pad, 0]) + p("embeddings." + name + ".bias")
        }
        func pool(_ x: MLXArray) -> MLXArray {  // kernel = stride = 3 over frequency, remainder dropped
            let n = x.dim(2) / 3
            return x[0..., 0..., 0 ..< (n * 3), 0...].reshaped(x.dim(0), x.dim(1), n, 3, x.dim(3)).max(axis: 3)
        }
        var h = elu(pool(conv(x, "conv0", pad: 1)))
        h = elu(pool(conv(h, "conv1", pad: 0)))
        h = elu(pool(conv(h, "conv2", pad: 1)))
        return layerNorm(h.squeezed(axis: 2), "embeddings.norm")
    }

    /// Neighborhood attention block: q, k, v projections, attention over `dims` spatial axes, output projection.
    /// `x`: [B, (Y,) T, C]; the attention functions are chosen by the number of spatial axes.
    func attention(_ x: MLXArray, _ pre: String, dilation: Int) -> MLXArray {
        let twoD = x.ndim == 4
        let shape = Array(x.shape.dropLast())
        func split(_ name: String) -> MLXArray {
            let y = linear(x, pre + "self." + name).reshaped(shape + [M.heads, M.dim / M.heads])
            return twoD ? y.transposed(0, 3, 1, 2, 4) : y.transposed(0, 2, 1, 3)  // heads before the spatial axes
        }
        let q = split("query") / Float(M.dim / M.heads).squareRoot(), k = split("key"), v = split("value")
        let rpb = p(pre + "self.rpb")
        let probs: MLXArray, out: MLXArray
        if twoD {
            probs = softmax(na2dScores(q, k, rpb: rpb, kernel: M.kernel, dilation: dilation), axis: -1)
            out = na2dApply(probs, v, kernel: M.kernel, dilation: dilation).transposed(0, 2, 3, 1, 4)
        } else {
            probs = softmax(na1dScores(q, k, rpb: rpb, kernel: M.kernel, dilation: dilation), axis: -1)
            out = na1dApply(probs, v, kernel: M.kernel, dilation: dilation).transposed(0, 2, 1, 3)
        }
        return linear(out.reshaped(shape + [M.dim]), pre + "output.dense")
    }

    /// LayerNorm -> MLP with residual; `shortcut` is the residual stream and `x` the (possibly concatenated) input.
    func feedForward(_ x: MLXArray, shortcut: MLXArray, _ pre: String) -> MLXArray {
        shortcut + linear(gelu(linear(layerNorm(x, pre + "layernorm_after"), pre + "intermediate.dense")), pre + "output.dense")
    }

    /// Time layer: two dilated neighborhood attentions (dilation d and 2d) in parallel over each stem. x: [K, T, C].
    func timeLayer(_ x: MLXArray, _ pre: String, dilation: Int) -> MLXArray {
        let t = x.dim(1)
        var h = layerNorm(x, pre + "layernorm_before")
        let window = M.kernel * dilation * 2
        if t < window { h = padded(h, widths: [[0, 0], [0, window - t], [0, 0]]) }
        let a = x + attention(h, pre + "attention.", dilation: dilation)[0..., 0 ..< t, 0...]
        let b = x + attention(h, pre + "attention2.", dilation: dilation * 2)[0..., 0 ..< t, 0...]
        return feedForward(concatenated([a, b], axis: -1), shortcut: (a + b) / 2, pre)
    }

    /// Instrument layer: 2D neighborhood attention across the stems and time. x: [1, K, T, C].
    func instLayer(_ x: MLXArray, _ pre: String) -> MLXArray {
        let (k, t) = (x.dim(1), x.dim(2))
        var h = layerNorm(x, pre + "layernorm_before")
        if k < M.kernel || t < M.kernel {  // zero-padded after the norm, like the reference
            h = padded(h, widths: [[0, 0], [0, max(0, M.kernel - k)], [0, max(0, M.kernel - t)], [0, 0]])
        }
        let a = x + attention(h, pre + "attention.", dilation: 1)[0..., 0 ..< k, 0 ..< t, 0...]
        return feedForward(a, shortcut: a, pre)
    }
}
