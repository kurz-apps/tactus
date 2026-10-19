import MLX

// Neighborhood attention (Hassani et al.) without NATTEN's kernels: the same four operations as
// natten{1,2}d{qkrpb,av}, written with gather. Index rules ported from NATTEN 0.17.3
// (csrc/include/natten/cpu/naive/natten_cpu_commons.h, MIT, Copyright (c) 2022-2024 Ali Hassani).
// Shapes: 1D q,k,v [B, H, T, D]; 2D q,k,v [B, H, Y, X, D]. All lengths must be >= kernel * dilation.

/// For each position, where its `kernel` neighbors are and which entry of the relative position bias each uses.
struct Neighborhood {
    let keys: MLXArray  // [length, kernel] position of each neighboring key
    let bias: MLXArray  // [length, kernel] index into the 2*kernel-1 relative position biases

    init(length: Int, kernel: Int, dilation: Int) {
        let nh = kernel / 2
        var keys = [Int32](), bias = [Int32]()
        keys.reserveCapacity(length * kernel)
        bias.reserveCapacity(length * kernel)
        for i in 0..<length {
            let start = Self.windowStart(i, length: length, nh: nh, dilation: dilation)
            let pb = Self.biasStart(i, length: length, kernel: kernel, nh: nh, dilation: dilation)
            for k in 0..<kernel {
                keys.append(Int32(start + k * dilation))
                bias.append(Int32(pb + k))
            }
        }
        self.keys = MLXArray(keys, [length, kernel])
        self.bias = MLXArray(bias, [length, kernel])
    }

    static func windowStart(_ index: Int, length: Int, nh: Int, dilation: Int) -> Int {
        let dilationIdx = index % dilation
        let indexPdp = index / dilation
        var lengthPdp = (length + dilation - 1) / dilation
        let numPadded = lengthPdp * dilation - length
        if dilationIdx >= dilation - numPadded { lengthPdp -= 1 }
        var start = max(indexPdp - nh, 0)
        if indexPdp + nh >= lengthPdp { start += lengthPdp - indexPdp - nh - 1 }
        return start * dilation + dilationIdx
    }

    static func biasStart(_ index: Int, length: Int, kernel: Int, nh: Int, dilation: Int) -> Int {
        if dilation <= 1 {
            var pb = nh
            if index < nh { pb += nh - index }
            if index + nh >= length { pb += length - index - 1 - nh }
            return pb
        }
        if index - nh * dilation < 0 { return kernel - 1 - index / dilation }
        if index + nh * dilation >= length { return (length - index - 1) / dilation }
        return nh
    }
}

/// q·k + relative position bias over each neighborhood. q, k: [B, H, T, D]; rpb: [H, 2K-1] -> [B, H, T, K].
public func na1dScores(_ q: MLXArray, _ k: MLXArray, rpb: MLXArray, kernel: Int, dilation: Int) -> MLXArray {
    let n = Neighborhood(length: q.dim(2), kernel: kernel, dilation: dilation)
    let keys = take(k, n.keys, axis: 2)  // [B, H, T, K, D]
    let bias = take(rpb, n.bias, axis: 1)  // [H, T, K]
    return (q.expandedDimensions(axis: 3) * keys).sum(axis: -1) + bias.expandedDimensions(axis: 0)
}

/// Weighted sum of the values of each neighborhood. attn: [B, H, T, K]; v: [B, H, T, D] -> [B, H, T, D].
public func na1dApply(_ attn: MLXArray, _ v: MLXArray, kernel: Int, dilation: Int) -> MLXArray {
    let n = Neighborhood(length: v.dim(2), kernel: kernel, dilation: dilation)
    let values = take(v, n.keys, axis: 2)  // [B, H, T, K, D]
    return (attn.expandedDimensions(axis: -1) * values).sum(axis: 3)
}

/// Neighbors of a 2D position: `kernel` x `kernel` keys, gathered as [B, H, Y, X, K, K, D].
private func gather2d(_ x: MLXArray, _ ny: Neighborhood, _ nx: Neighborhood) -> MLXArray {
    take(take(x, ny.keys, axis: 2), nx.keys, axis: 4)  // [B, H, Y, Ky, X, Kx, D]
        .transposed(0, 1, 2, 4, 3, 5, 6)
}

/// 2D version of `na1dScores`. q, k: [B, H, Y, X, D]; rpb: [H, 2K-1, 2K-1] -> [B, H, Y, X, K*K].
public func na2dScores(_ q: MLXArray, _ k: MLXArray, rpb: MLXArray, kernel: Int, dilation: Int) -> MLXArray {
    let ny = Neighborhood(length: q.dim(2), kernel: kernel, dilation: dilation)
    let nx = Neighborhood(length: q.dim(3), kernel: kernel, dilation: dilation)
    let keys = gather2d(k, ny, nx)
    let bias = take(take(rpb, ny.bias, axis: 1), nx.bias, axis: 3)  // [H, Y, Ky, X, Kx]
        .transposed(0, 1, 3, 2, 4)  // [H, Y, X, Ky, Kx]
    let scores = (q.expandedDimensions(axes: [4, 5]) * keys).sum(axis: -1) + bias.expandedDimensions(axis: 0)
    return scores.reshaped(q.dim(0), q.dim(1), q.dim(2), q.dim(3), kernel * kernel)
}

/// 2D version of `na1dApply`. attn: [B, H, Y, X, K*K]; v: [B, H, Y, X, D] -> [B, H, Y, X, D].
public func na2dApply(_ attn: MLXArray, _ v: MLXArray, kernel: Int, dilation: Int) -> MLXArray {
    let ny = Neighborhood(length: v.dim(2), kernel: kernel, dilation: dilation)
    let nx = Neighborhood(length: v.dim(3), kernel: kernel, dilation: dilation)
    let values = gather2d(v, ny, nx)  // [B, H, Y, X, Ky, Kx, D]
    let a = attn.reshaped(attn.dim(0), attn.dim(1), attn.dim(2), attn.dim(3), kernel, kernel, 1)
    return (a * values).sum(axes: [4, 5])
}
