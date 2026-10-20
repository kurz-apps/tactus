import Foundation

/// Section boundaries and labels, following `allin1/postprocessing/functional.py`: local maxima of the section activation,
/// peak picking over a ±12 s window, and for each segment the function label with the highest mean probability.
enum Sections {
    static let labels = ["start", "end", "intro", "outro", "break", "bridge", "inst", "solo", "verse", "chorus"]
    static let fps = 100, hop = 441, sampleRate = 44100.0
    static let filter = 4 * 24 + 1  // 4 * min_hops_per_beat + 1
    static let window = 12 * fps

    /// `section`: one logit per frame. `function`: `labels.count` logits per frame, label-major (`function[label * frames + t]`).
    static func segments(section: [Float], function: [Float]) -> [Segment] {
        let frames = section.count
        let boundary = boundaries(activation: localMaxima(section.map { 1 / (1 + exp(-$0)) }))

        let duration = Double(frames * hop) / sampleRate
        var times = boundary.map { Double($0 * hop) / sampleRate }
        if times.first != 0 { times.insert(0, at: 0) }
        if times.last != duration { times.append(duration) }

        // softmax over the labels at every frame
        var probability = function
        for t in 0..<frames {
            let top = (0..<labels.count).map { function[$0 * frames + t] }.max()!
            let sum = (0..<labels.count).reduce(Float(0)) { $0 + exp(function[$1 * frames + t] - top) }
            for l in 0..<labels.count { probability[l * frames + t] = exp(function[l * frames + t] - top) / sum }
        }
        let cuts = [0] + boundary.filter { $0 > 0 } + [frames]
        return (0..<times.count - 1).map { i in
            let mean = (0..<labels.count).map { l in (cuts[i] ..< cuts[i + 1]).reduce(0.0) { $0 + Double(probability[l * frames + $1]) } }
            var best = 0
            for l in 1..<labels.count where mean[l] > mean[best] { best = l }  // the first maximum wins
            return Segment(start: times[i], end: times[i + 1], label: labels[best])
        }
    }

    /// Keeps the value where it is the maximum of the surrounding `filter` frames, zero elsewhere (plateaus keep all their frames).
    static func localMaxima(_ x: [Float]) -> [Float] {
        let half = filter / 2
        return x.indices.map { i in
            let top = x[max(0, i - half) ... min(x.count - 1, i + half)].max()!
            return x[i] == top ? x[i] : 0
        }
    }

    /// Frames that are the maximum of the ±`window` frames around them and rise above the mean of that neighbourhood
    /// (`peak_picking`: strength = activation - (mean of the past window + mean of the future window) / 2, zeros outside the signal).
    static func boundaries(activation x: [Float]) -> [Int] {
        let peaks = x.indices.filter { x[$0] > 0 }  // only these can be candidates, and the zeros do not move any sum
        return peaks.filter { i in
            let near = peaks.filter { abs($0 - i) <= window }
            guard x[i] == near.map({ x[$0] }).max()! else { return false }
            let past = near.filter { $0 < i }.reduce(0.0) { $0 + Double(x[$1]) } / Double(window)
            let future = near.filter { $0 > i }.reduce(0.0) { $0 + Double(x[$1]) } / Double(window)
            return Double(x[i]) - (past + future) / 2 > 0
        }
    }
}
