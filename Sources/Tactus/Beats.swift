import Foundation

/// Beat and downbeat tracking: a port of madmom's `DBNDownBeatTrackingProcessor` (Böck, Krebs and Widmer, ISMIR 2016) with the
/// arguments `allin1` uses: bars of 3 or 4 beats, 55-215 BPM in 60 log-spaced tempo states, transition lambda 100, observation
/// lambda 16, 100 fps. A hidden Markov model per bar length, decoded with Viterbi; the one with the highest log probability wins.
/// Derived from madmom (BSD, see NOTICE).
enum DownbeatTracker {
    static let minBPM = 55.0, maxBPM = 215.0, tempi = 60, transitionLambda = 100.0, observationLambda: Float = 16, fps = 100.0

    /// `beat` and `downbeat` are the two activation columns (probabilities per frame).
    /// Returns the time in seconds of each beat and its position in the bar (1 is the downbeat).
    static func track(beat: [Float], downbeat: [Float], threshold: Double, beatsPerBar: [Int] = [3, 4]) -> (times: [Double], positions: [Int]) {
        // Keep only the span from the first to the last frame where either column reaches the threshold (madmom's `threshold_activations`).
        let limit = Float(threshold)  // numpy compares the float32 activations against the threshold as float32
        let hits = beat.indices.filter { beat[$0] >= limit || downbeat[$0] >= limit }
        // numpy's `idx.any()` is false when the only hit is frame 0
        guard let first = hits.first, let last = hits.last, last > 0 else { return ([], []) }
        let b = Array(beat[first ... last]), d = Array(downbeat[first ... last])
        guard b.contains(where: { $0 != 0 }) || d.contains(where: { $0 != 0 }) else { return ([], []) }

        // Log densities per frame: no beat, beat, downbeat.
        let densities: [[Double]] = zip(b, d).map { x, y in
            [Double(log((1 - (x + y)) / (observationLambda - 1))), Double(log(x)), Double(log(y))]
        }

        var best: (hmm: BarHMM, path: [Int], logProbability: Double)?
        for beats in beatsPerBar {
            let hmm = BarHMM(beats: beats)
            let (path, logProbability) = hmm.viterbi(densities)
            if best == nil || logProbability > best!.logProbability { best = (hmm, path, logProbability) }
        }
        guard let (hmm, path, _) = best, !path.isEmpty else { return ([], []) }

        // madmom's `correct`: within each run of frames the path spends in a (down)beat range, take the frame with the highest activation.
        var times: [Double] = [], positions: [Int] = []
        var run: Int?
        func close(_ left: Int, _ right: Int) {
            var peak = left, peakValue = -Float.infinity  // np.argmax over the flattened (frame, column) rows: the first maximum wins
            for t in left ..< right {
                if b[t] > peakValue { (peak, peakValue) = (t, b[t]) }
                if d[t] > peakValue { (peak, peakValue) = (t, d[t]) }
            }
            times.append(Double(peak + first) / fps)
            positions.append(Int(hmm.position[path[peak]]) + 1)
        }
        for (t, state) in path.enumerated() {
            let inBeat = hmm.pointer[state] >= 1
            if inBeat, run == nil { run = t }
            if !inBeat, let left = run { close(left, t); run = nil }
        }
        if let left = run { close(left, path.count) }
        return (times, positions)
    }
}

/// madmom's `BarStateSpace`, `BarTransitionModel` and `RNNDownBeatTrackingObservationModel` for one bar length.
/// States are laid out beat by beat, tempo by tempo; a tempo with beat interval `n` frames has `n` states, one per position in the beat.
struct BarHMM {
    let numStates: Int
    let position: [Double]  // position in the bar, 0 ..< beats
    let pointer: [UInt8]  // column of the log densities of each state: 0 no beat, 1 beat, 2 downbeat
    let firstIndex: [Int32]  // index into `first` of the states that start a beat, -1 for the others
    /// States that start a beat, and for each the states it can be entered from with their log probability.
    /// Every other state is entered with probability 1 from the previous state, so Viterbi stores back pointers only for these.
    let first: [Int]
    let entries: [[(from: Int, logProbability: Double)]]

    init(beats: Int) {
        let intervals = Self.intervals(min: 60 * DownbeatTracker.fps / DownbeatTracker.maxBPM,
                                       max: 60 * DownbeatTracker.fps / DownbeatTracker.minBPM, count: DownbeatTracker.tempi)
        let perBeat = intervals.reduce(0, +)
        numStates = beats * perBeat
        var position: [Double] = [], firstStates: [[Int]] = [], lastStates: [[Int]] = []
        for beat in 0..<beats {
            var firsts: [Int] = [], lasts: [Int] = []
            for n in intervals {
                firsts.append(position.count)
                let step = 1 / Double(n)  // numpy's linspace(0, 1, n, endpoint=False) multiplies by the step
                for k in 0..<n { position.append(Double(k) * step + Double(beat)) }
                lasts.append(position.count - 1)
            }
            firstStates.append(firsts)
            lastStates.append(lasts)
        }
        self.position = position
        let border = 1 / Double(DownbeatTracker.observationLambda)
        pointer = position.map { $0 < border ? 2 : $0.truncatingRemainder(dividingBy: 1) < border ? 1 : 0 }

        // Tempo changes only happen at the start of a beat: exponential in the ratio of the intervals, rows normalized.
        first = firstStates.flatMap { $0 }
        var index = [Int32](repeating: -1, count: numStates)
        for (i, s) in first.enumerated() { index[s] = Int32(i) }
        firstIndex = index
        var entries: [[(from: Int, logProbability: Double)]] = []
        for beat in 0..<beats {
            let from = lastStates[(beat + beats - 1) % beats]
            let probability: [[Double]] = intervals.map { fromInterval in
                var row = intervals.map { toInterval -> Double in
                    let p = exp(-DownbeatTracker.transitionLambda * abs(Double(toInterval) / Double(fromInterval) - 1))
                    return p <= Double.ulpOfOne ? 0 : p  // madmom's threshold is np.spacing(1)
                }
                let sum = row.reduce(0, +)
                row = row.map { $0 / sum }
                return row
            }
            for j in intervals.indices {
                entries.append(intervals.indices.compactMap { i in probability[i][j] > 0 ? (from[i], log(probability[i][j])) : nil })
            }
        }
        self.entries = entries
    }

    /// madmom's `BeatStateSpace`: the beat intervals (in frames) to model. Linear if they fit in `count` states, else log-spaced.
    static func intervals(min: Double, max: Double, count: Int) -> [Int] {
        let linear = Array(Int(min.rounded(.toNearestOrEven)) ... Int(max.rounded(.toNearestOrEven)))
        guard count < linear.count else { return linear }
        var n = count, intervals: [Int] = []
        let (lo, hi) = (log2(min), log2(max))
        while intervals.count < count {  // quantizing to integers merges neighbours, so ask for more points until enough remain
            let step = (hi - lo) / Double(n - 1)
            let points = (0..<n).map { $0 == n - 1 ? hi : Double($0) * step + lo }
            intervals = Array(Set(points.map { Int(pow(2, $0).rounded(.toNearestOrEven)) })).sorted()
            n += 1
        }
        return intervals
    }

    /// Viterbi over the log densities (one row of 3 per frame). Returns the best path (empty if none has finite probability) and its log probability.
    /// The initial distribution is uniform, which matters: it is what makes the 3- and 4-beat models comparable.
    func viterbi(_ densities: [[Double]]) -> (path: [Int], logProbability: Double) {
        var previous = [Double](repeating: log(1 / Double(numStates)), count: numStates)
        var current = previous
        var back = [UInt32](repeating: 0, count: densities.count * first.count)
        for (frame, d) in densities.enumerated() {
            for s in 1..<numStates { current[s] = previous[s - 1] + d[Int(pointer[s])] }
            for (i, s) in first.enumerated() {
                current[s] = -.infinity
                for (from, logProbability) in entries[i] {
                    let value = previous[from] + logProbability + d[Int(pointer[s])]
                    if value > current[s] { current[s] = value; back[frame * first.count + i] = UInt32(from) }
                }
            }
            swap(&previous, &current)
        }
        var state = 0
        for s in 1..<numStates where previous[s] > previous[state] { state = s }
        let logProbability = previous[state]
        guard logProbability.isFinite else { return ([], logProbability) }
        var path = [Int](repeating: 0, count: densities.count)
        for frame in stride(from: densities.count - 1, through: 0, by: -1) {
            path[frame] = state
            let i = firstIndex[state]
            state = i < 0 ? state - 1 : Int(back[frame * first.count + Int(i)])
        }
        return (path, logProbability)
    }
}
