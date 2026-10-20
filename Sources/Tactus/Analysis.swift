import Foundation
import MLX

public struct Segment: Codable, Equatable {
    public var start: Double
    public var end: Double
    public var label: String
}

/// The result for one song, in the same shape as `allin1`'s JSON.
public struct Analysis: Codable, Equatable {
    /// Most frequent rounded inter-beat tempo; nil with fewer than two beats.
    public var bpm: Int?
    public var beats: [Double]
    public var downbeats: [Double]
    public var beatPositions: [Int]
    public var segments: [Segment]

    enum CodingKeys: String, CodingKey {
        case bpm, beats, downbeats, segments
        case beatPositions = "beat_positions"
    }

    /// `downbeatThreshold`: the mean of the folds' `best_threshold_downbeat`.
    public init(_ logits: Logits, downbeatThreshold: Double) {
        let beat = logits.beat.asArray(Float.self), downbeat = logits.downbeat.asArray(Float.self)

        // The three activations madmom's DBN works from: beat but not downbeat, downbeat, nothing; normalized per frame.
        func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }
        var activation: [(Float, Float)] = []
        for (b, d) in zip(beat.map(sigmoid), downbeat.map(sigmoid)) {
            let combined = (max(1e-8, b - d), d, ((1 - b) + (1 - d)) / 2)
            let sum = combined.0 + combined.1 + combined.2
            activation.append((combined.0 / sum, combined.1 / sum))
        }
        let (times, positions) = DownbeatTracker.track(beat: activation.map(\.0), downbeat: activation.map(\.1), threshold: downbeatThreshold)
        beats = times
        beatPositions = positions
        downbeats = zip(times, positions).filter { $0.1 == 1 }.map(\.0)
        bpm = Self.tempo(beats: times)
        segments = Sections.segments(section: logits.section.asArray(Float.self), function: logits.function.asArray(Float.self))
    }

    /// `allin1`'s `estimate_tempo_from_beats`: the most frequent rounded BPM of the inter-beat intervals.
    static func tempo(beats: [Double]) -> Int? {
        guard beats.count >= 2 else { return nil }
        var count: [Int: Int] = [:]
        for (a, b) in zip(beats, beats.dropFirst()) { count[Int((60 / (b - a)).rounded(.toNearestOrEven)), default: 0] += 1 }
        // ponytail: ties go to the faster tempo; numpy's argsort leaves them to its sorting algorithm
        return count.max { ($0.value, $0.key) < ($1.value, $1.key) }!.key
    }
}

/// The whole pipeline: stems on disk to `Analysis`. `weights` is the directory written by tools/convert.py
/// (`allin1.safetensors` and `config.json`).
public struct Analyzer {
    let model: AllInOne
    let downbeatThreshold: Double

    public init(weights directory: URL) throws {
        model = try AllInOne(weights: directory.appendingPathComponent("allin1.safetensors"))
        struct Config: Decodable { var folds: [Fold]; struct Fold: Decodable { var best_threshold_downbeat: Double } }
        let config = try JSONDecoder().decode(Config.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        downbeatThreshold = config.folds.map(\.best_threshold_downbeat).reduce(0, +) / Double(config.folds.count)
    }

    /// `directory` holds `bass`, `drums`, `other` and `vocals` as .wav, .mp3, .m4a or .flac. `stage` is told how long each stage took.
    public func analyze(stems directory: URL, stage: (String, TimeInterval) -> Void = { _, _ in }) throws -> Analysis {
        func timed<T>(_ name: String, _ work: () throws -> T) rethrows -> T {
            let start = Date()
            defer { stage(name, Date().timeIntervalSince(start)) }
            return try work()
        }
        let spec = try timed("spectrogram") { try Spectrogram.stems(in: directory) }
        let logits = timed("network") { () -> Logits in
            let logits = model.ensemble(spec)
            eval(logits.beat, logits.downbeat, logits.section, logits.function)
            return logits
        }
        return timed("post-processing") { Analysis(logits, downbeatThreshold: downbeatThreshold) }
    }
}
