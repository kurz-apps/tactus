import Foundation
import MLX
import Testing

@testable import Tactus

/// Compares beats, downbeats, tempo and sections with allin1's own result (fixtures/ref/<song>/result.json, from tools/reference.py).
@Suite struct PostprocessingTests {
    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let ref = root.appendingPathComponent("fixtures/ref/drumshow90")
    static let frame = 0.01

    static func expected() throws -> Analysis {
        try JSONDecoder().decode(Analysis.self, from: Data(contentsOf: ref.appendingPathComponent("result.json")))
    }

    /// Same beats (within `tolerance` seconds), same positions in the bar, same sections.
    static func check(_ got: Analysis, _ want: Analysis, tolerance: Double) {
        #expect(got.bpm == want.bpm)
        #expect(got.beats.count == want.beats.count, "beats: \(got.beats.count) vs \(want.beats.count)")
        #expect(got.beatPositions == want.beatPositions)
        #expect(got.downbeats.count == want.downbeats.count)
        let late = zip(got.beats, want.beats).map { abs($0 - $1) }.max() ?? 0
        print("beats: max difference \(late) s")
        #expect(late <= tolerance)
        #expect(got.segments.map(\.label) == want.segments.map(\.label), "labels: \(got.segments.map(\.label)) vs \(want.segments.map(\.label))")
        let off = zip(got.segments, want.segments).map { max(abs($0.start - $1.start), abs($0.end - $1.end)) }.max() ?? 0
        print("segments: max difference \(off) s")
        #expect(off <= tolerance)
    }

    /// madmom's DBN on the exact input it received in the reference run: nothing but the port is being tested.
    @Test func trackerMatchesMadmom() throws {
        let input = try loadArray(url: Self.ref.appendingPathComponent("dbn_input.npy"))
        let (beat, downbeat) = (input[0..., 0].asArray(Float.self), input[0..., 1].asArray(Float.self))
        let want = try Self.expected()
        let start = Date()
        let (times, positions) = DownbeatTracker.track(beat: beat, downbeat: downbeat, threshold: 0.24)
        print("DBN on \(beat.count) frames: \(Date().timeIntervalSince(start)) s")
        #expect(positions == want.beatPositions)
        #expect(times == want.beats, "first differences: \(zip(times, want.beats).filter { $0 != $1 }.prefix(5))")
    }

    /// Everything after the network, from allin1's own ensemble logits.
    @Test func postprocessingMatchesAllin1() throws {
        let logits = try loadArrays(url: Self.ref.appendingPathComponent("logits.safetensors"))
        let got = Analysis(Logits(beat: logits["ensemble_beat"]!, downbeat: logits["ensemble_downbeat"]!,
                                  section: logits["ensemble_section"]!, function: logits["ensemble_function"]!), downbeatThreshold: 0.24)
        Self.check(got, try Self.expected(), tolerance: 1e-9)
    }

    /// Stems on disk to JSON: the spectrogram and the network are ours too, so one frame of slack.
    @Test func pipelineMatchesAllin1() throws {
        let analyzer = try Analyzer(weights: Self.root.appendingPathComponent("weights"))
        let got = try analyzer.analyze(stems: Self.root.appendingPathComponent("fixtures/drumshow90")) { print("\($0): \($1) s") }
        Self.check(got, try Self.expected(), tolerance: Self.frame + 1e-9)
    }
}
