import Foundation
import MLX
import Testing

@testable import Tactus

/// Compares the four operations with NATTEN's own outputs (fixtures/na/na.safetensors, from tools/reference.py).
@Suite struct NeighborhoodAttentionTests {
    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    @Test func matchesNatten() throws {
        let file = Self.root.appendingPathComponent("fixtures/na/na.safetensors")
        let arrays = try loadArrays(url: file)
        let names = Set(arrays.keys.map { String($0.split(separator: "__")[0]) }).sorted()
        #expect(!names.isEmpty)
        for name in names {
            let a = { (key: String) in arrays["\(name)__\(key)"]! }
            let is1d = name.hasPrefix("na1d")
            let dilation = Int(name.split(separator: "_").last!)!
            let scores = is1d
                ? na1dScores(a("q"), a("k"), rpb: a("rpb"), kernel: 5, dilation: dilation)
                : na2dScores(a("q"), a("k"), rpb: a("rpb"), kernel: 5, dilation: dilation)
            let out = is1d
                ? na1dApply(softmax(a("scores"), axis: -1), a("v"), kernel: 5, dilation: dilation)
                : na2dApply(softmax(a("scores"), axis: -1), a("v"), kernel: 5, dilation: dilation)
            let errScores = abs(scores - a("scores")).max().item(Float.self)
            let errOut = abs(out - a("out")).max().item(Float.self)
            #expect(errScores < 1e-4, "\(name) scores: max error \(errScores)")
            #expect(errOut < 1e-4, "\(name) out: max error \(errOut)")
            print("\(name): scores \(errScores), out \(errOut)")
        }
    }
}
