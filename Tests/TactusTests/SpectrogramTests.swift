import Foundation
import MLX
import Testing

@testable import Tactus

/// Compares the features with madmom's (fixtures/filterbank.safetensors and fixtures/ref/<song>/spec.safetensors, from tools/reference.py).
@Suite struct SpectrogramTests {
    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    @Test func filterbankMatchesMadmom() throws {
        let want = try loadArrays(url: Self.root.appendingPathComponent("fixtures/filterbank.safetensors"))["array"]!
        #expect(Spectrogram.filterbank.shape == want.shape)
        let err = abs(Spectrogram.filterbank - want).max().item(Float.self)
        print("filterbank: max error \(err)")
        #expect(err < 1e-6)
    }

    @Test func spectrogramMatchesMadmom() throws {
        let want = try loadArrays(url: Self.root.appendingPathComponent("fixtures/ref/drumshow90/spec.safetensors"))["array"]!
        let start = Date()
        let got = try Spectrogram.stems(in: Self.root.appendingPathComponent("fixtures/drumshow90"))
        eval(got)
        print("spectrogram of 4 stems: \(Date().timeIntervalSince(start)) s")
        #expect(got.shape == want.shape, "shape \(got.shape) vs \(want.shape)")
        let err = abs(got - want).max().item(Float.self)
        print("spectrogram: max error \(err)")
        #expect(err < 1e-4)
    }
}
