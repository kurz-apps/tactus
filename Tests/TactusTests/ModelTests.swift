import Foundation
import MLX
import Testing

@testable import Tactus

/// Compares the logits with the original allin1 + NATTEN (fixtures/ref/<song>, from tools/reference.py), per fold and for the ensemble.
@Suite struct ModelTests {
    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    @Test func matchesAllin1() throws {
        let ref = Self.root.appendingPathComponent("fixtures/ref/drumshow90")
        let model = try AllInOne(weights: Self.root.appendingPathComponent("weights/allin1.safetensors"))
        let spec = try loadArrays(url: ref.appendingPathComponent("spec.safetensors"))["array"]!
        let expected = try loadArrays(url: ref.appendingPathComponent("logits.safetensors"))

        func check(_ name: String, _ got: Logits, prefix: String) {
            for (key, value) in [("beat", got.beat), ("downbeat", got.downbeat), ("section", got.section), ("function", got.function)] {
                let want = expected["\(prefix)_\(key)"]!
                let err = abs(value - want).max().item(Float.self)
                print("\(name) \(key): max error \(err)")
                #expect(value.shape == want.shape, "\(name) \(key): shape \(value.shape) vs \(want.shape)")
                #expect(err < 1e-3, "\(name) \(key): max error \(err)")
            }
        }
        let start = Date()
        for fold in 0..<model.folds { check("fold\(fold)", model.logits(spec, fold: fold), prefix: "harmonix-fold\(fold)") }
        check("ensemble", model.ensemble(spec), prefix: "ensemble")
        print("8 folds + ensemble (16 forward passes): \(Date().timeIntervalSince(start)) s")
    }
}
