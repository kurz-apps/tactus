import AVFoundation
import Testing

@testable import Tactus

/// Writes one second of a 440 Hz tone in each format and reads it back as a `Stem`.
@Suite struct AudioTests {
    @Test(arguments: [(kAudioFormatFLAC, "flac", 44100.0), (kAudioFormatMPEG4AAC, "m4a", 48000.0)])
    func decodesCompressedFormats(format: AudioFormatID, ext: String, rate: Double) throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).\(ext)")
        defer { try? FileManager.default.removeItem(at: url) }
        let tone = (0..<Int(rate)).map { Float(0.5 * sin(2 * Double.pi * 440 * Double($0) / rate)) }
        do {  // closed when it goes out of scope, which finishes the file
            let file = try AVAudioFile(forWriting: url, settings: [AVFormatIDKey: format, AVSampleRateKey: rate, AVNumberOfChannelsKey: 2, AVLinearPCMBitDepthKey: 16])
            let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(tone.count))!
            buffer.frameLength = buffer.frameCapacity
            for channel in 0..<2 { tone.withUnsafeBufferPointer { buffer.floatChannelData![channel].update(from: $0.baseAddress!, count: tone.count) } }
            try file.write(from: buffer)
        }

        let stem = try Stem(contentsOf: url)
        let crossings = zip(stem.samples, stem.samples.dropFirst()).filter { ($0 < 0) != ($1 < 0) }.count
        let peak = stem.samples.map(abs).max()!
        print("\(ext) at \(rate) Hz: \(stem.samples.count) samples, peak \(peak), \(crossings) zero crossings")
        #expect(stem.integer)
        #expect(abs(stem.samples.count - 44100) < 2200)  // lossy encoders add padding
        #expect(abs(peak - 16384) < 1700)
        #expect(abs(crossings - 880) < 45)
        if ext == "flac" { #expect(zip(stem.samples, tone).map { abs($0 - $1 * 32768) }.max()! <= 1) }  // lossless: the 16-bit tone, give or take the rounding
    }
}
