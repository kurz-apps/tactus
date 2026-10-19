import AVFoundation

public enum TactusError: Error, CustomStringConvertible {
    case unsupported(String)

    public var description: String {
        switch self { case .unsupported(let message): return message }
    }
}

/// A stem down-mixed to mono the way madmom's `Signal(path, num_channels=1)` does it.
/// 44.1 kHz WAV (or anything AVFoundation decodes to 16-bit integer or float samples).
public struct Stem {
    public var samples: [Float]
    /// 16-bit PCM: madmom keeps integers (the mono mix is truncated toward zero) and folds 1/32767 into the window.
    public var integer: Bool

    public init(contentsOf url: URL) throws {
        let file = try AVAudioFile(forReading: url)
        let format = file.fileFormat
        guard format.sampleRate == 44100 else { throw TactusError.unsupported("\(url.lastPathComponent): \(format.sampleRate) Hz, expected 44100") }
        let integer = format.commonFormat == .pcmFormatInt16
        self.integer = integer
        guard integer || format.commonFormat == .pcmFormatFloat32 else {
            throw TactusError.unsupported("\(url.lastPathComponent): only 16-bit integer or 32-bit float audio is supported")
        }
        let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: buffer)
        let (n, channels) = (Int(buffer.frameLength), Int(buffer.format.channelCount))
        let data = buffer.floatChannelData!  // deinterleaved Float32; 16-bit values arrive divided by 32768, so rounding recovers them
        samples = (0..<n).map { i in
            if integer {
                let sum = (0..<channels).reduce(Int32(0)) { $0 + Int32((data[$1][i] * 32768).rounded()) }
                return Float(sum / Int32(channels))  // integer division truncates toward zero, like numpy's astype(int16)
            }
            return (0..<channels).reduce(Float(0)) { $0 + data[$1][i] } / Float(channels)
        }
    }
}
