import AVFoundation

public enum TactusError: Error, CustomStringConvertible {
    case unsupported(String)

    public var description: String {
        switch self { case .unsupported(let message): return message }
    }
}

/// A stem down-mixed to mono the way madmom's `Signal(path, num_channels=1)` does it.
/// WAV, FLAC, MP3, M4A: anything AVFoundation decodes, from the content (the extension is not used). Other sample rates are resampled to 44.1 kHz.
public struct Stem {
    public var samples: [Float]
    /// 16-bit PCM: madmom keeps integers (the mono mix is truncated toward zero) and folds 1/32767 into the window.
    public var integer: Bool

    public init(contentsOf url: URL) throws {
        guard let file = try? AVAudioFile(forReading: url) else { throw TactusError.unsupported("\(url.lastPathComponent): cannot be read as audio") }
        // 32-bit float WAV stays float; everything else is rounded to 16 bits, which is what madmom gets from ffmpeg for compressed files.
        let integer = file.fileFormat.commonFormat != .pcmFormatFloat32
        self.integer = integer
        var buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        do { try file.read(into: buffer) } catch { throw TactusError.unsupported("\(url.lastPathComponent): cannot be decoded") }
        if buffer.format.sampleRate != 44100 {
            let source = buffer, target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 44100, channels: source.format.channelCount, interleaved: false)!
            buffer = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: AVAudioFrameCount(Double(source.frameLength) * 44100 / source.format.sampleRate) + 1024)!
            var fed = false, failure: NSError?
            _ = AVAudioConverter(from: source.format, to: target)!.convert(to: buffer, error: &failure) { _, status in
                defer { fed = true }
                status.pointee = fed ? .endOfStream : .haveData
                return fed ? nil : source
            }
            if failure != nil { throw TactusError.unsupported("\(url.lastPathComponent): cannot be resampled to 44100 Hz") }
        }
        let (n, channels) = (Int(buffer.frameLength), Int(buffer.format.channelCount))
        let data = buffer.floatChannelData!  // deinterleaved Float32; 16-bit values arrive divided by 32768, so rounding recovers them
        samples = (0..<n).map { i in
            if integer {
                let sum = (0..<channels).reduce(Int32(0)) { $0 + Int32(min(max((data[$1][i] * 32768).rounded(), -32768), 32767)) }  // decoders can overshoot full scale
                return Float(sum / Int32(channels))  // integer division truncates toward zero, like numpy's astype(int16)
            }
            return (0..<channels).reduce(Float(0)) { $0 + data[$1][i] } / Float(channels)
        }
    }
}
