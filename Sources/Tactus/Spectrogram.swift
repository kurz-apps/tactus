import Foundation
import MLX

/// The input features of the All-In-One model, reproducing the madmom chain used by the reference (`allin1/spectrogram.py`):
/// 2048-sample frames at 100 fps, Hann window, STFT magnitude, a normalized logarithmic filterbank with 12 bands per
/// octave from 30 to 17000 Hz, and `log10(1 + x)`. For a song: `[4, T, 81]`, one slice per stem (bass, drums, other, vocals).
public enum Spectrogram {
    static let sampleRate = 44100, frameSize = 2048, hop = 441, fftBins = 1024
    public static let stemNames = ["bass", "drums", "other", "vocals"]

    /// Spectrogram of the four stems `bass.wav`, `drums.wav`, `other.wav` and `vocals.wav` in `directory`.
    public static func stems(in directory: URL) throws -> MLXArray {
        let specs = try stemNames.map { name -> MLXArray in
            let stem = try Stem(contentsOf: directory.appendingPathComponent(name + ".wav"))
            let s = compute(stem.samples, integer: stem.integer)
            eval(s)
            return s
        }
        guard Set(specs.map { $0.dim(0) }).count == 1 else { throw TactusError.unsupported("the four stems must have the same length") }
        return stacked(specs)
    }

    /// `[T, 81]` for one mono signal. `integer` is true when the samples came from 16-bit PCM, which madmom scales through the window.
    public static func compute(_ samples: [Float], integer: Bool) -> MLXArray {
        let frames = (samples.count + hop - 1) / hop  // a frame for every hop in which the signal starts
        let half = frameSize / 2
        let length = (frames - 1) * hop + frameSize  // frame t covers [t*hop - half, t*hop + half) in signal time
        let signal = padded(MLXArray(samples), widths: [[half, length - samples.count - half]])
        let framed = asStrided(signal, [frames, frameSize], strides: [hop, 1])
        let scale = integer ? Double(Int16.max) : 1
        let window = MLXArray((0..<frameSize).map { Float((0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(frameSize - 1))) / scale) })
        let magnitude = abs(rfft(framed * window, axis: 1))[0..., 0 ..< fftBins]  // the Nyquist bin is dropped
        return log10(matmul(magnitude, filterbank) + 1)
    }

    /// madmom's `LogarithmicFilterbank` for 44.1 kHz and 2048-sample frames: `[1024, 81]`.
    static let filterbank: MLXArray = {
        let (bands, fmin, fmax, fref) = (12.0, 30.0, 17000.0, 440.0)
        // np.fft.fftfreq(2048, 1 / 44100)[:1024]
        let step = 1.0 / (Double(frameSize) * (1.0 / Double(sampleRate)))
        let binFrequencies = (0..<fftBins).map { Double($0) * step }

        // log_frequencies: bands per octave around fref, restricted to [fmin, fmax]
        let lo = floor(log2(fmin / fref) * bands), hi = ceil(log2(fmax / fref) * bands)
        let frequencies = stride(from: lo, to: hi, by: 1).map { fref * pow(2.0, $0 / bands) }.filter { $0 >= fmin && $0 <= fmax }

        // frequencies2bins: the nearest FFT bin, duplicates removed
        let bins = Set(frequencies.map { f -> Int in
            var i = binFrequencies.firstIndex { $0 >= f } ?? binFrequencies.count
            i = min(max(i, 1), binFrequencies.count - 1)
            return f - binFrequencies[i - 1] < binFrequencies[i] - f ? i - 1 : i
        }).sorted()

        // One triangular filter per consecutive triple of bins, area-normalized, overlapping neighbours.
        let count = bins.count - 2
        var matrix = [Float](repeating: 0, count: fftBins * count)
        for band in 0..<count {
            var (start, center, stop) = (bins[band], bins[band + 1], bins[band + 2])
            if stop - start < 2 { (center, stop) = (start, start + 1) }
            let (up, down) = (center - start, stop - center)
            var filter = (0..<up).map { Float(Double($0) / Double(up)) } + (0..<down).map { Float(1 - Double($0) / Double(down)) }
            let sum = filter.reduce(0, +)
            filter = filter.map { $0 / sum }
            for (k, v) in filter.enumerated() where start + k < fftBins {
                matrix[(start + k) * count + band] = max(matrix[(start + k) * count + band], v)
            }
        }
        return MLXArray(matrix, [fftBins, count])
    }()
}
