import Foundation
import Network
import Tactus

// tactus-cli analyze <stems dir> [--weights DIR]    JSON on stdout, time per stage on stderr
// tactus-cli serve [--port 8080] [--weights DIR]    POST /analyze, multipart/form-data with the files bass, drums, other, vocals (.wav)

var args = Array(CommandLine.arguments.dropFirst())
func option(_ name: String, default value: String) -> String {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return value }
    defer { args.removeSubrange(i ... i + 1) }
    return args[i + 1]
}
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

let weights = URL(fileURLWithPath: option("--weights", default: "weights"))
let port = option("--port", default: "8080")
let encoder = JSONEncoder()
encoder.outputFormatting = .sortedKeys  // otherwise the key order changes from run to run
let analyzer: Analyzer
do { analyzer = try Analyzer(weights: weights) } catch { fail("weights: \(error)") }

switch args.first {
case "analyze" where args.count == 2:
    do {
        let analysis = try analyzer.analyze(stems: URL(fileURLWithPath: args[1])) { name, seconds in
            FileHandle.standardError.write(Data(String(format: "%@: %.2f s\n", name, seconds).utf8))
        }
        print(String(decoding: try encoder.encode(analysis), as: UTF8.self))
    } catch { fail("\(error)") }
case "serve" where args.count == 1:
    guard let port = NWEndpoint.Port(port) else { fail("invalid port") }
    let parameters = NWParameters.tcp
    parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: port)  // no authentication: this machine only
    guard let listener = try? NWListener(using: parameters) else { fail("cannot listen on port \(port)") }
    listener.newConnectionHandler = { Request($0).start() }
    listener.start(queue: .main)
    FileHandle.standardError.write(Data("listening on http://127.0.0.1:\(port)/analyze\n".utf8))
    dispatchMain()
default:
    fail("usage: tactus-cli analyze <stems dir> [--weights DIR]\n       tactus-cli serve [--port 8080] [--weights DIR]")
}

/// One HTTP request, read to the end of its body and answered once. Runs on the main queue, so songs are analyzed one at a time
/// (analysis blocks the queue; other connections wait). No chunked bodies, no keep-alive.
final class Request {
    static let limit = 2 << 30
    let connection: NWConnection
    var data = Data()
    var body: (start: Int, length: Int, boundary: String)?
    var continued = false

    init(_ connection: NWConnection) { self.connection = connection }

    func start() {
        connection.start(queue: .main)
        receive()
    }

    func receive() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { chunk, _, finished, error in
            if let chunk { self.data.append(chunk) }
            if error != nil || (finished && chunk == nil) { return self.connection.cancel() }
            self.advance()
        }
    }

    func advance() {
        if body == nil {
            guard let end = data.range(of: Data("\r\n\r\n".utf8)) else {
                return data.count > 1 << 16 ? reply(431, "headers too large") : receive()
            }
            let lines = String(decoding: data[..<end.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")
            var headers: [String: String] = [:]
            for line in lines.dropFirst() {
                guard let colon = line.firstIndex(of: ":") else { continue }
                headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            }
            guard lines[0].hasPrefix("POST /analyze ") else { return reply(404, "POST /analyze is the only endpoint") }
            guard headers["transfer-encoding"] == nil else { return reply(501, "chunked bodies are not supported") }
            guard let length = headers["content-length"].flatMap({ Int($0) }) else { return reply(411, "Content-Length required") }
            guard length <= Self.limit else { return reply(413, "body too large") }
            guard let boundary = headers["content-type"]?.components(separatedBy: "boundary=").dropFirst().first?.trimmingCharacters(in: CharacterSet(charactersIn: "\"")) else {
                return reply(415, "multipart/form-data expected")
            }
            body = (end.upperBound, length, boundary)
            if headers["expect"]?.lowercased() == "100-continue", !continued {  // curl waits for this before sending a large body
                continued = true
                connection.send(content: Data("HTTP/1.1 100 Continue\r\n\r\n".utf8), completion: .idempotent)
            }
        }
        if let body, data.count - body.start >= body.length { analyze(data.subdata(in: body.start ..< body.start + body.length), boundary: body.boundary) } else { receive() }
    }

    func analyze(_ body: Data, boundary: String) {
        let files = Self.parts(of: body, boundary: boundary)
        guard let missing = Spectrogram.stemNames.first(where: { files[$0] == nil }) else {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            do {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                for name in Spectrogram.stemNames { try files[name]!.write(to: directory.appendingPathComponent(name + ".wav")) }  // only the known names reach the file system
                reply(200, json: try encoder.encode(analyzer.analyze(stems: directory)))
            } catch let error as TactusError {
                reply(422, "\(error)")
            } catch {
                reply(500, "\(error)")
            }
            return
        }
        reply(400, "missing file: \(missing)")
    }

    /// The named files of a multipart/form-data body.
    static func parts(of body: Data, boundary: String) -> [String: Data] {
        let delimiter = Data("--\(boundary)".utf8), blank = Data("\r\n\r\n".utf8)
        var files: [String: Data] = [:]
        var cursor = body.range(of: delimiter)?.upperBound
        while let start = cursor, let next = body.range(of: delimiter, in: start ..< body.endIndex) {
            let part = body[start ..< next.lowerBound]  // CRLF, headers, blank line, content, CRLF
            if let split = part.range(of: blank) {
                let name = String(decoding: part[..<split.lowerBound], as: UTF8.self)
                    .components(separatedBy: ";").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .first { $0.hasPrefix("name=\"") }.map { String($0.dropFirst(6).dropLast()) }
                if let name { files[name] = Data(part[split.upperBound ..< part.endIndex - 2]) }
            }
            cursor = next.upperBound
        }
        return files
    }

    func reply(_ status: Int, _ error: String) {
        reply(status, json: (try? JSONEncoder().encode(["error": error])) ?? Data())
    }

    func reply(_ status: Int, json: Data) {
        let head = "HTTP/1.1 \(status) \(HTTPURLResponse.localizedString(forStatusCode: status))\r\nContent-Type: application/json\r\nContent-Length: \(json.count)\r\nConnection: close\r\n\r\n"
        connection.send(content: Data(head.utf8) + json, completion: .contentProcessed { _ in self.connection.cancel() })
    }
}
