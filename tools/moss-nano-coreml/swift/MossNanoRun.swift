// Runs the MOSS-TTS-Nano Core ML generation loop natively, to time it without Python's input copies
// and to check the Swift contract. Built and driven by `moss-nano-report`.
//
// usage: MossNanoRun <compiled dir> <request.json> <output prefix> <runs>
// request: {"rows": [[17 x Int32]], "depth": 16, "settings": [6 x Float], "seed": UInt64, "max_frames": Int}
// writes <prefix>.f32 (planar stereo Float32), <prefix>.json (codes, timings)

import CoreML
import Foundation

struct Request: Decodable {
    let rows: [[Int32]]
    let depth: Int
    let settings: [Float]
    let seed: UInt64
    let max_frames: Int
}

struct Config: Decodable {
    let prefill_rows: [Int]
    let contexts: [Int]
    let chunk_frames: Int
    let codec_window_frames: Int
    let codec_caches: [String: [Int]]
    let masked: Float
}

struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func gumbel() -> Float16 {
        let u = (Double(next() >> 11) + 0.5) / 9_007_199_254_740_992
        return Float16(-log(-log(u)))
    }
}

func array(_ shape: [Int], _ type: MLMultiArrayDataType = .float16) throws -> MLMultiArray {
    try MLMultiArray(shape: shape.map(NSNumber.init), dataType: type)
}

func halves(_ a: MLMultiArray) -> UnsafeMutablePointer<Float16> {
    a.dataPointer.assumingMemoryBound(to: Float16.self)
}

func predict(_ model: MLModel, _ inputs: [String: MLMultiArray]) throws -> MLFeatureProvider {
    try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) }))
}

func output(_ result: MLFeatureProvider, _ name: String) -> MLMultiArray {
    result.featureValue(for: name)!.multiArrayValue!
}

/// Copies a Float16 output into a flat row-major array, honoring the strides Core ML chose
/// (Neural Engine outputs can pad rows). Rows are copied whole when the innermost stride is 1.
func values(_ a: MLMultiArray) -> [Float16] {
    let shape = a.shape.map(\.intValue), strides = a.strides.map(\.intValue)
    let source = halves(a), width = shape.last!, rows = a.count / width
    var flat = [Float16](repeating: 0, count: a.count)
    var index = [Int](repeating: 0, count: shape.count - 1)
    flat.withUnsafeMutableBufferPointer { target in
        for row in 0..<rows {
            var offset = 0
            for d in 0..<index.count { offset += index[d] * strides[d] }
            if strides.last! == 1 {
                (target.baseAddress! + row * width).update(from: source + offset, count: width)
            } else {
                for i in 0..<width { target[row * width + i] = source[offset + i * strides.last!] }
            }
            var d = index.count - 1
            while d >= 0 { index[d] += 1; if index[d] < shape[d] { break }; index[d] = 0; d -= 1 }
        }
    }
    return flat
}

let arguments = CommandLine.arguments
let directory = URL(fileURLWithPath: arguments[1])
let request = try JSONDecoder().decode(Request.self, from: Data(contentsOf: URL(fileURLWithPath: arguments[2])))
let config = try JSONDecoder().decode(Config.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
let prefix = arguments[3]
let runs = Int(arguments[4])!

func load(_ package: String, _ function: String) throws -> MLModel {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuAndNeuralEngine
    configuration.functionName = function
    return try MLModel(contentsOf: directory.appendingPathComponent("\(package).mlmodelc"), configuration: configuration)
}

let loadStart = Date()
let prefills = try Dictionary(uniqueKeysWithValues: config.prefill_rows.map { ($0, try load("GlobalDecoder", "prefill_\($0)")) })
let steps = try Dictionary(uniqueKeysWithValues: config.contexts.map { ($0, try load("GlobalDecoder", "step_\($0)")) })
let local = try load("LocalDecoder", "frame_\(request.depth)")
let codec = try load("CodecDecoder", "chunk_\(config.chunk_frames)")
let loadSeconds = Date().timeIntervalSince(loadStart)

let layers = 12, heads = 12, headDim = 64, codebooks = 16, candidates = 32
// Masked slots still enter the matmuls (score + mask, probability 0 x value), so they must hold finite
// numbers: zero the caches once; stale values from earlier utterances are finite and masked.
func zeros(_ shape: [Int]) throws -> MLMultiArray {
    let a = try array(shape)
    memset(a.dataPointer, 0, a.count * 2)
    return a
}
let caches = try Dictionary(uniqueKeysWithValues: config.contexts.map {
    ($0, (try zeros([layers, heads, $0, headDim]), try zeros([layers, heads, $0, headDim])))
})

struct Result {
    var codes: [[Int32]] = []
    var audio: [[Float]] = [[], []]
    var ended = false
    var seconds = 0.0
    var prefill = 0.0, local: [Double] = [], step: [Double] = [], codec: [Double] = []
}

func timed<T>(_ into: inout [Double], _ body: () throws -> T) rethrows -> T {
    let start = DispatchTime.now().uptimeNanoseconds
    defer { into.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9) }
    return try body()
}

func synthesize() throws -> Result {
    var result = Result()
    var rng = SplitMix64(state: request.seed)
    let started = Date()
    let length = request.rows.count
    // Prefill: pad to the smallest bucket; causal attention keeps padding from touching real rows.
    let size = config.prefill_rows.first { $0 >= length }!
    let rows = try array([size, 17], .int32)
    let rowPointer = rows.dataPointer.assumingMemoryBound(to: Int32.self)
    for r in 0..<size {
        for c in 0..<17 { rowPointer[r * 17 + c] = r < length ? request.rows[r][c] : (c == 0 ? 3 : 1024) }
    }
    var prefillTimes: [Double] = []
    let prefilled = try timed(&prefillTimes) { try predict(prefills[size]!, ["rows": rows]) }
    result.prefill = prefillTimes[0]
    let keys = values(output(prefilled, "keys")), vals = values(output(prefilled, "values"))
    for (context, (k, v)) in caches {
        let n = min(length, context)
        for lh in 0..<(layers * heads) {
            keys.withUnsafeBufferPointer { (halves(k) + lh * context * headDim).update(from: $0.baseAddress! + lh * size * headDim, count: n * headDim) }
            vals.withUnsafeBufferPointer { (halves(v) + lh * context * headDim).update(from: $0.baseAddress! + lh * size * headDim, count: n * headDim) }
        }
    }
    let hidden = try array([1, 768])
    let prefillHidden = values(output(prefilled, "hidden"))
    for i in 0..<768 { halves(hidden)[i] = prefillHidden[(length - 1) * 768 + i] }

    let depth = request.depth
    let seen = try array([depth, 1024]), gumbel = try array([depth, candidates]), textGumbel = try array([1, 2])
    let settings = try array([1, 6])
    memset(seen.dataPointer, 0, depth * 1024 * 2)
    for i in 0..<6 { halves(settings)[i] = Float16(request.settings[i]) }

    var codecCaches = try config.codec_caches.mapValues(zeros)
    var pending: [[Int32]] = [], decoded = 0
    func decode() throws {
        let chunk = config.chunk_frames
        let codes = try array([chunk, codebooks])
        for f in 0..<chunk { for c in 0..<codebooks { halves(codes)[f * codebooks + c] = Float16(f < pending.count ? Float(pending[f][c]) : 1024) } }
        let past = try array([1, 1]); halves(past)[0] = Float16(Float(min(decoded, config.codec_window_frames)))
        let out = try timed(&result.codec) { try predict(codec, codecCaches.merging(["codes": codes, "past_frames": past]) { a, _ in a }) }
        for name in codecCaches.keys { codecCaches[name] = output(out, "new_\(name)") }
        let audio = values(output(out, "audio")), samples = audio.count / 2, kept = pending.count * 3840
        for ch in 0..<2 { result.audio[ch] += (0..<kept).map { Float(audio[ch * samples + $0]) } }
        decoded += pending.count
        pending = []
    }

    let largest = config.contexts.max()!
    let row = try array([1, 17], .int32), cos = try array([1, headDim]), sin = try array([1, headDim])
    while result.codes.count < request.max_frames && length + result.codes.count < largest {
        let noise = halves(gumbel), textNoise = halves(textGumbel)
        for i in 0..<(depth * candidates) { noise[i] = rng.gumbel() }
        textNoise[0] = rng.gumbel(); textNoise[1] = rng.gumbel()
        let sampled = try timed(&result.local) {
            try predict(local, ["hidden": hidden, "seen": seen, "gumbel": gumbel, "text_gumbel": textGumbel, "settings": settings])
        }
        if Float(values(output(sampled, "end"))[0]) > 0.5 { result.ended = true; break }
        let drawn = values(output(sampled, "codes"))
        var frame = [Int32](repeating: 1024, count: codebooks)
        let seenFlags = halves(seen)
        for c in 0..<depth { frame[c] = Int32(Float(drawn[c]).rounded()); seenFlags[c * 1024 + Int(frame[c])] = 1 }
        result.codes.append(frame)
        pending.append(frame)
        if pending.count == config.chunk_frames { try decode() }

        // Feed the frame at `position`; the cache holds rows 0..<position.
        let position = length + result.codes.count - 1
        let context = config.contexts.first { $0 >= position }!
        let rowIDs = row.dataPointer.assumingMemoryBound(to: Int32.self)
        rowIDs[0] = 9
        for c in 0..<codebooks { rowIDs[c + 1] = frame[c] }
        let stepMask = try array([1, context])
        let maskValues = halves(stepMask), cosValues = halves(cos), sinValues = halves(sin)
        for j in 0..<context { maskValues[j] = j < position ? 0 : Float16(config.masked) }
        for i in 0..<(headDim / 2) {
            let angle = Double(position) * pow(10000.0, -Double(2 * i) / Double(headDim))
            cosValues[i] = Float16(Foundation.cos(angle)); cosValues[i + 32] = cosValues[i]
            sinValues[i] = Float16(Foundation.sin(angle)); sinValues[i + 32] = sinValues[i]
        }
        let (k, v) = caches[context]!
        let stepped = try timed(&result.step) {
            try predict(steps[context]!, ["row": row, "key_cache": k, "value_cache": v, "mask": stepMask, "cos": cos, "sin": sin])
        }
        let newKey = values(output(stepped, "key")), newValue = values(output(stepped, "value"))
        for (other, (ok, ov)) in caches where position < other {
            let keyRows = halves(ok), valueRows = halves(ov)
            for lh in 0..<(layers * heads) {
                newKey.withUnsafeBufferPointer { (keyRows + (lh * other + position) * headDim).update(from: $0.baseAddress! + lh * headDim, count: headDim) }
                newValue.withUnsafeBufferPointer { (valueRows + (lh * other + position) * headDim).update(from: $0.baseAddress! + lh * headDim, count: headDim) }
            }
        }
        let nextHidden = values(output(stepped, "hidden"))
        nextHidden.withUnsafeBufferPointer { halves(hidden).update(from: $0.baseAddress!, count: 768) }
    }
    if !pending.isEmpty { try decode() }
    result.seconds = Date().timeIntervalSince(started)
    return result
}

var last = Result()
var totals: [Double] = []
for run in 0...runs {
    last = try synthesize()
    if run > 0 { totals.append(last.seconds) }  // run 0 warms up
}
func median(_ x: [Double]) -> Double { x.isEmpty ? 0 : x.sorted()[x.count / 2] }
let duration = Double(last.audio[0].count) / 48000
var bytes = Data()
for channel in last.audio { channel.withUnsafeBufferPointer { bytes.append(Data(buffer: $0)) } }
try bytes.write(to: URL(fileURLWithPath: prefix + ".f32"))
let report: [String: Any] = [
    "codes": last.codes.map { $0.map(Int.init) },
    "ended": last.ended,
    "load_seconds": loadSeconds,
    "duration": duration,
    "seconds": median(totals),
    "rtf": median(totals) / duration,
    "prefill_ms": last.prefill * 1000,
    "local_ms": median(last.local) * 1000,
    "step_ms": median(last.step) * 1000,
    "codec_ms": median(last.codec) * 1000,
    "frames": last.codes.count,
    "host_ms_per_frame": (last.seconds - last.prefill - last.local.reduce(0, +) - last.step.reduce(0, +)
        - last.codec.reduce(0, +)) * 1000 / Double(max(last.codes.count, 1)),
]
try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]).write(to: URL(fileURLWithPath: prefix + ".json"))
print(String(data: try JSONSerialization.data(withJSONObject: report.filter { $0.key != "codes" }, options: [.sortedKeys]), encoding: .utf8)!)
