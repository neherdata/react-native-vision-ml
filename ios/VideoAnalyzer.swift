import Foundation
import AVFoundation
import Vision
import Photos
import UniformTypeIdentifiers

/// Video analysis modes for NSFW detection
enum VideoScanMode: String {
  /// Quick check - just check beginning, middle, and end (3 frames)
  case quickCheck = "quick_check"

  /// Sampled scan - check at regular intervals (e.g., every 5 seconds)
  case sampled = "sampled"

  /// Thorough scan - use Vision framework to find frames with humans, then ONNX those
  /// Best for newer devices with good Vision performance
  case thorough = "thorough"

  /// Binary search - start at middle, expand outward to find NSFW regions
  /// Good fallback for older devices where Vision is slow
  case binarySearch = "binary_search"

  /// Full scan with short-circuit - check every N seconds until first detection
  case fullWithShortCircuit = "full_short_circuit"
}

/// Result from analyzing a single frame
struct FrameAnalysisResult {
  let timestamp: Double  // seconds
  let isNSFW: Bool
  let confidence: Float
  let detections: [NMS.Detection]
  let processingTime: Int  // ms
}

/// Result from analyzing entire video
struct VideoAnalysisResult {
  let isNSFW: Bool
  let nsfwFrameCount: Int
  let totalFramesAnalyzed: Int
  let firstNSFWTimestamp: Double?  // seconds, nil if no NSFW found
  let nsfwTimestamps: [Double]
  let highestConfidence: Float
  let totalProcessingTime: Int  // ms
  let videoDuration: Double  // seconds
  let scanMode: String
  let humanFramesDetected: Int  // frames where Vision detected humans (for thorough mode)
}

/// Delegate protocol for progress and Live Activity updates
protocol VideoAnalyzerDelegate: AnyObject {
  func videoAnalyzer(_ analyzer: VideoAnalyzer, didUpdateProgress progress: Float)
  func videoAnalyzer(_ analyzer: VideoAnalyzer, didFindNSFWAt timestamp: Double, confidence: Float)
  func videoAnalyzer(_ analyzer: VideoAnalyzer, didComplete result: VideoAnalysisResult)
}

/// Video analyzer using Apple Vision framework + ONNX inference
/// For thorough mode: Uses VNDetectHumanRectanglesRequest to find candidate frames,
/// then runs ONNX NSFW detection only on those frames (much more efficient)
class VideoAnalyzer {

  weak var delegate: VideoAnalyzerDelegate?

  private let detector: ONNXInference
  private let inputSize: Int

  // NSFW class indices from NudeNet model
  private let nsfwClassIndices = Set([2, 3, 4, 6, 14])

  // Configuration
  private var sampleInterval: Double = 5.0
  private let quickCheckPoints: [Double] = [0.0, 0.5, 1.0]  // start, middle, end (as ratio)

  // State for cancellation
  private var isCancelled = false

  init(detector: ONNXInference, inputSize: Int = 640) {
    self.detector = detector
    self.inputSize = inputSize
  }

  func cancel() {
    isCancelled = true
  }

  // MARK: - Public API

  /// Analyze a video from PHAsset
  func analyzeVideo(
    assetId: String,
    mode: VideoScanMode,
    sampleInterval: Double? = nil,
    confidenceThreshold: Float = 0.6
  ) throws -> VideoAnalysisResult {

    isCancelled = false
    self.sampleInterval = sampleInterval ?? 5.0

    guard let phAsset = PHAsset.fetchAssets(withLocalIdentifiers: [assetId], options: nil).firstObject,
          phAsset.mediaType == .video else {
      throw VideoAnalyzerError.assetNotFound
    }

    let avAsset = try getAVAsset(from: phAsset)
    let duration = CMTimeGetSeconds(avAsset.duration)

    NSLog("[VideoAnalyzer] Video: %.1fs, mode: %@, interval: %.1fs", duration, mode.rawValue, self.sampleInterval)

    switch mode {
    case .quickCheck:
      return try analyzeQuickCheck(avAsset: avAsset, duration: duration, confidenceThreshold: confidenceThreshold)

    case .sampled, .fullWithShortCircuit:
      return try analyzeSampled(avAsset: avAsset, duration: duration, mode: mode, confidenceThreshold: confidenceThreshold)

    case .thorough:
      return try analyzeThorough(avAsset: avAsset, duration: duration, confidenceThreshold: confidenceThreshold)

    case .binarySearch:
      return try analyzeBinarySearch(avAsset: avAsset, duration: duration, confidenceThreshold: confidenceThreshold)
    }
  }

  // MARK: - Quick Check (3 frames using AVAssetImageGenerator)

  private func analyzeQuickCheck(
    avAsset: AVAsset,
    duration: Double,
    confidenceThreshold: Float
  ) throws -> VideoAnalysisResult {

    let startTime = Date()
    var results: [FrameAnalysisResult] = []

    let timestamps = quickCheckPoints.map { $0 * max(0.1, duration - 0.1) }

    let generator = createImageGenerator(for: avAsset)

    for (index, timestamp) in timestamps.enumerated() {
      guard !isCancelled else { break }

      delegate?.videoAnalyzer(self, didUpdateProgress: Float(index) / Float(timestamps.count))

      if let result = try? analyzeFrameAtTime(generator: generator, timestamp: timestamp, confidenceThreshold: confidenceThreshold) {
        results.append(result)
        if result.isNSFW {
          delegate?.videoAnalyzer(self, didFindNSFWAt: timestamp, confidence: result.confidence)
        }
      }
    }

    return buildResult(results: results, duration: duration, mode: .quickCheck, startTime: startTime, humanFrames: 0)
  }

  // MARK: - Sampled Analysis (AVAssetImageGenerator at intervals)

  private func analyzeSampled(
    avAsset: AVAsset,
    duration: Double,
    mode: VideoScanMode,
    confidenceThreshold: Float
  ) throws -> VideoAnalysisResult {

    let startTime = Date()
    var results: [FrameAnalysisResult] = []

    // Generate sample timestamps
    var timestamps: [Double] = []
    var t = 0.0
    while t < duration {
      timestamps.append(t)
      t += sampleInterval
    }
    // Include near-end frame
    if timestamps.last ?? 0 < duration - 1.0 {
      timestamps.append(max(0, duration - 0.5))
    }

    NSLog("[VideoAnalyzer] Sampling %d frames at %.1fs intervals", timestamps.count, sampleInterval)

    let generator = createImageGenerator(for: avAsset)

    for (index, timestamp) in timestamps.enumerated() {
      guard !isCancelled else { break }

      delegate?.videoAnalyzer(self, didUpdateProgress: Float(index) / Float(timestamps.count))

      if let result = try? analyzeFrameAtTime(generator: generator, timestamp: timestamp, confidenceThreshold: confidenceThreshold) {
        results.append(result)

        if result.isNSFW {
          delegate?.videoAnalyzer(self, didFindNSFWAt: timestamp, confidence: result.confidence)

          // Short-circuit if requested
          if mode == .fullWithShortCircuit {
            NSLog("[VideoAnalyzer] NSFW at %.1fs, short-circuiting", timestamp)
            break
          }
        }
      }
    }

    return buildResult(results: results, duration: duration, mode: mode, startTime: startTime, humanFrames: 0)
  }

  // MARK: - Thorough Analysis (Vision human detection + ONNX)

  private func analyzeThorough(
    avAsset: AVAsset,
    duration: Double,
    confidenceThreshold: Float
  ) throws -> VideoAnalysisResult {

    let startTime = Date()

    // Step 1: Use Vision to find frames with humans (much faster than ONNX on all frames)
    NSLog("[VideoAnalyzer] Phase 1: Scanning for humans with Vision framework...")

    let humanTimestamps = try findHumanFrames(avAsset: avAsset, duration: duration)

    NSLog("[VideoAnalyzer] Found %d frames with humans", humanTimestamps.count)

    if humanTimestamps.isEmpty {
      // No humans = no NSFW content
      return VideoAnalysisResult(
        isNSFW: false,
        nsfwFrameCount: 0,
        totalFramesAnalyzed: Int(duration / sampleInterval),
        firstNSFWTimestamp: nil,
        nsfwTimestamps: [],
        highestConfidence: 0,
        totalProcessingTime: Int(Date().timeIntervalSince(startTime) * 1000),
        videoDuration: duration,
        scanMode: VideoScanMode.thorough.rawValue,
        humanFramesDetected: 0
      )
    }

    // Step 2: Run ONNX only on frames with humans
    NSLog("[VideoAnalyzer] Phase 2: Running ONNX on %d candidate frames...", humanTimestamps.count)

    var results: [FrameAnalysisResult] = []
    let generator = createImageGenerator(for: avAsset)

    for (index, timestamp) in humanTimestamps.enumerated() {
      guard !isCancelled else { break }

      // Progress: 50% for human detection, 50% for ONNX
      let progress = 0.5 + (Float(index) / Float(humanTimestamps.count)) * 0.5
      delegate?.videoAnalyzer(self, didUpdateProgress: progress)

      if let result = try? analyzeFrameAtTime(generator: generator, timestamp: timestamp, confidenceThreshold: confidenceThreshold) {
        results.append(result)

        if result.isNSFW {
          delegate?.videoAnalyzer(self, didFindNSFWAt: timestamp, confidence: result.confidence)
        }
      }
    }

    return buildResult(results: results, duration: duration, mode: .thorough, startTime: startTime, humanFrames: humanTimestamps.count)
  }

  // MARK: - Binary Search (fallback for older devices)

  private func analyzeBinarySearch(
    avAsset: AVAsset,
    duration: Double,
    confidenceThreshold: Float
  ) throws -> VideoAnalysisResult {

    let startTime = Date()
    var analyzedTimestamps = Set<Double>()
    var results: [FrameAnalysisResult] = []

    let generator = createImageGenerator(for: avAsset)
    let binarySearchWindow: Double = 5.0
    let binarySearchDepth = 3

    // Start with middle ± window
    let middle = duration / 2.0
    var queue: [Double] = []
    for offset in stride(from: -binarySearchWindow, through: binarySearchWindow, by: 1.0) {
      let t = middle + offset
      if t >= 0 && t <= duration {
        queue.append(t)
      }
    }

    var depth = 0
    while depth < binarySearchDepth && !queue.isEmpty {
      guard !isCancelled else { break }

      var nextQueue: [Double] = []

      for timestamp in queue {
        let roundedTime = round(timestamp * 2) / 2
        if analyzedTimestamps.contains(roundedTime) { continue }
        analyzedTimestamps.insert(roundedTime)

        delegate?.videoAnalyzer(self, didUpdateProgress: Float(analyzedTimestamps.count) / Float(min(50, Int(duration / 2))))

        if let result = try? analyzeFrameAtTime(generator: generator, timestamp: timestamp, confidenceThreshold: confidenceThreshold) {
          results.append(result)

          if result.isNSFW {
            delegate?.videoAnalyzer(self, didFindNSFWAt: timestamp, confidence: result.confidence)
            NSLog("[VideoAnalyzer] Binary search: NSFW at %.1fs, expanding", timestamp)

            let before = timestamp - binarySearchWindow
            let after = timestamp + binarySearchWindow
            if before >= 0 { nextQueue.append(before) }
            if after <= duration { nextQueue.append(after) }
          }
        }
      }

      queue = nextQueue
      depth += 1
    }

    return buildResult(results: results, duration: duration, mode: .binarySearch, startTime: startTime, humanFrames: 0)
  }

  /// Use VNDetectHumanRectanglesRequest to find frames containing humans
  private func findHumanFrames(avAsset: AVAsset, duration: Double) throws -> [Double] {
    var humanTimestamps: [Double] = []

    let generator = AVAssetImageGenerator(asset: avAsset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = CMTime(seconds: 0.5, preferredTimescale: 600)
    generator.requestedTimeToleranceAfter = CMTime(seconds: 0.5, preferredTimescale: 600)
    // Smaller size for faster Vision processing
    generator.maximumSize = CGSize(width: 320, height: 320)

    var t = 0.0
    var frameIndex = 0
    let totalFrames = Int(duration / sampleInterval) + 1

    while t < duration {
      guard !isCancelled else { break }

      // Progress for phase 1 (0-50%)
      delegate?.videoAnalyzer(self, didUpdateProgress: Float(frameIndex) / Float(totalFrames) * 0.5)

      let requestTime = CMTime(seconds: t, preferredTimescale: 600)
      var actualTime = CMTime.zero

      if let cgImage = try? generator.copyCGImage(at: requestTime, actualTime: &actualTime) {
        // Run Vision human detection
        let hasHuman = detectHumanInFrame(cgImage: cgImage)
        if hasHuman {
          humanTimestamps.append(t)
        }
      }

      t += sampleInterval
      frameIndex += 1
    }

    return humanTimestamps
  }

  /// Quick Vision-based human detection for pre-filtering
  private func detectHumanInFrame(cgImage: CGImage) -> Bool {
    let request = VNDetectHumanRectanglesRequest()
    request.upperBodyOnly = false

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

    do {
      try handler.perform([request])
      if let results = request.results, !results.isEmpty {
        return true
      }
    } catch {
      // Silently continue on Vision errors
    }

    return false
  }

  // MARK: - Helper Methods

  private func getAVAsset(from phAsset: PHAsset) throws -> AVAsset {
    let semaphore = DispatchSemaphore(value: 0)
    var resultAsset: AVAsset?
    var resultError: Error?

    let options = PHVideoRequestOptions()
    options.isNetworkAccessAllowed = false
    options.deliveryMode = .automatic

    PHImageManager.default().requestAVAsset(forVideo: phAsset, options: options) { asset, _, info in
      if let error = info?[PHImageErrorKey] as? Error {
        resultError = error
      } else {
        resultAsset = asset
      }
      semaphore.signal()
    }

    semaphore.wait()

    if let error = resultError { throw error }
    guard let asset = resultAsset else { throw VideoAnalyzerError.failedToLoadVideo }
    return asset
  }

  private func createImageGenerator(for avAsset: AVAsset) -> AVAssetImageGenerator {
    let generator = AVAssetImageGenerator(asset: avAsset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = CMTime(seconds: 0.1, preferredTimescale: 600)
    generator.requestedTimeToleranceAfter = CMTime(seconds: 0.1, preferredTimescale: 600)
    generator.maximumSize = CGSize(width: inputSize, height: inputSize)
    return generator
  }

  private func analyzeFrameAtTime(
    generator: AVAssetImageGenerator,
    timestamp: Double,
    confidenceThreshold: Float
  ) throws -> FrameAnalysisResult {

    let frameStart = Date()
    let requestTime = CMTime(seconds: timestamp, preferredTimescale: 600)

    var actualTime = CMTime.zero
    let cgImage = try generator.copyCGImage(at: requestTime, actualTime: &actualTime)

    // Write to temp file for ONNX detector
    let tempURL = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString)
      .appendingPathExtension("jpg")

    defer { try? FileManager.default.removeItem(at: tempURL) }

    // UTType requires iOS 14+ which is our minimum deployment target
    let jpegType = UTType.jpeg.identifier as CFString

    guard let destination = CGImageDestinationCreateWithURL(tempURL as CFURL, jpegType, 1, nil) else {
      throw VideoAnalyzerError.frameExtractionFailed
    }
    CGImageDestinationAddImage(destination, cgImage, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw VideoAnalyzerError.frameExtractionFailed
    }

    let result = try detector.detect(
      imageUri: tempURL.absoluteString,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: 0.45
    )

    let nsfwDetections = result.detections.filter { nsfwClassIndices.contains($0.classIndex) }
    let processingTime = Int(Date().timeIntervalSince(frameStart) * 1000)

    return FrameAnalysisResult(
      timestamp: timestamp,
      isNSFW: !nsfwDetections.isEmpty,
      confidence: nsfwDetections.map { $0.score }.max() ?? 0,
      detections: result.detections,
      processingTime: processingTime
    )
  }

  private func buildResult(
    results: [FrameAnalysisResult],
    duration: Double,
    mode: VideoScanMode,
    startTime: Date,
    humanFrames: Int
  ) -> VideoAnalysisResult {

    let nsfwFrames = results.filter { $0.isNSFW }
    let totalTime = Int(Date().timeIntervalSince(startTime) * 1000)

    let result = VideoAnalysisResult(
      isNSFW: !nsfwFrames.isEmpty,
      nsfwFrameCount: nsfwFrames.count,
      totalFramesAnalyzed: results.count,
      firstNSFWTimestamp: nsfwFrames.first?.timestamp,
      nsfwTimestamps: nsfwFrames.map { $0.timestamp }.sorted(),
      highestConfidence: results.map { $0.confidence }.max() ?? 0,
      totalProcessingTime: totalTime,
      videoDuration: duration,
      scanMode: mode.rawValue,
      humanFramesDetected: humanFrames
    )

    delegate?.videoAnalyzer(self, didComplete: result)
    delegate?.videoAnalyzer(self, didUpdateProgress: 1.0)

    return result
  }
}

// MARK: - Errors

enum VideoAnalyzerError: Error, LocalizedError {
  case assetNotFound
  case failedToLoadVideo
  case frameExtractionFailed
  case detectorNotReady
  case cancelled

  var errorDescription: String? {
    switch self {
    case .assetNotFound: return "Video asset not found"
    case .failedToLoadVideo: return "Failed to load video from Photos library"
    case .frameExtractionFailed: return "Failed to extract frame from video"
    case .detectorNotReady: return "ONNX detector not initialized"
    case .cancelled: return "Video analysis was cancelled"
    }
  }
}
