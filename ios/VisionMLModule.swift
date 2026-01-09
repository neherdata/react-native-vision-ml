import Foundation
import React
import Vision
import Photos
import UIKit
#if canImport(SensitiveContentAnalysis)
import SensitiveContentAnalysis
#endif

@objc(VisionML)
class VisionMLModule: NSObject {

  // Map of detector instances by ID
  private var detectors: [String: ONNXInference] = [:]
  private let detectorsQueue = DispatchQueue(label: "com.visionml.detectors", attributes: .concurrent)

  // Counter for generating unique detector IDs
  private var detectorIdCounter = 0
  private let counterQueue = DispatchQueue(label: "com.visionml.counter")

  /// Create a new detector instance with its own model and configuration
  /// Returns a unique detector ID for subsequent operations
  @objc(createDetector:classLabels:inputSize:resolve:reject:)
  func createDetector(
    _ modelPath: String,
    classLabels: [String],
    inputSize: NSNumber,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        // Create inference instance
        let inference = ONNXInference(
          classLabels: classLabels,
          inputSize: inputSize.intValue
        )

        // Load model
        try inference.loadModel(modelPath: modelPath)

        // Generate unique ID
        let detectorId = self.counterQueue.sync {
          self.detectorIdCounter += 1
          return "detector_\(self.detectorIdCounter)"
        }

        // Store detector
        self.detectorsQueue.async(flags: .barrier) {
          self.detectors[detectorId] = inference
        }

        DispatchQueue.main.async {
          resolve([
            "detectorId": detectorId,
            "success": true,
            "message": "Detector created successfully"
          ])
        }
      } catch {
        DispatchQueue.main.async {
          reject("DETECTOR_CREATE_ERROR", "Failed to create detector: \(error.localizedDescription)", error)
        }
      }
    }
  }

  /// Run detection using a specific detector instance
  @objc(detect:imageUri:confidenceThreshold:iouThreshold:resolve:reject:)
  func detect(
    _ detectorId: String,
    imageUri: String,
    confidenceThreshold: NSNumber,
    iouThreshold: NSNumber,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.global(qos: .userInitiated).async {
      // Get detector instance
      let detector = self.detectorsQueue.sync {
        return self.detectors[detectorId]
      }

      guard let inference = detector else {
        DispatchQueue.main.async {
          reject("DETECTOR_NOT_FOUND", "Detector with ID '\(detectorId)' not found. Create a detector first.", nil)
        }
        return
      }

      do {
        // Run full detection pipeline
        let result = try inference.detect(
          imageUri: imageUri,
          confidenceThreshold: confidenceThreshold.floatValue,
          iouThreshold: iouThreshold.floatValue
        )

        // Convert detections to JSON-serializable format
        let detectionsArray = result.detections.map { detection in
          return [
            "box": detection.box,  // [x1, y1, x2, y2] - already scaled to original image
            "score": NSNumber(value: detection.score),  // Explicitly convert to NSNumber for RN bridge
            "classIndex": detection.classIndex,
            "className": detection.className
          ] as [String: Any]
        }

        DispatchQueue.main.async {
          resolve([
            "detections": detectionsArray,
            "inferenceTime": result.inferenceTime,
            "postProcessTime": result.postProcessTime,
            "totalTime": result.totalTime,
            "debugInfo": result.debugInfo
          ])
        }
      } catch {
        DispatchQueue.main.async {
          reject("INFERENCE_ERROR", "Inference failed: \(error.localizedDescription)", error)
        }
      }
    }
  }

  /// Dispose of a specific detector instance
  @objc(disposeDetector:resolve:reject:)
  func disposeDetector(
    _ detectorId: String,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    detectorsQueue.async(flags: .barrier) {
      if let detector = self.detectors.removeValue(forKey: detectorId) {
        detector.dispose()
        DispatchQueue.main.async {
          resolve(["success": true, "message": "Detector disposed"])
        }
      } else {
        DispatchQueue.main.async {
          reject("DETECTOR_NOT_FOUND", "Detector with ID '\(detectorId)' not found", nil)
        }
      }
    }
  }

  /// Dispose of all detector instances
  @objc(disposeAllDetectors:reject:)
  func disposeAllDetectors(
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    detectorsQueue.async(flags: .barrier) {
      for (_, detector) in self.detectors {
        detector.dispose()
      }
      self.detectors.removeAll()
      DispatchQueue.main.async {
        resolve(["success": true, "message": "All detectors disposed"])
      }
    }
  }

  // MARK: - Live Activity Methods

  /// Check if Live Activities are available on this device
  @objc(isLiveActivityAvailable:reject:)
  func isLiveActivityAvailable(
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    if #available(iOS 16.1, *) {
      resolve(VideoScanActivityManager.isAvailable)
    } else {
      resolve(false)
    }
  }

  /// Start a Live Activity for video scanning
  @objc(startVideoScanActivity:videoDuration:scanMode:resolve:reject:)
  func startVideoScanActivity(
    _ videoName: String,
    videoDuration: NSNumber,
    scanMode: String,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    if #available(iOS 16.1, *) {
      if let activityId = VideoScanActivityManager.shared.startActivity(
        videoName: videoName,
        videoDuration: videoDuration.doubleValue,
        scanMode: scanMode
      ) {
        resolve(["activityId": activityId, "success": true])
      } else {
        resolve(["activityId": NSNull(), "success": false])
      }
    } else {
      resolve(["activityId": NSNull(), "success": false])
    }
  }

  /// Update Live Activity progress
  @objc(updateVideoScanActivity:phase:nsfwCount:framesAnalyzed:resolve:reject:)
  func updateVideoScanActivity(
    _ progress: NSNumber,
    phase: String,
    nsfwCount: NSNumber,
    framesAnalyzed: NSNumber,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    if #available(iOS 16.1, *) {
      VideoScanActivityManager.shared.updateProgress(
        progress: progress.floatValue,
        phase: phase,
        nsfwCount: nsfwCount.intValue,
        framesAnalyzed: framesAnalyzed.intValue
      )
      resolve(true)
    } else {
      resolve(false)
    }
  }

  /// End Live Activity with results
  @objc(endVideoScanActivity:framesAnalyzed:isNSFW:resolve:reject:)
  func endVideoScanActivity(
    _ nsfwCount: NSNumber,
    framesAnalyzed: NSNumber,
    isNSFW: Bool,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    if #available(iOS 16.1, *) {
      VideoScanActivityManager.shared.completeActivity(
        nsfwCount: nsfwCount.intValue,
        framesAnalyzed: framesAnalyzed.intValue,
        isNSFW: isNSFW
      )
      resolve(true)
    } else {
      resolve(false)
    }
  }

  // MARK: - Video Analysis Methods

  /// Analyze a video for NSFW content
  /// - Parameters:
  ///   - detectorId: ID of the detector to use
  ///   - assetId: PHAsset local identifier
  ///   - mode: Scan mode (full_short_circuit, sampled, binary_search, quick_check)
  ///   - sampleInterval: Seconds between samples (for sampled mode)
  ///   - confidenceThreshold: Minimum confidence threshold
  @objc(analyzeVideo:assetId:mode:sampleInterval:confidenceThreshold:resolve:reject:)
  func analyzeVideo(
    _ detectorId: String,
    assetId: String,
    mode: String,
    sampleInterval: NSNumber,
    confidenceThreshold: NSNumber,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.global(qos: .userInitiated).async {
      // Get detector instance
      let detector = self.detectorsQueue.sync {
        return self.detectors[detectorId]
      }

      guard let inference = detector else {
        DispatchQueue.main.async {
          reject("DETECTOR_NOT_FOUND", "Detector with ID '\(detectorId)' not found", nil)
        }
        return
      }

      // Parse scan mode
      let scanMode: VideoScanMode
      switch mode {
      case "quick_check":
        scanMode = .quickCheck
      case "sampled":
        scanMode = .sampled
      case "thorough":
        scanMode = .thorough
      case "binary_search":
        scanMode = .binarySearch
      case "full_short_circuit":
        scanMode = .fullWithShortCircuit
      default:
        scanMode = .sampled
      }

      let analyzer = VideoAnalyzer(detector: inference, inputSize: 640)

      do {
        let result = try analyzer.analyzeVideo(
          assetId: assetId,
          mode: scanMode,
          sampleInterval: sampleInterval.doubleValue > 0 ? sampleInterval.doubleValue : nil,
          confidenceThreshold: confidenceThreshold.floatValue
        )

        DispatchQueue.main.async {
          resolve([
            "isNSFW": result.isNSFW,
            "nsfwFrameCount": result.nsfwFrameCount,
            "totalFramesAnalyzed": result.totalFramesAnalyzed,
            "firstNSFWTimestamp": result.firstNSFWTimestamp ?? NSNull(),
            "nsfwTimestamps": result.nsfwTimestamps,
            "highestConfidence": result.highestConfidence,
            "totalProcessingTime": result.totalProcessingTime,
            "videoDuration": result.videoDuration,
            "scanMode": result.scanMode,
            "humanFramesDetected": result.humanFramesDetected
          ])
        }
      } catch {
        DispatchQueue.main.async {
          reject("VIDEO_ANALYSIS_ERROR", "Video analysis failed: \(error.localizedDescription)", error)
        }
      }
    }
  }

  /// Quick check a video (start, middle, end only)
  @objc(quickCheckVideo:assetId:confidenceThreshold:resolve:reject:)
  func quickCheckVideo(
    _ detectorId: String,
    assetId: String,
    confidenceThreshold: NSNumber,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    analyzeVideo(
      detectorId,
      assetId: assetId,
      mode: "quick_check",
      sampleInterval: 0,
      confidenceThreshold: confidenceThreshold,
      resolve: resolve,
      reject: reject
    )
  }

  // MARK: - Vision Framework Methods

  @objc(analyzeAnimals:resolve:reject:)
  func analyzeAnimals(
    _ assetId: String,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [assetId], options: nil).firstObject else {
      reject("ASSET_NOT_FOUND", "Could not find asset with ID: \(assetId)", nil)
      return
    }

    let options = PHImageRequestOptions()
    options.deliveryMode = .highQualityFormat
    options.isSynchronous = false
    options.isNetworkAccessAllowed = false

    PHImageManager.default().requestImage(
      for: asset,
      targetSize: CGSize(width: 640, height: 640),
      contentMode: .aspectFit,
      options: options
    ) { image, _ in
      guard let image = image, let cgImage = image.cgImage else {
        reject("IMAGE_LOAD_FAILED", "Could not load image for asset", nil)
        return
      }

      self.detectAnimals(cgImage: cgImage) { result in
        resolve(result)
      }
    }
  }

  @objc(analyzeHumanPose:resolve:reject:)
  func analyzeHumanPose(
    _ assetId: String,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [assetId], options: nil).firstObject else {
      reject("ASSET_NOT_FOUND", "Could not find asset with ID: \(assetId)", nil)
      return
    }

    let options = PHImageRequestOptions()
    options.deliveryMode = .highQualityFormat
    options.isSynchronous = false
    options.isNetworkAccessAllowed = false

    PHImageManager.default().requestImage(
      for: asset,
      targetSize: CGSize(width: 640, height: 640),
      contentMode: .aspectFit,
      options: options
    ) { image, _ in
      guard let image = image, let cgImage = image.cgImage else {
        reject("IMAGE_LOAD_FAILED", "Could not load image for asset", nil)
        return
      }

      self.detectHumanPose(cgImage: cgImage) { result in
        resolve(result)
      }
    }
  }

  @objc(analyzeComprehensive:resolve:reject:)
  func analyzeComprehensive(
    _ assetId: String,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [assetId], options: nil).firstObject else {
      reject("ASSET_NOT_FOUND", "Could not find asset with ID: \(assetId)", nil)
      return
    }

    let options = PHImageRequestOptions()
    options.deliveryMode = .highQualityFormat
    options.isSynchronous = false
    options.isNetworkAccessAllowed = false

    PHImageManager.default().requestImage(
      for: asset,
      targetSize: CGSize(width: 640, height: 640),
      contentMode: .aspectFit,
      options: options
    ) { image, _ in
      guard let image = image, let cgImage = image.cgImage else {
        reject("IMAGE_LOAD_FAILED", "Could not load image for asset", nil)
        return
      }

      self.performComprehensiveAnalysis(cgImage: cgImage) { result in
        resolve(result)
      }
    }
  }

  // MARK: - Vision Framework Helper Methods

  private func detectAnimals(cgImage: CGImage, completion: @escaping ([String: Any]) -> Void) {
    let request = VNRecognizeAnimalsRequest { request, error in
      guard error == nil else {
        completion(["animals": [], "count": 0])
        return
      }

      guard let observations = request.results as? [VNRecognizedObjectObservation] else {
        completion(["animals": [], "count": 0])
        return
      }

      let animals = observations.map { observation -> [String: Any] in
        let labels = observation.labels.map { label -> [String: Any] in
          return [
            "identifier": label.identifier,
            "confidence": label.confidence
          ]
        }

        return [
          "boundingBox": [
            "x": observation.boundingBox.origin.x,
            "y": observation.boundingBox.origin.y,
            "width": observation.boundingBox.size.width,
            "height": observation.boundingBox.size.height
          ],
          "confidence": observation.confidence,
          "labels": labels
        ]
      }

      completion([
        "animals": animals,
        "count": animals.count
      ])
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    do {
      try handler.perform([request])
    } catch {
      completion(["animals": [], "count": 0])
    }
  }

  private func detectHumanPose(cgImage: CGImage, completion: @escaping ([String: Any]) -> Void) {
    let request = VNDetectHumanBodyPoseRequest { request, error in
      guard error == nil else {
        completion(["humans": [], "humanCount": 0])
        return
      }

      guard let observations = request.results as? [VNHumanBodyPoseObservation] else {
        completion(["humans": [], "humanCount": 0])
        return
      }

      let humans = observations.compactMap { observation -> [String: Any]? in
        guard let recognizedPoints = try? observation.recognizedPoints(.all) else {
          return nil
        }

        var points: [String: [String: Any]] = [:]
        var pointCount = 0

        for (jointName, point) in recognizedPoints {
          if point.confidence > 0.1 {
            let keyString = String(describing: jointName.rawValue)
            points[keyString] = [
              "x": point.location.x,
              "y": point.location.y,
              "confidence": point.confidence
            ]
            pointCount += 1
          }
        }

        return [
          "points": points,
          "pointCount": pointCount
        ]
      }

      completion([
        "humans": humans,
        "humanCount": humans.count
      ])
    }

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    do {
      try handler.perform([request])
    } catch {
      completion(["humans": [], "humanCount": 0])
    }
  }

  private func performComprehensiveAnalysis(cgImage: CGImage, completion: @escaping ([String: Any]) -> Void) {
    var result: [String: Any] = [:]
    let resultQueue = DispatchQueue(label: "com.visionml.result", attributes: .concurrent)
    let group = DispatchGroup()

    // Scene Classification
    group.enter()
    let sceneRequest = VNClassifyImageRequest { request, error in
      defer { group.leave() }
      guard error == nil,
            let observations = request.results as? [VNClassificationObservation] else {
        return
      }

      let scenes = observations
        .filter { $0.confidence > 0.1 }
        .prefix(10)
        .map { ["identifier": $0.identifier, "confidence": $0.confidence] }

      resultQueue.async(flags: .barrier) {
        result["scenes"] = scenes
      }
    }

    // Face Detection
    group.enter()
    let faceRequest = VNDetectFaceRectanglesRequest { request, error in
      defer { group.leave() }
      guard error == nil,
            let observations = request.results as? [VNFaceObservation] else {
        return
      }

      let faces = observations.map { face -> [String: Any] in
        return [
          "boundingBox": [
            "x": face.boundingBox.origin.x,
            "y": face.boundingBox.origin.y,
            "width": face.boundingBox.size.width,
            "height": face.boundingBox.size.height
          ],
          "confidence": face.confidence
        ]
      }

      resultQueue.async(flags: .barrier) {
        result["faces"] = faces
        result["faceCount"] = faces.count
      }
    }

    // Animal Detection
    group.enter()
    let animalRequest = VNRecognizeAnimalsRequest { request, error in
      defer { group.leave() }
      guard error == nil,
            let observations = request.results as? [VNRecognizedObjectObservation] else {
        return
      }

      let animals = observations.map { observation -> [String: Any] in
        let labels = observation.labels.map { label -> [String: Any] in
          return [
            "identifier": label.identifier,
            "confidence": label.confidence
          ]
        }

        return [
          "boundingBox": [
            "x": observation.boundingBox.origin.x,
            "y": observation.boundingBox.origin.y,
            "width": observation.boundingBox.size.width,
            "height": observation.boundingBox.size.height
          ],
          "confidence": observation.confidence,
          "labels": labels
        ]
      }

      resultQueue.async(flags: .barrier) {
        result["animals"] = animals
        result["animalCount"] = animals.count
      }
    }

    // Human Pose Detection
    group.enter()
    let poseRequest = VNDetectHumanBodyPoseRequest { request, error in
      defer { group.leave() }
      guard error == nil,
            let observations = request.results as? [VNHumanBodyPoseObservation] else {
        return
      }

      resultQueue.async(flags: .barrier) {
        result["humanCount"] = observations.count
        result["hasHumans"] = observations.count > 0
      }
    }

    // Text Detection
    group.enter()
    let textRequest = VNRecognizeTextRequest { request, error in
      defer { group.leave() }
      guard error == nil,
            let observations = request.results as? [VNRecognizedTextObservation] else {
        return
      }

      resultQueue.async(flags: .barrier) {
        result["hasText"] = observations.count > 0
        result["textRegions"] = observations.count
      }
    }

    // Rectangle Detection (for screenshots)
    group.enter()
    let rectangleRequest = VNDetectRectanglesRequest { request, error in
      defer { group.leave() }
      guard error == nil,
            let observations = request.results as? [VNRectangleObservation] else {
        return
      }

      let rectangleCount = observations.count
      resultQueue.async(flags: .barrier) {
        result["rectangles"] = rectangleCount
        result["likelyScreenshot"] = rectangleCount > 5
      }
    }

    // Execute all requests
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    do {
      try handler.perform([
        sceneRequest,
        faceRequest,
        animalRequest,
        poseRequest,
        textRequest,
        rectangleRequest
      ])
    } catch {
      // Return partial results on error
    }

    // Wait for all requests to complete
    group.notify(queue: .main) {
      // Ensure all barrier writes have completed before reading
      resultQueue.sync(flags: .barrier) {
        completion(result)
      }
    }
  }

  // MARK: - Sensitive Content Analysis (iOS 17+)

  /// Get the status of Sensitive Content Analysis
  /// Returns availability, policy status, and settings URL
  @objc(getSensitiveContentAnalysisStatus:reject:)
  func getSensitiveContentAnalysisStatus(
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    #if canImport(SensitiveContentAnalysis)
    if #available(iOS 17.0, *) {
      let analyzer = SCSensitivityAnalyzer()
      let policy = analyzer.analysisPolicy

      var policyString: String
      var isEnabled: Bool
      var policyRawValue: Int = -1

      switch policy {
      case .disabled:
        policyString = "disabled"
        isEnabled = false
        policyRawValue = 0
      case .simpleInterventions:
        policyString = "simple_interventions"
        isEnabled = true
        policyRawValue = 1
      case .descriptiveInterventions:
        policyString = "descriptive_interventions"
        isEnabled = true
        policyRawValue = 2
      @unknown default:
        policyString = "unknown"
        isEnabled = false
        policyRawValue = 99
      }

      // Debug: Log the raw policy value
      print("[VisionML] SCA Policy check - raw: \(policy), string: \(policyString), enabled: \(isEnabled)")

      resolve([
        "available": true,
        "enabled": isEnabled,
        "policy": policyString,
        "policyRawValue": policyRawValue,
        "iosVersion": UIDevice.current.systemVersion,
        "hint": isEnabled ? "SCA is ready to use" : "Enable at least one option under Sensitive Content Warning in iOS Settings"
      ])
    } else {
      resolve([
        "available": false,
        "enabled": false,
        "policy": "unsupported",
        "reason": "Requires iOS 17.0 or later",
        "iosVersion": UIDevice.current.systemVersion
      ])
    }
    #else
    resolve([
      "available": false,
      "enabled": false,
      "policy": "unsupported",
      "reason": "SensitiveContentAnalysis framework not available",
      "iosVersion": UIDevice.current.systemVersion
    ])
    #endif
  }

  /// Open iOS Settings - no reliable deep link to SCA exists
  @objc(openSensitiveContentSettings:reject:)
  func openSensitiveContentSettings(
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.main.async {
      // Apple doesn't provide a public deep link to Privacy & Security or SCA
      // App-Prefs: URLs are private APIs that break frequently
      // Just open the main Settings app and provide clear instructions
      if let url = URL(string: "App-Prefs:root=Privacy") {
        UIApplication.shared.open(url, options: [:]) { success in
          resolve([
            "opened": success,
            "url": "Privacy",
            "instructions": "Scroll down to 'Sensitive Content Warning' and turn it ON"
          ])
        }
      } else if let url = URL(string: UIApplication.openSettingsURLString) {
        UIApplication.shared.open(url, options: [:]) { success in
          resolve([
            "opened": success,
            "url": "Settings",
            "instructions": "Go to: Privacy & Security → Sensitive Content Warning (scroll down)"
          ])
        }
      } else {
        resolve([
          "opened": false,
          "url": nil as String?,
          "instructions": "Open Settings → Privacy & Security → Sensitive Content Warning"
        ])
      }
    }
  }

  /// Analyze a single image for sensitive content using Apple's SCA
  /// Returns isSensitive boolean - much faster than ONNX but no bounding boxes
  @objc(analyzeSensitiveContent:resolve:reject:)
  func analyzeSensitiveContent(
    _ assetId: String,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    #if canImport(SensitiveContentAnalysis)
    if #available(iOS 17.0, *) {
      let analyzer = SCSensitivityAnalyzer()

      guard analyzer.analysisPolicy != .disabled else {
        resolve([
          "available": false,
          "isSensitive": false,
          "reason": "disabled_by_user"
        ])
        return
      }

      // Fetch the PHAsset
      guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [assetId], options: nil).firstObject else {
        reject("ASSET_NOT_FOUND", "Could not find asset with ID: \(assetId)", nil)
        return
      }

      let options = PHImageRequestOptions()
      options.deliveryMode = .highQualityFormat
      options.isSynchronous = false
      options.isNetworkAccessAllowed = false

      PHImageManager.default().requestImage(
        for: asset,
        targetSize: CGSize(width: 640, height: 640),
        contentMode: .aspectFit,
        options: options
      ) { image, _ in
        guard let image = image, let cgImage = image.cgImage else {
          reject("IMAGE_LOAD_FAILED", "Could not load image for asset", nil)
          return
        }

        Task {
          do {
            let response = try await analyzer.analyzeImage(cgImage)

            DispatchQueue.main.async {
              resolve([
                "available": true,
                "isSensitive": response.isSensitive,
                "assetId": assetId
              ])
            }
          } catch {
            DispatchQueue.main.async {
              reject("SCA_ERROR", "Sensitive content analysis failed: \(error.localizedDescription)", error)
            }
          }
        }
      }
    } else {
      resolve([
        "available": false,
        "isSensitive": false,
        "reason": "ios_version"
      ])
    }
    #else
    resolve([
      "available": false,
      "isSensitive": false,
      "reason": "framework_unavailable"
    ])
    #endif
  }

  /// Batch analyze multiple images for sensitive content
  /// Returns array of results - use as pre-filter before detailed ONNX scan
  @objc(batchAnalyzeSensitiveContent:resolve:reject:)
  func batchAnalyzeSensitiveContent(
    _ assetIds: [String],
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    #if canImport(SensitiveContentAnalysis)
    if #available(iOS 17.0, *) {
      let analyzer = SCSensitivityAnalyzer()

      guard analyzer.analysisPolicy != .disabled else {
        resolve([
          "available": false,
          "results": [],
          "reason": "disabled_by_user"
        ])
        return
      }

      DispatchQueue.global(qos: .userInitiated).async {
        var results: [[String: Any]] = []
        let resultQueue = DispatchQueue(label: "com.visionml.sca.results")
        let group = DispatchGroup()

        for assetId in assetIds {
          group.enter()

          guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [assetId], options: nil).firstObject else {
            resultQueue.async {
              results.append([
                "assetId": assetId,
                "isSensitive": false,
                "error": "asset_not_found"
              ])
            }
            group.leave()
            continue
          }

          let options = PHImageRequestOptions()
          options.deliveryMode = .fastFormat  // Use fast format for batch
          options.isSynchronous = true
          options.isNetworkAccessAllowed = false

          PHImageManager.default().requestImage(
            for: asset,
            targetSize: CGSize(width: 320, height: 320),  // Smaller size for speed
            contentMode: .aspectFit,
            options: options
          ) { image, _ in
            guard let image = image, let cgImage = image.cgImage else {
              resultQueue.async {
                results.append([
                  "assetId": assetId,
                  "isSensitive": false,
                  "error": "image_load_failed"
                ])
              }
              group.leave()
              return
            }

            Task {
              do {
                let response = try await analyzer.analyzeImage(cgImage)

                resultQueue.async {
                  results.append([
                    "assetId": assetId,
                    "isSensitive": response.isSensitive
                  ])
                }
              } catch {
                resultQueue.async {
                  results.append([
                    "assetId": assetId,
                    "isSensitive": false,
                    "error": error.localizedDescription
                  ])
                }
              }
              group.leave()
            }
          }
        }

        group.notify(queue: .main) {
          resultQueue.sync {
            let sensitiveCount = results.filter { ($0["isSensitive"] as? Bool) == true }.count
            resolve([
              "available": true,
              "results": results,
              "totalAnalyzed": results.count,
              "sensitiveCount": sensitiveCount
            ])
          }
        }
      }
    } else {
      resolve([
        "available": false,
        "results": [],
        "reason": "ios_version"
      ])
    }
    #else
    resolve([
      "available": false,
      "results": [],
      "reason": "framework_unavailable"
    ])
    #endif
  }

  /// Analyze a video for sensitive content using Apple's SCA
  /// Much faster than ONNX frame-by-frame analysis, returns boolean result
  @objc(analyzeVideoSensitiveContent:resolve:reject:)
  func analyzeVideoSensitiveContent(
    _ assetId: String,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    #if canImport(SensitiveContentAnalysis)
    if #available(iOS 17.0, *) {
      let analyzer = SCSensitivityAnalyzer()

      guard analyzer.analysisPolicy != .disabled else {
        resolve([
          "available": false,
          "isSensitive": false,
          "reason": "disabled_by_user"
        ])
        return
      }

      // Fetch the video asset
      guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [assetId], options: nil).firstObject else {
        reject("ASSET_NOT_FOUND", "Could not find video asset with ID: \(assetId)", nil)
        return
      }

      guard asset.mediaType == .video else {
        reject("NOT_VIDEO", "Asset is not a video", nil)
        return
      }

      // Request the video file URL
      let options = PHVideoRequestOptions()
      options.version = .current
      options.deliveryMode = .fastFormat
      options.isNetworkAccessAllowed = false

      PHImageManager.default().requestAVAsset(forVideo: asset, options: options) { avAsset, _, _ in
        guard let urlAsset = avAsset as? AVURLAsset else {
          DispatchQueue.main.async {
            reject("VIDEO_LOAD_FAILED", "Could not load video file", nil)
          }
          return
        }

        let videoURL = urlAsset.url
        let startTime = CFAbsoluteTimeGetCurrent()

        Task {
          do {
            var isSensitive = false
            let analysisHandler = analyzer.videoAnalysis(forFileAt: videoURL)

            // Process the async stream of results
            for try await result in analysisHandler.results {
              if result.isSensitive {
                isSensitive = true
                break // Short-circuit on first sensitive detection
              }
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime

            DispatchQueue.main.async {
              resolve([
                "available": true,
                "isSensitive": isSensitive,
                "assetId": assetId,
                "analysisTime": elapsed * 1000, // Convert to milliseconds
                "videoDuration": asset.duration
              ])
            }
          } catch {
            DispatchQueue.main.async {
              reject("SCA_VIDEO_ERROR", "Video analysis failed: \(error.localizedDescription)", error)
            }
          }
        }
      }
    } else {
      resolve([
        "available": false,
        "isSensitive": false,
        "reason": "ios_version"
      ])
    }
    #else
    resolve([
      "available": false,
      "isSensitive": false,
      "reason": "framework_unavailable"
    ])
    #endif
  }

  /// Batch analyze multiple videos for sensitive content using SCA
  /// Returns array of results with boolean isSensitive for each
  @objc(batchAnalyzeVideosSensitiveContent:resolve:reject:)
  func batchAnalyzeVideosSensitiveContent(
    _ assetIds: [String],
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    #if canImport(SensitiveContentAnalysis)
    if #available(iOS 17.0, *) {
      let analyzer = SCSensitivityAnalyzer()

      guard analyzer.analysisPolicy != .disabled else {
        resolve([
          "available": false,
          "results": [],
          "reason": "disabled_by_user"
        ])
        return
      }

      DispatchQueue.global(qos: .userInitiated).async {
        var results: [[String: Any]] = []
        let resultQueue = DispatchQueue(label: "com.visionml.sca.videoresults")
        let group = DispatchGroup()

        for assetId in assetIds {
          group.enter()

          guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [assetId], options: nil).firstObject,
                asset.mediaType == .video else {
            resultQueue.async {
              results.append([
                "assetId": assetId,
                "isSensitive": false,
                "error": "asset_not_found_or_not_video"
              ])
            }
            group.leave()
            continue
          }

          let options = PHVideoRequestOptions()
          options.version = .current
          options.deliveryMode = .fastFormat
          options.isNetworkAccessAllowed = false

          PHImageManager.default().requestAVAsset(forVideo: asset, options: options) { avAsset, _, _ in
            guard let urlAsset = avAsset as? AVURLAsset else {
              resultQueue.async {
                results.append([
                  "assetId": assetId,
                  "isSensitive": false,
                  "error": "video_load_failed"
                ])
              }
              group.leave()
              return
            }

            Task {
              do {
                var isSensitive = false
                let analysisHandler = analyzer.videoAnalysis(forFileAt: urlAsset.url)

                for try await result in analysisHandler.results {
                  if result.isSensitive {
                    isSensitive = true
                    break
                  }
                }

                resultQueue.async {
                  results.append([
                    "assetId": assetId,
                    "isSensitive": isSensitive,
                    "duration": asset.duration
                  ])
                }
              } catch {
                resultQueue.async {
                  results.append([
                    "assetId": assetId,
                    "isSensitive": false,
                    "error": error.localizedDescription
                  ])
                }
              }
              group.leave()
            }
          }
        }

        group.notify(queue: .main) {
          resultQueue.sync {
            let sensitiveCount = results.filter { ($0["isSensitive"] as? Bool) == true }.count
            resolve([
              "available": true,
              "results": results,
              "totalAnalyzed": results.count,
              "sensitiveCount": sensitiveCount
            ])
          }
        }
      }
    } else {
      resolve([
        "available": false,
        "results": [],
        "reason": "ios_version"
      ])
    }
    #else
    resolve([
      "available": false,
      "results": [],
      "reason": "framework_unavailable"
    ])
    #endif
  }

  @objc
  static func requiresMainQueueSetup() -> Bool {
    return false
  }
}
