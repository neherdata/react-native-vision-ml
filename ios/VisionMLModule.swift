import Foundation
import React
import Vision
import Photos

@objc(VisionML)
class VisionMLModule: NSObject {

  private var inference: ONNXInference?

  @objc(loadModel:classLabels:inputSize:resolve:reject:)
  func loadModel(
    _ modelPath: String,
    classLabels: [String],
    inputSize: NSNumber,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        // Create inference instance
        self.inference = ONNXInference(
          classLabels: classLabels,
          inputSize: inputSize.intValue
        )

        // Load model
        try self.inference?.loadModel(modelPath: modelPath)

        DispatchQueue.main.async {
          resolve([
            "success": true,
            "message": "Model loaded successfully"
          ])
        }
      } catch {
        DispatchQueue.main.async {
          reject("MODEL_LOAD_ERROR", "Failed to load ONNX model: \(error.localizedDescription)", error)
        }
      }
    }
  }

  @objc(detect:confidenceThreshold:iouThreshold:resolve:reject:)
  func detect(
    _ imageUri: String,
    confidenceThreshold: NSNumber,
    iouThreshold: NSNumber,
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    DispatchQueue.global(qos: .userInitiated).async {
      guard let inference = self.inference else {
        DispatchQueue.main.async {
          reject("MODEL_NOT_LOADED", "Model not loaded. Call loadModel first.", nil)
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
            "score": detection.score,
            "classIndex": detection.classIndex,
            "className": detection.className
          ] as [String: Any]
        }

        DispatchQueue.main.async {
          resolve([
            "detections": detectionsArray,
            "inferenceTime": result.inferenceTime,
            "postProcessTime": result.postProcessTime,
            "totalTime": result.totalTime
          ])
        }
      } catch {
        DispatchQueue.main.async {
          reject("INFERENCE_ERROR", "Inference failed: \(error.localizedDescription)", error)
        }
      }
    }
  }

  @objc(dispose:reject:)
  func dispose(
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    inference?.dispose()
    inference = nil
    resolve(["success": true])
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
            points[jointName.rawValue.description] = [
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

      result["scenes"] = scenes
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

      result["faces"] = faces
      result["faceCount"] = faces.count
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

      result["animals"] = animals
      result["animalCount"] = animals.count
    }

    // Human Pose Detection
    group.enter()
    let poseRequest = VNDetectHumanBodyPoseRequest { request, error in
      defer { group.leave() }
      guard error == nil,
            let observations = request.results as? [VNHumanBodyPoseObservation] else {
        return
      }

      result["humanCount"] = observations.count
      result["hasHumans"] = observations.count > 0
    }

    // Text Detection
    group.enter()
    let textRequest = VNRecognizeTextRequest { request, error in
      defer { group.leave() }
      guard error == nil,
            let observations = request.results as? [VNRecognizedTextObservation] else {
        return
      }

      result["hasText"] = observations.count > 0
      result["textRegions"] = observations.count
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
      result["rectangles"] = rectangleCount
      result["likelyScreenshot"] = rectangleCount > 5
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
      completion(result)
    }
  }

  @objc
  static func requiresMainQueueSetup() -> Bool {
    return false
  }
}
