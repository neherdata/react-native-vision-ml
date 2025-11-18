import Foundation
import React

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

  @objc
  static func requiresMainQueueSetup() -> Bool {
    return false
  }
}
