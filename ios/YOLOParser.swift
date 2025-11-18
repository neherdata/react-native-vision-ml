import Foundation
import Accelerate

/// YOLO v5/v8 output parser
class YOLOParser {

  private let classLabels: [String]
  private let inputSize: Int
  private let numPredictions: Int
  private let numClasses: Int

  init(classLabels: [String], inputSize: Int = 320) {
    self.classLabels = classLabels
    self.inputSize = inputSize
    self.numPredictions = 25200  // 80x80 + 40x40 + 20x20 grid cells
    self.numClasses = classLabels.count
  }

  /// Parse YOLO output tensor to detections
  /// - Parameters:
  ///   - output: ONNX output tensor data as Float32Array
  ///   - confidenceThreshold: Minimum confidence score
  ///   - originalWidth: Original image width for coordinate scaling
  ///   - originalHeight: Original image height for coordinate scaling
  /// - Returns: Array of parsed detections
  func parse(
    output: [Float],
    confidenceThreshold: Float,
    originalWidth: Int,
    originalHeight: Int
  ) -> [NMS.Detection] {

    var detections: [NMS.Detection] = []

    // YOLO output format: [1, 25200, 85]
    // Each prediction: [x, y, w, h, objectness, class1, class2, ..., classN]
    let stride = 5 + numClasses  // 85 for COCO, varies by model

    let scaleX = Float(originalWidth) / Float(inputSize)
    let scaleY = Float(originalHeight) / Float(inputSize)

    NSLog("[YOLOParser] Parsing %d predictions with %d classes", numPredictions, numClasses)
    NSLog("[YOLOParser] Scale factors: X=%.2f, Y=%.2f", scaleX, scaleY)

    for i in 0..<numPredictions {
      let offset = i * stride

      guard offset + stride <= output.count else {
        NSLog("[YOLOParser] WARNING: Offset %d exceeds output size %d", offset, output.count)
        break
      }

      // Extract box coordinates (center format)
      let cx = output[offset]
      let cy = output[offset + 1]
      let w = output[offset + 2]
      let h = output[offset + 3]
      let objectness = output[offset + 4]

      // Skip low objectness predictions early
      if objectness < confidenceThreshold { continue }

      // Find best class
      var maxClassScore: Float = 0
      var bestClassIdx = 0

      for c in 0..<numClasses {
        let classScore = output[offset + 5 + c]
        if classScore > maxClassScore {
          maxClassScore = classScore
          bestClassIdx = c
        }
      }

      let confidence = objectness * maxClassScore
      if confidence < confidenceThreshold { continue }

      // Convert center format to corner format and scale to original image
      let x1 = (cx - w / 2) * scaleX
      let y1 = (cy - h / 2) * scaleY
      let x2 = (cx + w / 2) * scaleX
      let y2 = (cy + h / 2) * scaleY

      detections.append(NMS.Detection(
        box: [x1, y1, x2, y2],
        score: confidence,
        classIndex: bestClassIdx,
        className: classLabels[bestClassIdx]
      ))
    }

    NSLog("[YOLOParser] Parsed %d detections before NMS", detections.count)
    return detections
  }
}
