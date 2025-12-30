import Foundation
import Accelerate

/// YOLO v8 output parser (NudeNet format)
/// NudeNet uses YOLOv8 which has a different output format than v5:
/// - Output shape: [1, 4+numClasses, numPredictions] - needs transpose
/// - No objectness score - class scores directly (no sigmoid needed)
/// - Box format: [cx, cy, w, h] center coordinates
class YOLOParser {

  private let classLabels: [String]
  private let inputSize: Int
  private let numClasses: Int

  init(classLabels: [String], inputSize: Int = 320) {
    self.classLabels = classLabels
    self.inputSize = inputSize
    self.numClasses = classLabels.count
  }

  /// Parse YOLO v8 output tensor to detections
  /// - Parameters:
  ///   - output: ONNX output tensor data as flat Float array
  ///   - confidenceThreshold: Minimum confidence score (default 0.2 like NudeNet)
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

    // YOLOv8 output format: [1, 4+numClasses, numPredictions]
    // For NudeNet 320n: [1, 22, 6300] where 22 = 4 box coords + 18 classes
    // 6300 predictions = 80x80 (small) + 40x40 (medium) = 6400+1600 ≈ adjusted
    let valuesPerPrediction = 4 + numClasses  // 22 for NudeNet (4 box + 18 classes)

    // Calculate number of predictions from output size
    let numPredictions = output.count / valuesPerPrediction

    NSLog("[YOLOParser] Output size: %d, valuesPerPrediction: %d, numPredictions: %d",
          output.count, valuesPerPrediction, numPredictions)

    // DEBUG: Log first few raw output values to verify tensor format
    if output.count > 50 {
      NSLog("[YOLOParser] First 10 values (should be cx for first 10 preds): %@",
            output[0..<10].map { String(format: "%.2f", $0) }.joined(separator: ", "))
      NSLog("[YOLOParser] Values at prediction 0 [cx,cy,w,h,c0,c1,c2]: %@",
            (0..<7).map { output[$0 * numPredictions] }.map { String(format: "%.3f", $0) }.joined(separator: ", "))
    }

    // NudeNet uses letterboxing (padding to square) then resize
    // The output coordinates are relative to the padded square image
    // We need to scale back to original image dimensions

    // Calculate padding that was applied during preprocessing
    let maxDim = max(originalWidth, originalHeight)
    let xPad = maxDim - originalWidth
    let yPad = maxDim - originalHeight

    // Scale factors from model input to padded image
    let scaleX = Float(maxDim) / Float(inputSize)
    let scaleY = Float(maxDim) / Float(inputSize)

    NSLog("[YOLOParser] Original: %dx%d, maxDim: %d, padding: x=%d y=%d, scale: %.2f x %.2f",
          originalWidth, originalHeight, maxDim, xPad, yPad, scaleX, scaleY)

    // The output is in format [values, predictions] and needs to be transposed
    // to [predictions, values] for easier processing
    // Access pattern: output[valueIndex * numPredictions + predictionIndex]

    var detectionCount = 0
    for i in 0..<numPredictions {
      // Extract box coordinates (already in model input scale)
      // YOLOv8 format: [cx, cy, w, h] - center coordinates
      let cx = output[0 * numPredictions + i]
      let cy = output[1 * numPredictions + i]
      let w = output[2 * numPredictions + i]
      let h = output[3 * numPredictions + i]

      // Find best class score (no objectness in YOLOv8, scores are direct)
      var maxClassScore: Float = 0
      var bestClassIdx = 0

      for c in 0..<numClasses {
        let classScore = output[(4 + c) * numPredictions + i]
        if classScore > maxClassScore {
          maxClassScore = classScore
          bestClassIdx = c
        }
      }

      // In YOLOv8, the class score IS the confidence (no objectness multiplication)
      let confidence = maxClassScore

      // Skip low confidence predictions (NudeNet uses 0.2 threshold)
      if confidence < confidenceThreshold { continue }

      // Convert center coordinates to corner format
      var x1 = cx - w / 2
      var y1 = cy - h / 2
      var x2 = cx + w / 2
      var y2 = cy + h / 2

      // Scale from model coordinates to padded image coordinates
      x1 = x1 * scaleX
      y1 = y1 * scaleY
      x2 = x2 * scaleX
      y2 = y2 * scaleY

      // Clip to original image boundaries (remove padding effect)
      x1 = max(0, min(x1, Float(originalWidth)))
      y1 = max(0, min(y1, Float(originalHeight)))
      x2 = max(0, min(x2, Float(originalWidth)))
      y2 = max(0, min(y2, Float(originalHeight)))

      // Skip boxes with zero or negative area
      if x2 <= x1 || y2 <= y1 { continue }

      // DEBUG: Log first few detections
      if detectionCount < 5 {
        NSLog("[YOLOParser] Detection #%d: class=%@, confidence=%.3f, box=[%.1f,%.1f,%.1f,%.1f]",
              detectionCount, classLabels[bestClassIdx], confidence, x1, y1, x2, y2)
      }

      detections.append(NMS.Detection(
        box: [x1, y1, x2, y2],
        score: confidence,
        classIndex: bestClassIdx,
        className: classLabels[bestClassIdx]
      ))

      detectionCount += 1
    }

    NSLog("[YOLOParser] Parsed %d detections before NMS (threshold: %.2f)", detections.count, confidenceThreshold)
    return detections
  }
}
