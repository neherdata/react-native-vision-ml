import Foundation
import Accelerate

/// YOLO v8 output parser - matches NudeNet Python implementation exactly
/// Reference: https://github.com/notAI-tech/NudeNet
///
/// NudeNet uses YOLOv8 with output format:
/// - Output shape: [1, 4+numClasses, numPredictions]
/// - Box format: [cx, cy, w, h] center coordinates
/// - Class scores are direct (no objectness multiplication needed)
class YOLOParser {

  private let classLabels: [String]
  private let inputSize: Int
  private let numClasses: Int

  // Store debug info for last parse call
  var lastDebugInfo: [String: Any] = [:]

  init(classLabels: [String], inputSize: Int = 320) {
    self.classLabels = classLabels
    self.inputSize = inputSize
    self.numClasses = classLabels.count
  }

  /// Parse YOLO v8 output tensor to detections
  /// Matches NudeNet Python postprocessing exactly
  ///
  /// - Parameters:
  ///   - output: ONNX output tensor data as flat Float array [1, 22, numPredictions]
  ///   - minConfidence: Minimum confidence to include (default 0.01 for debugging)
  ///   - minBoxSize: Minimum box dimension in pixels (filters tiny noise boxes)
  ///   - originalWidth: Original image width for coordinate scaling
  ///   - originalHeight: Original image height for coordinate scaling
  /// - Returns: Array of parsed detections (before NMS)
  func parse(
    output: [Float],
    confidenceThreshold: Float,
    originalWidth: Int,
    originalHeight: Int
  ) -> [NMS.Detection] {

    var detections: [NMS.Detection] = []

    // YOLOv8 output format: [1, 4+numClasses, numPredictions]
    // For NudeNet 320n: [1, 22, 6300] where 22 = 4 box coords + 18 classes
    let valuesPerPrediction = 4 + numClasses  // 22 for NudeNet

    // Calculate number of predictions from output size
    let numPredictions = output.count / valuesPerPrediction

    NSLog("[YOLOParser] Output size: %d, valuesPerPrediction: %d, numPredictions: %d",
          output.count, valuesPerPrediction, numPredictions)

    // NudeNet uses letterboxing: pad to square (right/bottom), then resize to inputSize
    // Image is at TOP-LEFT of padded square, so NO offset subtraction needed
    // Just scale from model coords (0-320) to original image coords (0-maxDim)
    let maxDim = max(originalWidth, originalHeight)
    let scale = Float(maxDim) / Float(inputSize)

    NSLog("[YOLOParser] Original: %dx%d, maxDim: %d, scale: %.4f (no offset - image at top-left)",
          originalWidth, originalHeight, maxDim, scale)

    // Access pattern for [values, predictions] tensor stored row-major:
    // output[valueIndex * numPredictions + predictionIndex]
    // This is equivalent to Python's: np.transpose(output)[predictionIndex][valueIndex]

    // Track max scores per NSFW class for debugging
    let nsfwClassIndices = [2, 3, 4, 6, 14]
    var maxNSFWScores: [Float] = Array(repeating: 0, count: 5)
    var maxNSFWPredIdx: [Int] = Array(repeating: 0, count: 5)

    // Use very low threshold for debugging - we'll filter in JS
    let effectiveThreshold: Float = 0.01  // Catch everything

    var detectionCount = 0

    for i in 0..<numPredictions {
      // Extract box coordinates (center format)
      let cx = output[0 * numPredictions + i]
      let cy = output[1 * numPredictions + i]
      let w = output[2 * numPredictions + i]
      let h = output[3 * numPredictions + i]

      // Find best class score (YOLOv8: class score IS confidence)
      var maxClassScore: Float = 0
      var bestClassIdx = 0

      for c in 0..<numClasses {
        let classScore = output[(4 + c) * numPredictions + i]
        if classScore > maxClassScore {
          maxClassScore = classScore
          bestClassIdx = c
        }

        // Track max NSFW scores for debugging
        if let nsfwIdx = nsfwClassIndices.firstIndex(of: c) {
          if classScore > maxNSFWScores[nsfwIdx] {
            maxNSFWScores[nsfwIdx] = classScore
            maxNSFWPredIdx[nsfwIdx] = i
          }
        }
      }

      // Skip very low confidence (just noise)
      if maxClassScore < effectiveThreshold { continue }

      // Convert center coordinates to corner format (matching Python)
      // Python: x = x - w/2, y = y - h/2 (top-left)
      // We use x1,y1,x2,y2 format
      var x1 = cx - w / 2
      var y1 = cy - h / 2
      var x2 = cx + w / 2
      var y2 = cy + h / 2

      // IMPORTANT: Flip Y-axis because CGContext has origin at bottom-left
      // but the model expects origin at top-left (standard image coordinates)
      // Flip in model space (0-inputSize) before scaling
      let y1_flipped = Float(inputSize) - y2  // Note: y1 and y2 swap when flipping
      let y2_flipped = Float(inputSize) - y1

      // Scale from model coordinates to padded square (maxDim x maxDim)
      // No offset needed since image is at top-left of padded square
      x1 = x1 * scale
      y1 = y1_flipped * scale
      x2 = x2 * scale
      y2 = y2_flipped * scale

      // Clip to original image boundaries (Python does this too)
      x1 = max(0, min(x1, Float(originalWidth)))
      y1 = max(0, min(y1, Float(originalHeight)))
      x2 = max(0, min(x2, Float(originalWidth)))
      y2 = max(0, min(y2, Float(originalHeight)))

      // Skip boxes with zero or negative area
      let boxWidth = x2 - x1
      let boxHeight = y2 - y1
      if boxWidth <= 0 || boxHeight <= 0 { continue }

      // Minimum box size filter (instead of confidence threshold)
      // Skip tiny boxes that are likely noise
      let minDimension: Float = 10.0  // 10 pixels minimum
      if boxWidth < minDimension && boxHeight < minDimension { continue }

      // Log first few detections for debugging
      if detectionCount < 10 {
        NSLog("[YOLOParser] Detection #%d: class=%@(%d), conf=%.4f, box=[%.1f,%.1f,%.1f,%.1f] size=%.1fx%.1f",
              detectionCount, classLabels[bestClassIdx], bestClassIdx, maxClassScore,
              x1, y1, x2, y2, boxWidth, boxHeight)

        // Show top 5 class scores for this detection to diagnose misclassification
        var classScores: [(Int, Float)] = []
        for c in 0..<numClasses {
          let score = output[(4 + c) * numPredictions + i]
          classScores.append((c, score))
        }
        classScores.sort { $0.1 > $1.1 }
        let top5 = classScores.prefix(5).map { "\(classLabels[$0.0]):\(String(format: "%.3f", $0.1))" }.joined(separator: ", ")
        NSLog("[YOLOParser]   Top5: %@", top5)
      }

      detections.append(NMS.Detection(
        box: [x1, y1, x2, y2],
        score: maxClassScore,
        classIndex: bestClassIdx,
        className: classLabels[bestClassIdx]
      ))

      detectionCount += 1
    }

    // Log max NSFW scores for debugging
    NSLog("[YOLOParser] Max NSFW scores: BUTTOCKS=%.4f(pred %d), F_BREAST=%.4f(pred %d), F_GEN=%.4f(pred %d), ANUS=%.4f(pred %d), M_GEN=%.4f(pred %d)",
          maxNSFWScores[0], maxNSFWPredIdx[0],
          maxNSFWScores[1], maxNSFWPredIdx[1],
          maxNSFWScores[2], maxNSFWPredIdx[2],
          maxNSFWScores[3], maxNSFWPredIdx[3],
          maxNSFWScores[4], maxNSFWPredIdx[4])

    // Also check male-specific classes to diagnose gender misclassification
    // Index 5: MALE_BREAST_EXPOSED, Index 12: FACE_MALE
    var maxMaleBreast: Float = 0
    var maxFaceMale: Float = 0
    for i in 0..<numPredictions {
      let maleBreastScore = output[(4 + 5) * numPredictions + i]  // MALE_BREAST_EXPOSED
      let faceMaleScore = output[(4 + 12) * numPredictions + i]   // FACE_MALE
      maxMaleBreast = max(maxMaleBreast, maleBreastScore)
      maxFaceMale = max(maxFaceMale, faceMaleScore)
    }
    NSLog("[YOLOParser] Max MALE scores: M_BREAST=%.4f, FACE_MALE=%.4f (compare to F_BREAST=%.4f, M_GEN=%.4f)",
          maxMaleBreast, maxFaceMale, maxNSFWScores[1], maxNSFWScores[4])

    NSLog("[YOLOParser] Parsed %d detections before NMS", detections.count)

    // Store debug info for retrieval via JS bridge
    lastDebugInfo = [
      "maxNSFWScores": [
        "BUTTOCKS_EXPOSED": maxNSFWScores[0],
        "FEMALE_BREAST_EXPOSED": maxNSFWScores[1],
        "FEMALE_GENITALIA_EXPOSED": maxNSFWScores[2],
        "ANUS_EXPOSED": maxNSFWScores[3],
        "MALE_GENITALIA_EXPOSED": maxNSFWScores[4]
      ],
      "maxMaleScores": [
        "MALE_BREAST_EXPOSED": maxMaleBreast,
        "FACE_MALE": maxFaceMale
      ],
      "numPredictions": numPredictions,
      "detectionsBeforeNMS": detections.count,
      "nativeOriginalWidth": originalWidth,
      "nativeOriginalHeight": originalHeight,
      "nativeMaxDim": maxDim,
      "nativeScaleFactor": scale,
      "letterboxStyle": "right-bottom"  // Image at top-left, padding on right/bottom
    ]

    return detections
  }

  /// Parse ALL classes above threshold, not just best class per prediction
  /// This catches NSFW even when face scores higher
  func parseAllClasses(
    output: [Float],
    confidenceThreshold: Float,
    originalWidth: Int,
    originalHeight: Int
  ) -> [NMS.Detection] {

    var detections: [NMS.Detection] = []

    let valuesPerPrediction = 4 + numClasses
    let numPredictions = output.count / valuesPerPrediction

    let maxDim = max(originalWidth, originalHeight)
    let scale = Float(maxDim) / Float(inputSize)

    let minDimension: Float = 10.0

    for i in 0..<numPredictions {
      let cx = output[0 * numPredictions + i]
      let cy = output[1 * numPredictions + i]
      let w = output[2 * numPredictions + i]
      let h = output[3 * numPredictions + i]

      // Check EVERY class, not just the best one
      for c in 0..<numClasses {
        let classScore = output[(4 + c) * numPredictions + i]

        if classScore < confidenceThreshold { continue }

        // Convert to corner format
        let x1_raw = cx - w / 2
        let y1_raw = cy - h / 2
        let x2_raw = cx + w / 2
        let y2_raw = cy + h / 2

        // Flip Y-axis (CGContext bottom-left origin → standard top-left)
        let y1_flipped = Float(inputSize) - y2_raw
        let y2_flipped = Float(inputSize) - y1_raw

        // Scale to original image coordinates
        var x1 = x1_raw * scale
        var y1 = y1_flipped * scale
        var x2 = x2_raw * scale
        var y2 = y2_flipped * scale

        x1 = max(0, min(x1, Float(originalWidth)))
        y1 = max(0, min(y1, Float(originalHeight)))
        x2 = max(0, min(x2, Float(originalWidth)))
        y2 = max(0, min(y2, Float(originalHeight)))

        let boxWidth = x2 - x1
        let boxHeight = y2 - y1
        if boxWidth < minDimension && boxHeight < minDimension { continue }
        if boxWidth <= 0 || boxHeight <= 0 { continue }

        detections.append(NMS.Detection(
          box: [x1, y1, x2, y2],
          score: classScore,
          classIndex: c,
          className: classLabels[c]
        ))
      }
    }

    NSLog("[YOLOParser] parseAllClasses found %d detections", detections.count)
    return detections
  }
}
