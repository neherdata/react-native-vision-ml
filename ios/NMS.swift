import Foundation
import Accelerate

/// Non-Maximum Suppression using Accelerate framework for performance
class NMS {

  struct Detection {
    let box: [Float]  // [x1, y1, x2, y2]
    let score: Float
    let classIndex: Int
    let className: String
  }

  /// Apply NMS to filter overlapping detections
  /// - Parameters:
  ///   - detections: Array of detections to filter
  ///   - iouThreshold: IoU threshold (default 0.45)
  /// - Returns: Filtered array of detections
  static func apply(detections: [Detection], iouThreshold: Float = 0.45) -> [Detection] {
    guard !detections.isEmpty else { return [] }

    // Sort by confidence score (highest first)
    let sorted = detections.sorted { $0.score > $1.score }
    var keep: [Detection] = []
    var suppressed = Set<Int>()

    for (i, detection) in sorted.enumerated() {
      if suppressed.contains(i) { continue }
      keep.append(detection)

      // Suppress overlapping detections
      for (j, other) in sorted.enumerated() {
        if j <= i || suppressed.contains(j) { continue }

        let iou = calculateIoU(box1: detection.box, box2: other.box)
        if iou > iouThreshold {
          suppressed.insert(j)
        }
      }
    }

    return keep
  }

  /// Calculate Intersection over Union between two boxes
  /// - Parameters:
  ///   - box1: First box [x1, y1, x2, y2]
  ///   - box2: Second box [x1, y1, x2, y2]
  /// - Returns: IoU value (0.0 to 1.0)
  static func calculateIoU(box1: [Float], box2: [Float]) -> Float {
    let x1_1 = box1[0]
    let y1_1 = box1[1]
    let x2_1 = box1[2]
    let y2_1 = box1[3]

    let x1_2 = box2[0]
    let y1_2 = box2[1]
    let x2_2 = box2[2]
    let y2_2 = box2[3]

    // Calculate intersection
    let x1_i = max(x1_1, x1_2)
    let y1_i = max(y1_1, y1_2)
    let x2_i = min(x2_1, x2_2)
    let y2_i = min(y2_1, y2_2)

    let intersectionWidth = max(0, x2_i - x1_i)
    let intersectionHeight = max(0, y2_i - y1_i)
    let intersection = intersectionWidth * intersectionHeight

    // Calculate union
    let area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    let area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    let union = area1 + area2 - intersection

    return union > 0 ? intersection / union : 0
  }
}
