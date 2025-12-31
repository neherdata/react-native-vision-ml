import Foundation

/// ONNX Runtime inference wrapper with integrated preprocessing and postprocessing
/// Uses ObjC wrapper to avoid Swift modular headers issue with onnxruntime-objc
class ONNXInference {

  enum InferenceError: Error {
    case modelNotLoaded
    case sessionCreationFailed
    case inferenceFailedNot(String)
    case invalidOutput
    case tensorCreationFailed
  }

  struct InferenceResult {
    let detections: [NMS.Detection]
    let inferenceTime: Int  // milliseconds
    let postProcessTime: Int  // milliseconds
    let totalTime: Int  // milliseconds
    let debugInfo: [String: Any]  // Debug info from parser
  }

  private var wrapper: ONNXWrapper?
  private let parser: YOLOParser
  private let inputSize: Int

  init(classLabels: [String], inputSize: Int = 320) {
    self.parser = YOLOParser(classLabels: classLabels, inputSize: inputSize)
    self.inputSize = inputSize
  }

  /// Load ONNX model from file path
  /// - Parameter modelPath: Path to .onnx model file
  /// - Throws: InferenceError if model loading fails
  func loadModel(modelPath: String) throws {
    NSLog("[ONNXInference] Loading model from: %@", modelPath)

    do {
      // Swift auto-bridges ObjC (NSError **) parameter to throwing function
      wrapper = try ONNXWrapper(modelPath: modelPath)

      guard wrapper?.isModelLoaded == true else {
        throw InferenceError.sessionCreationFailed
      }

      NSLog("[ONNXInference] ✓ Model loaded successfully")
    } catch {
      NSLog("[ONNXInference] ERROR: Failed to load model: %@", error.localizedDescription)
      throw InferenceError.sessionCreationFailed
    }
  }

  /// Run full inference pipeline: decode → inference → parse → NMS
  /// Matches NudeNet Python processing exactly
  ///
  /// - Parameters:
  ///   - imageUri: file:// URI to image
  ///   - confidenceThreshold: Minimum confidence score (default 0.6)
  ///   - iouThreshold: IoU threshold for NMS (default 0.45)
  /// - Returns: Inference result with filtered detections and timing info
  /// - Throws: InferenceError or ImageDecoder.DecodeError
  func detect(
    imageUri: String,
    confidenceThreshold: Float = 0.6,
    iouThreshold: Float = 0.45
  ) throws -> InferenceResult {

    let totalStart = Date()

    guard let wrapper = wrapper, wrapper.isModelLoaded else {
      throw InferenceError.modelNotLoaded
    }

    // Step 1: Decode and resize image (with letterbox padding like NudeNet Python)
    NSLog("[ONNXInference] Step 1: Decoding image...")
    let decoded = try ImageDecoder.decode(imageUri: imageUri, targetSize: inputSize)
    let originalWidth = decoded.originalWidth
    let originalHeight = decoded.originalHeight
    NSLog("[ONNXInference] Original dimensions: %d x %d", originalWidth, originalHeight)

    // Step 2: Convert to NCHW format for ONNX
    NSLog("[ONNXInference] Step 2: Converting to NCHW tensor format...")
    let tensorData = convertToNCHW(hwcData: decoded.data, width: inputSize, height: inputSize)

    // Step 3: Run inference via ObjC wrapper
    NSLog("[ONNXInference] Step 3: Running ONNX inference...")
    let inferenceStart = Date()

    let inputData = tensorData.map { NSNumber(value: $0) }
    let inputShape: [NSNumber] = [1, 3, NSNumber(value: inputSize), NSNumber(value: inputSize)]

    let outputArray: [NSNumber]
    do {
      outputArray = try wrapper.runInference(withInputData: inputData, inputShape: inputShape)
    } catch {
      NSLog("[ONNXInference] ERROR: Inference failed: %@", error.localizedDescription)
      throw InferenceError.inferenceFailedNot(error.localizedDescription)
    }

    let inferenceTime = Int(Date().timeIntervalSince(inferenceStart) * 1000)
    NSLog("[ONNXInference] ✓ Inference complete in %dms", inferenceTime)

    // Step 4: Parse YOLO output (matches NudeNet Python _postprocess)
    NSLog("[ONNXInference] Step 4: Parsing YOLO output...")
    let postProcessStart = Date()

    let floatArray = outputArray.map { $0.floatValue }

    // Parse using standard best-class-per-prediction (like NudeNet Python)
    let rawDetections = parser.parse(
      output: floatArray,
      confidenceThreshold: confidenceThreshold,
      originalWidth: originalWidth,
      originalHeight: originalHeight
    )

    // Step 5: Apply NMS (matches NudeNet Python cv2.dnn.NMSBoxes)
    NSLog("[ONNXInference] Step 5: Applying NMS with IoU threshold %.2f...", iouThreshold)
    let filteredDetections = NMS.apply(detections: rawDetections, iouThreshold: iouThreshold)

    // Step 6: ALSO get all-class detections for NSFW that might be suppressed
    // This catches NSFW when face has higher score at same location
    NSLog("[ONNXInference] Step 6: Parsing all classes for NSFW recovery...")
    let allClassDetections = parser.parseAllClasses(
      output: floatArray,
      confidenceThreshold: confidenceThreshold,
      originalWidth: originalWidth,
      originalHeight: originalHeight
    )

    // Filter to only NSFW classes from the all-class parse
    let nsfwClassIndices = Set([2, 3, 4, 6, 14])
    let nsfwFromAllClass = allClassDetections.filter { nsfwClassIndices.contains($0.classIndex) }

    // Apply NMS to NSFW-only detections (separate from faces)
    let nsfwFiltered = NMS.apply(detections: nsfwFromAllClass, iouThreshold: iouThreshold)

    // Combine: standard detections + NSFW-specific detections
    // Remove duplicates by checking if NSFW detection already exists in filtered
    var finalDetections = filteredDetections
    for nsfwDet in nsfwFiltered {
      let isDuplicate = finalDetections.contains { existing in
        existing.classIndex == nsfwDet.classIndex &&
        abs(existing.box[0] - nsfwDet.box[0]) < 10 &&
        abs(existing.box[1] - nsfwDet.box[1]) < 10
      }
      if !isDuplicate {
        finalDetections.append(nsfwDet)
      }
    }

    NSLog("[ONNXInference] Final: %d standard + %d NSFW-recovered = %d total",
          filteredDetections.count, nsfwFiltered.count, finalDetections.count)

    let postProcessTime = Int(Date().timeIntervalSince(postProcessStart) * 1000)
    let totalTime = Int(Date().timeIntervalSince(totalStart) * 1000)

    NSLog("[ONNXInference] ✓ Complete: %d detections in %dms total (inference: %dms, post-process: %dms)",
          finalDetections.count, totalTime, inferenceTime, postProcessTime)

    return InferenceResult(
      detections: finalDetections,
      inferenceTime: inferenceTime,
      postProcessTime: postProcessTime,
      totalTime: totalTime,
      debugInfo: parser.lastDebugInfo
    )
  }

  /// Convert HWC (Height, Width, Channels) to NCHW (Batch, Channels, Height, Width)
  private func convertToNCHW(hwcData: [Float], width: Int, height: Int) -> [Float] {
    var nchwData = [Float](repeating: 0, count: 3 * height * width)

    for y in 0..<height {
      for x in 0..<width {
        let hwcIndex = (y * width + x) * 3
        let baseIndex = y * width + x

        nchwData[baseIndex] = hwcData[hwcIndex]                        // R channel
        nchwData[height * width + baseIndex] = hwcData[hwcIndex + 1]   // G channel
        nchwData[2 * height * width + baseIndex] = hwcData[hwcIndex + 2]  // B channel
      }
    }

    return nchwData
  }

  /// Dispose of ONNX session and environment
  func dispose() {
    wrapper = nil
    NSLog("[ONNXInference] Session disposed")
  }
}
