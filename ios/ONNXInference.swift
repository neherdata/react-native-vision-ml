import Foundation
import onnxruntime_objc

/// ONNX Runtime inference wrapper with integrated preprocessing and postprocessing
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
  }

  private var session: ORTSession?
  private var environment: ORTEnv?
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
      // Create ONNX Runtime environment
      environment = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)

      // Create session options with CoreML execution provider
      let options = try ORTSessionOptions()
      try options.appendCoreMLExecutionProvider(with: [:])

      // Create inference session
      session = try ORTSession(env: environment!, modelPath: modelPath, sessionOptions: options)

      NSLog("[ONNXInference] ✓ Model loaded successfully")
    } catch {
      NSLog("[ONNXInference] ERROR: Failed to load model: %@", error.localizedDescription)
      throw InferenceError.sessionCreationFailed
    }
  }

  /// Run full inference pipeline: decode → inference → parse → NMS
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

    guard let session = session else {
      throw InferenceError.modelNotLoaded
    }

    // Step 1: Decode and resize image
    NSLog("[ONNXInference] Step 1: Decoding image...")
    let decoded = try ImageDecoder.decode(imageUri: imageUri, targetSize: inputSize)
    let originalWidth = decoded.width
    let originalHeight = decoded.height

    // Step 2: Convert to NCHW format for ONNX
    NSLog("[ONNXInference] Step 2: Converting to NCHW tensor format...")
    let tensorData = convertToNCHW(hwcData: decoded.data, width: inputSize, height: inputSize)

    // Step 3: Create ONNX tensor
    NSLog("[ONNXInference] Step 3: Creating ONNX tensor...")
    let shape: [NSNumber] = [1, 3, NSNumber(value: inputSize), NSNumber(value: inputSize)]
    let inputName = try session.inputNames()[0]

    guard let tensor = try? ORTValue(
      tensorData: NSMutableData(bytes: tensorData, length: tensorData.count * MemoryLayout<Float>.size),
      elementType: ORTTensorElementDataType.float,
      shape: shape
    ) else {
      throw InferenceError.tensorCreationFailed
    }

    // Step 4: Run inference
    NSLog("[ONNXInference] Step 4: Running ONNX inference...")
    let inferenceStart = Date()

    guard let outputs = try? session.run(
      withInputs: [inputName: tensor],
      outputNames: try session.outputNames(),
      runOptions: nil
    ) else {
      throw InferenceError.inferenceFailedNot("Session run failed")
    }

    let inferenceTime = Int(Date().timeIntervalSince(inferenceStart) * 1000)
    NSLog("[ONNXInference] ✓ Inference complete in %dms", inferenceTime)

    // Step 5: Extract output tensor
    let outputName = try session.outputNames()[0]
    guard let outputTensor = outputs[outputName] else {
      throw InferenceError.invalidOutput
    }

    // Step 6: Parse YOLO output
    NSLog("[ONNXInference] Step 5: Parsing YOLO output...")
    let postProcessStart = Date()

    guard let tensorData = try? outputTensor.tensorData() as Data else {
      throw InferenceError.invalidOutput
    }

    let floatArray = tensorData.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [Float] in
      let buffer = pointer.bindMemory(to: Float.self)
      return Array(buffer)
    }

    let rawDetections = parser.parse(
      output: floatArray,
      confidenceThreshold: confidenceThreshold,
      originalWidth: originalWidth,
      originalHeight: originalHeight
    )

    // Step 7: Apply NMS
    NSLog("[ONNXInference] Step 6: Applying NMS...")
    let finalDetections = NMS.apply(detections: rawDetections, iouThreshold: iouThreshold)

    let postProcessTime = Int(Date().timeIntervalSince(postProcessStart) * 1000)
    let totalTime = Int(Date().timeIntervalSince(totalStart) * 1000)

    NSLog("[ONNXInference] ✓ Complete: %d detections in %dms total (inference: %dms, post-process: %dms)",
          finalDetections.count, totalTime, inferenceTime, postProcessTime)

    return InferenceResult(
      detections: finalDetections,
      inferenceTime: inferenceTime,
      postProcessTime: postProcessTime,
      totalTime: totalTime
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
    session = nil
    environment = nil
    NSLog("[ONNXInference] Session disposed")
  }
}
