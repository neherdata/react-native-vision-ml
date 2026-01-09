package com.visionml

import android.graphics.Bitmap

/**
 * ONNX Runtime inference wrapper for Android
 *
 * TODO: Implement using com.microsoft.onnxruntime:onnxruntime-android
 * See ANDROID_PLAN.md for implementation details
 */
class ONNXInference(
    private val classLabels: List<String>,
    private val inputSize: Int = 320
) {
    data class Detection(
        val box: List<Float>,      // [x1, y1, x2, y2] in original image coords
        val score: Float,
        val classIndex: Int,
        val className: String
    )

    data class InferenceResult(
        val detections: List<Detection>,
        val inferenceTime: Int,    // ms
        val postProcessTime: Int,  // ms
        val totalTime: Int,        // ms
        val debugInfo: Map<String, Any>
    )

    private var isLoaded = false

    /**
     * Load ONNX model from file path
     */
    fun loadModel(modelPath: String) {
        // TODO: Implement
        // val environment = OrtEnvironment.getEnvironment()
        // val sessionOptions = OrtSession.SessionOptions()
        // sessionOptions.addNnapi() // Enable NNAPI acceleration
        // session = environment.createSession(modelPath, sessionOptions)
        throw NotImplementedError("Android ONNX support not yet implemented")
    }

    /**
     * Run inference on an image URI
     */
    fun detect(
        imageUri: String,
        confidenceThreshold: Float = 0.6f,
        iouThreshold: Float = 0.45f
    ): InferenceResult {
        // TODO: Implement
        // 1. Load and decode image from URI
        // 2. Resize with letterbox padding
        // 3. Convert to NCHW tensor
        // 4. Run ONNX inference
        // 5. Parse YOLO output
        // 6. Apply NMS
        throw NotImplementedError("Android ONNX support not yet implemented")
    }

    /**
     * Run inference on a Bitmap directly (for video frames)
     */
    fun detectBitmap(
        bitmap: Bitmap,
        confidenceThreshold: Float = 0.6f,
        iouThreshold: Float = 0.45f
    ): InferenceResult {
        // TODO: Implement
        throw NotImplementedError("Android ONNX support not yet implemented")
    }

    /**
     * Release ONNX session resources
     */
    fun dispose() {
        // TODO: session?.close()
        isLoaded = false
    }
}
