package com.visionml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer
import kotlin.math.abs

/**
 * ONNX Runtime inference wrapper for Android
 * Ported from iOS ONNXInference.swift
 *
 * Full inference pipeline: decode → resize → NCHW tensor → ONNX inference → YOLO parse → NMS
 */
class ONNXInference(
    private val context: Context,
    private val classLabels: List<String>,
    private val inputSize: Int = 320
) {
    companion object {
        private const val TAG = "ONNXInference"
        private val NSFW_CLASS_INDICES = setOf(2, 3, 4, 6, 14)
    }

    sealed class InferenceError : Exception() {
        object ModelNotLoaded : InferenceError()
        object SessionCreationFailed : InferenceError()
        data class InferenceFailed(override val message: String) : InferenceError()
        object InvalidOutput : InferenceError()
        object TensorCreationFailed : InferenceError()
    }

    data class InferenceResult(
        val detections: List<NMS.Detection>,
        val inferenceTime: Int,      // milliseconds
        val postProcessTime: Int,    // milliseconds
        val totalTime: Int,          // milliseconds
        val debugInfo: Map<String, Any>
    )

    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private val parser: YOLOParser = YOLOParser(classLabels, inputSize)
    private val imageDecoder: ImageDecoder = ImageDecoder(context)

    val isModelLoaded: Boolean
        get() = ortSession != null

    /**
     * Load ONNX model from file path
     * @param modelPath Path to .onnx model file
     */
    fun loadModel(modelPath: String) {
        Log.d(TAG, "Loading model from: $modelPath")

        try {
            ortEnvironment = OrtEnvironment.getEnvironment()

            // Create session options
            val sessionOptions = OrtSession.SessionOptions().apply {
                // Enable optimizations
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

                // Try to use NNAPI for hardware acceleration (GPU/NPU)
                // This will fall back to CPU if NNAPI is not available
                try {
                    addNnapi()
                    Log.d(TAG, "NNAPI execution provider enabled")
                } catch (e: Exception) {
                    Log.w(TAG, "NNAPI not available, using CPU: ${e.message}")
                }
            }

            ortSession = ortEnvironment?.createSession(modelPath, sessionOptions)

            if (ortSession == null) {
                throw InferenceError.SessionCreationFailed
            }

            Log.d(TAG, "✓ Model loaded successfully")

            // Log input/output info
            ortSession?.let { session ->
                Log.d(TAG, "Input names: ${session.inputNames}")
                Log.d(TAG, "Output names: ${session.outputNames}")
                session.inputInfo.forEach { (name, info) ->
                    Log.d(TAG, "Input '$name': ${info.info}")
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "ERROR: Failed to load model: ${e.message}")
            throw InferenceError.SessionCreationFailed
        }
    }

    /**
     * Run full inference pipeline: decode → inference → parse → NMS
     * Matches NudeNet Python processing exactly
     *
     * @param imageUri file:// URI to image
     * @param confidenceThreshold Minimum confidence score (default 0.6)
     * @param iouThreshold IoU threshold for NMS (default 0.45)
     * @return Inference result with filtered detections and timing info
     */
    fun detect(
        imageUri: String,
        confidenceThreshold: Float = 0.6f,
        iouThreshold: Float = 0.45f
    ): InferenceResult {
        val totalStart = System.currentTimeMillis()

        val session = ortSession ?: throw InferenceError.ModelNotLoaded
        val env = ortEnvironment ?: throw InferenceError.ModelNotLoaded

        // Step 1: Decode and resize image (with letterbox padding like NudeNet Python)
        Log.d(TAG, "Step 1: Decoding image...")
        val decoded = imageDecoder.decode(imageUri, inputSize)
        val originalWidth = decoded.originalWidth
        val originalHeight = decoded.originalHeight
        Log.d(TAG, "Original dimensions: ${originalWidth}x$originalHeight")

        // Step 2: Convert to NCHW format for ONNX
        Log.d(TAG, "Step 2: Converting to NCHW tensor format...")
        val nchwData = imageDecoder.convertToNCHW(decoded.data, inputSize, inputSize)

        // Step 3: Run inference
        Log.d(TAG, "Step 3: Running ONNX inference...")
        val inferenceStart = System.currentTimeMillis()

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(nchwData),
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )

        val outputArray: FloatArray
        try {
            val inputs = mapOf("images" to inputTensor)
            val results = session.run(inputs)

            // Get output tensor (typically named "output0" for YOLOv8)
            val outputTensor = results.get(0) as OnnxTensor
            outputArray = outputTensor.floatBuffer.array()

            results.close()
        } finally {
            inputTensor.close()
        }

        val inferenceTime = (System.currentTimeMillis() - inferenceStart).toInt()
        Log.d(TAG, "✓ Inference complete in ${inferenceTime}ms")

        // Step 4: Parse YOLO output
        Log.d(TAG, "Step 4: Parsing YOLO output...")
        val postProcessStart = System.currentTimeMillis()

        val rawDetections = parser.parse(
            output = outputArray,
            confidenceThreshold = confidenceThreshold,
            originalWidth = originalWidth,
            originalHeight = originalHeight
        )

        // Step 5: Apply NMS
        Log.d(TAG, "Step 5: Applying NMS with IoU threshold $iouThreshold...")
        val filteredDetections = NMS.apply(rawDetections, iouThreshold)

        // Step 6: Also get all-class detections for NSFW recovery
        Log.d(TAG, "Step 6: Parsing all classes for NSFW recovery...")
        val allClassDetections = parser.parseAllClasses(
            output = outputArray,
            confidenceThreshold = confidenceThreshold,
            originalWidth = originalWidth,
            originalHeight = originalHeight
        )

        // Filter to NSFW classes and apply NMS
        val nsfwFromAllClass = allClassDetections.filter { it.classIndex in NSFW_CLASS_INDICES }
        val nsfwFiltered = NMS.apply(nsfwFromAllClass, iouThreshold)

        // Combine: standard detections + NSFW-specific detections
        val finalDetections = filteredDetections.toMutableList()
        for (nsfwDet in nsfwFiltered) {
            val isDuplicate = finalDetections.any { existing ->
                existing.classIndex == nsfwDet.classIndex &&
                abs(existing.box[0] - nsfwDet.box[0]) < 10 &&
                abs(existing.box[1] - nsfwDet.box[1]) < 10
            }
            if (!isDuplicate) {
                finalDetections.add(nsfwDet)
            }
        }

        Log.d(TAG, "Final: ${filteredDetections.size} standard + ${nsfwFiltered.size} NSFW-recovered = ${finalDetections.size} total")

        val postProcessTime = (System.currentTimeMillis() - postProcessStart).toInt()
        val totalTime = (System.currentTimeMillis() - totalStart).toInt()

        Log.d(TAG, "✓ Complete: ${finalDetections.size} detections in ${totalTime}ms total " +
                  "(inference: ${inferenceTime}ms, post-process: ${postProcessTime}ms)")

        return InferenceResult(
            detections = finalDetections,
            inferenceTime = inferenceTime,
            postProcessTime = postProcessTime,
            totalTime = totalTime,
            debugInfo = parser.lastDebugInfo
        )
    }

    /**
     * Run inference directly on a Bitmap (for video frames)
     */
    fun detectBitmap(
        bitmap: Bitmap,
        confidenceThreshold: Float = 0.6f,
        iouThreshold: Float = 0.45f
    ): InferenceResult {
        val totalStart = System.currentTimeMillis()

        val session = ortSession ?: throw InferenceError.ModelNotLoaded
        val env = ortEnvironment ?: throw InferenceError.ModelNotLoaded

        // Decode bitmap directly
        val decoded = imageDecoder.decodeBitmap(bitmap, inputSize)
        val originalWidth = decoded.originalWidth
        val originalHeight = decoded.originalHeight

        // Convert to NCHW
        val nchwData = imageDecoder.convertToNCHW(decoded.data, inputSize, inputSize)

        // Run inference
        val inferenceStart = System.currentTimeMillis()

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(nchwData),
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )

        val outputArray: FloatArray
        try {
            val inputs = mapOf("images" to inputTensor)
            val results = session.run(inputs)
            val outputTensor = results.get(0) as OnnxTensor
            outputArray = outputTensor.floatBuffer.array()
            results.close()
        } finally {
            inputTensor.close()
        }

        val inferenceTime = (System.currentTimeMillis() - inferenceStart).toInt()

        // Parse and NMS
        val postProcessStart = System.currentTimeMillis()

        val rawDetections = parser.parse(outputArray, confidenceThreshold, originalWidth, originalHeight)
        val filteredDetections = NMS.apply(rawDetections, iouThreshold)

        // NSFW recovery
        val allClassDetections = parser.parseAllClasses(outputArray, confidenceThreshold, originalWidth, originalHeight)
        val nsfwFromAllClass = allClassDetections.filter { it.classIndex in NSFW_CLASS_INDICES }
        val nsfwFiltered = NMS.apply(nsfwFromAllClass, iouThreshold)

        val finalDetections = filteredDetections.toMutableList()
        for (nsfwDet in nsfwFiltered) {
            val isDuplicate = finalDetections.any { existing ->
                existing.classIndex == nsfwDet.classIndex &&
                abs(existing.box[0] - nsfwDet.box[0]) < 10 &&
                abs(existing.box[1] - nsfwDet.box[1]) < 10
            }
            if (!isDuplicate) {
                finalDetections.add(nsfwDet)
            }
        }

        val postProcessTime = (System.currentTimeMillis() - postProcessStart).toInt()
        val totalTime = (System.currentTimeMillis() - totalStart).toInt()

        return InferenceResult(
            detections = finalDetections,
            inferenceTime = inferenceTime,
            postProcessTime = postProcessTime,
            totalTime = totalTime,
            debugInfo = parser.lastDebugInfo
        )
    }

    /**
     * Release ONNX session resources
     */
    fun dispose() {
        ortSession?.close()
        ortSession = null
        // Note: OrtEnvironment is shared, don't close it
        Log.d(TAG, "Session disposed")
    }
}
