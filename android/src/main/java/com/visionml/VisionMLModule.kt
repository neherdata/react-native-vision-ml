package com.visionml

import android.util.Log
import com.facebook.react.bridge.*
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * React Native module for VisionML
 * Provides ONNX inference, video analysis, and ML Kit features for Android
 *
 * Ported from iOS VisionMLModule.swift
 */
class VisionMLModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        private const val TAG = "VisionMLModule"
    }

    // Detector management
    private val detectors = ConcurrentHashMap<String, ONNXInference>()
    private val detectorIdCounter = AtomicInteger(0)

    // ML Kit analyzer
    private var mlKitAnalyzer: MLKitAnalyzer? = null

    // Coroutine scope for async operations
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    override fun getName(): String = "VisionML"

    private fun getMLKitAnalyzer(): MLKitAnalyzer {
        return mlKitAnalyzer ?: MLKitAnalyzer(reactContext).also { mlKitAnalyzer = it }
    }

    // MARK: - Detector Management

    @ReactMethod
    fun createDetector(
        modelPath: String,
        classLabels: ReadableArray,
        inputSize: Int,
        promise: Promise
    ) {
        scope.launch {
            try {
                // Convert ReadableArray to List<String>
                val labels = mutableListOf<String>()
                for (i in 0 until classLabels.size()) {
                    labels.add(classLabels.getString(i) ?: "")
                }

                // Create inference instance
                val inference = ONNXInference(
                    context = reactContext,
                    classLabels = labels,
                    inputSize = inputSize
                )

                // Load model
                inference.loadModel(modelPath)

                // Generate unique ID
                val detectorId = "detector_${detectorIdCounter.incrementAndGet()}"

                // Store detector
                detectors[detectorId] = inference

                val result = Arguments.createMap().apply {
                    putString("detectorId", detectorId)
                    putBoolean("success", true)
                    putString("message", "Detector created successfully")
                }

                promise.resolve(result)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to create detector: ${e.message}")
                promise.reject("DETECTOR_CREATE_ERROR", "Failed to create detector: ${e.message}", e)
            }
        }
    }

    @ReactMethod
    fun detect(
        detectorId: String,
        imageUri: String,
        confidenceThreshold: Double,
        iouThreshold: Double,
        promise: Promise
    ) {
        scope.launch {
            try {
                val detector = detectors[detectorId]
                    ?: throw IllegalArgumentException("Detector with ID '$detectorId' not found")

                val result = detector.detect(
                    imageUri = imageUri,
                    confidenceThreshold = confidenceThreshold.toFloat(),
                    iouThreshold = iouThreshold.toFloat()
                )

                // Convert detections to React Native format
                val detectionsArray = Arguments.createArray()
                for (detection in result.detections) {
                    val detectionMap = Arguments.createMap().apply {
                        // Box array
                        val boxArray = Arguments.createArray()
                        for (coord in detection.box) {
                            boxArray.pushDouble(coord.toDouble())
                        }
                        putArray("box", boxArray)
                        putDouble("score", detection.score.toDouble())
                        putInt("classIndex", detection.classIndex)
                        putString("className", detection.className)
                    }
                    detectionsArray.pushMap(detectionMap)
                }

                // Convert debug info
                val debugInfoMap = Arguments.createMap()
                for ((key, value) in result.debugInfo) {
                    when (value) {
                        is Number -> debugInfoMap.putDouble(key, value.toDouble())
                        is String -> debugInfoMap.putString(key, value)
                        is Boolean -> debugInfoMap.putBoolean(key, value)
                        is Map<*, *> -> {
                            val nestedMap = Arguments.createMap()
                            @Suppress("UNCHECKED_CAST")
                            for ((k, v) in value as Map<String, Any>) {
                                when (v) {
                                    is Number -> nestedMap.putDouble(k, v.toDouble())
                                    is String -> nestedMap.putString(k, v)
                                }
                            }
                            debugInfoMap.putMap(key, nestedMap)
                        }
                    }
                }

                val resultMap = Arguments.createMap().apply {
                    putArray("detections", detectionsArray)
                    putInt("inferenceTime", result.inferenceTime)
                    putInt("postProcessTime", result.postProcessTime)
                    putInt("totalTime", result.totalTime)
                    putMap("debugInfo", debugInfoMap)
                }

                promise.resolve(resultMap)
            } catch (e: Exception) {
                Log.e(TAG, "Inference failed: ${e.message}")
                promise.reject("INFERENCE_ERROR", "Inference failed: ${e.message}", e)
            }
        }
    }

    @ReactMethod
    fun disposeDetector(detectorId: String, promise: Promise) {
        val detector = detectors.remove(detectorId)
        if (detector != null) {
            detector.dispose()
            promise.resolve(Arguments.createMap().apply {
                putBoolean("success", true)
                putString("message", "Detector disposed")
            })
        } else {
            promise.reject("DETECTOR_NOT_FOUND", "Detector with ID '$detectorId' not found")
        }
    }

    @ReactMethod
    fun disposeAllDetectors(promise: Promise) {
        for ((_, detector) in detectors) {
            detector.dispose()
        }
        detectors.clear()
        promise.resolve(Arguments.createMap().apply {
            putBoolean("success", true)
            putString("message", "All detectors disposed")
        })
    }

    // MARK: - Video Analysis

    @ReactMethod
    fun analyzeVideo(
        detectorId: String,
        assetId: String,
        mode: String,
        sampleInterval: Double,
        confidenceThreshold: Double,
        promise: Promise
    ) {
        scope.launch {
            try {
                val detector = detectors[detectorId]
                    ?: throw IllegalArgumentException("Detector with ID '$detectorId' not found")

                val scanMode = VideoAnalyzer.ScanMode.fromString(mode)

                val analyzer = VideoAnalyzer(
                    context = reactContext,
                    detector = detector,
                    inputSize = 640
                )

                val result = analyzer.analyzeVideo(
                    assetId = assetId,
                    mode = scanMode,
                    sampleInterval = if (sampleInterval > 0) sampleInterval else 5.0,
                    confidenceThreshold = confidenceThreshold.toFloat()
                )

                // Convert timestamps to array
                val timestampsArray = Arguments.createArray()
                for (ts in result.nsfwTimestamps) {
                    timestampsArray.pushDouble(ts)
                }

                val resultMap = Arguments.createMap().apply {
                    putBoolean("isNSFW", result.isNSFW)
                    putInt("nsfwFrameCount", result.nsfwFrameCount)
                    putInt("totalFramesAnalyzed", result.totalFramesAnalyzed)
                    if (result.firstNSFWTimestamp != null) {
                        putDouble("firstNSFWTimestamp", result.firstNSFWTimestamp)
                    } else {
                        putNull("firstNSFWTimestamp")
                    }
                    putArray("nsfwTimestamps", timestampsArray)
                    putDouble("highestConfidence", result.highestConfidence.toDouble())
                    putInt("totalProcessingTime", result.totalProcessingTime)
                    putDouble("videoDuration", result.videoDuration)
                    putString("scanMode", result.scanMode)
                    putInt("humanFramesDetected", result.humanFramesDetected)
                }

                promise.resolve(resultMap)
            } catch (e: Exception) {
                Log.e(TAG, "Video analysis failed: ${e.message}")
                promise.reject("VIDEO_ANALYSIS_ERROR", "Video analysis failed: ${e.message}", e)
            }
        }
    }

    @ReactMethod
    fun quickCheckVideo(
        detectorId: String,
        assetId: String,
        confidenceThreshold: Double,
        promise: Promise
    ) {
        analyzeVideo(
            detectorId = detectorId,
            assetId = assetId,
            mode = "quick_check",
            sampleInterval = 0.0,
            confidenceThreshold = confidenceThreshold,
            promise = promise
        )
    }

    // MARK: - ML Kit Analysis (Vision Framework equivalent)

    @ReactMethod
    fun analyzeAnimals(assetId: String, promise: Promise) {
        scope.launch {
            try {
                val result = getMLKitAnalyzer().analyzeAnimals(assetId)
                promise.resolve(convertMapToWritableMap(result))
            } catch (e: Exception) {
                Log.e(TAG, "Animal analysis failed: ${e.message}")
                promise.reject("ML_KIT_ERROR", "Animal analysis failed: ${e.message}", e)
            }
        }
    }

    @ReactMethod
    fun analyzeHumanPose(assetId: String, promise: Promise) {
        scope.launch {
            try {
                val result = getMLKitAnalyzer().analyzeHumanPose(assetId)
                promise.resolve(convertMapToWritableMap(result))
            } catch (e: Exception) {
                Log.e(TAG, "Pose analysis failed: ${e.message}")
                promise.reject("ML_KIT_ERROR", "Pose analysis failed: ${e.message}", e)
            }
        }
    }

    @ReactMethod
    fun analyzeComprehensive(assetId: String, promise: Promise) {
        scope.launch {
            try {
                val result = getMLKitAnalyzer().analyzeComprehensive(assetId)
                promise.resolve(convertMapToWritableMap(result))
            } catch (e: Exception) {
                Log.e(TAG, "Comprehensive analysis failed: ${e.message}")
                promise.reject("ML_KIT_ERROR", "Comprehensive analysis failed: ${e.message}", e)
            }
        }
    }

    // MARK: - Progress Notification (Live Activity equivalent)

    @ReactMethod
    fun isLiveActivityAvailable(promise: Promise) {
        // Android uses notification instead of Live Activity
        // Always return true since notifications are always available
        promise.resolve(true)
    }

    @ReactMethod
    fun startVideoScanActivity(
        videoName: String,
        videoDuration: Double,
        scanMode: String,
        promise: Promise
    ) {
        // TODO: Implement foreground service with notification
        // For now, return success but no-op
        promise.resolve(Arguments.createMap().apply {
            putNull("activityId")
            putBoolean("success", true)
        })
    }

    @ReactMethod
    fun updateVideoScanActivity(
        progress: Double,
        phase: String,
        nsfwCount: Int,
        framesAnalyzed: Int,
        promise: Promise
    ) {
        // TODO: Update foreground notification
        promise.resolve(true)
    }

    @ReactMethod
    fun endVideoScanActivity(
        nsfwCount: Int,
        framesAnalyzed: Int,
        isNSFW: Boolean,
        promise: Promise
    ) {
        // TODO: End foreground service
        promise.resolve(true)
    }

    // MARK: - Utility Methods

    @Suppress("UNCHECKED_CAST")
    private fun convertMapToWritableMap(map: Map<String, Any>): WritableMap {
        val writableMap = Arguments.createMap()

        for ((key, value) in map) {
            when (value) {
                is Boolean -> writableMap.putBoolean(key, value)
                is Int -> writableMap.putInt(key, value)
                is Double -> writableMap.putDouble(key, value)
                is Float -> writableMap.putDouble(key, value.toDouble())
                is String -> writableMap.putString(key, value)
                is List<*> -> {
                    val array = Arguments.createArray()
                    for (item in value) {
                        when (item) {
                            is Map<*, *> -> array.pushMap(convertMapToWritableMap(item as Map<String, Any>))
                            is String -> array.pushString(item)
                            is Int -> array.pushInt(item)
                            is Double -> array.pushDouble(item)
                            is Float -> array.pushDouble(item.toDouble())
                            is Boolean -> array.pushBoolean(item)
                        }
                    }
                    writableMap.putArray(key, array)
                }
                is Map<*, *> -> {
                    writableMap.putMap(key, convertMapToWritableMap(value as Map<String, Any>))
                }
            }
        }

        return writableMap
    }

    override fun onCatalystInstanceDestroy() {
        super.onCatalystInstanceDestroy()
        // Clean up
        scope.cancel()
        for ((_, detector) in detectors) {
            detector.dispose()
        }
        detectors.clear()
        mlKitAnalyzer?.dispose()
    }
}
