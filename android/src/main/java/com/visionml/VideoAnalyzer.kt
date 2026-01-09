package com.visionml

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import kotlin.math.max
import kotlin.math.round

/**
 * Video analyzer for NSFW content detection
 * Ported from iOS VideoAnalyzer.swift
 *
 * Uses MediaMetadataRetriever for frame extraction + ONNXInference for detection
 */
class VideoAnalyzer(
    private val context: Context,
    private val detector: ONNXInference,
    private val inputSize: Int = 640
) {
    companion object {
        private const val TAG = "VideoAnalyzer"
        private val NSFW_CLASS_INDICES = setOf(2, 3, 4, 6, 14)
    }

    enum class ScanMode(val value: String) {
        /** Quick check - just check beginning, middle, and end (3 frames) */
        QUICK_CHECK("quick_check"),
        /** Sampled scan - check at regular intervals (e.g., every 5 seconds) */
        SAMPLED("sampled"),
        /** Thorough scan - use ML Kit to find frames with humans, then ONNX those */
        THOROUGH("thorough"),
        /** Binary search - start at middle, expand outward to find NSFW regions */
        BINARY_SEARCH("binary_search"),
        /** Full scan with short-circuit - check every N seconds until first detection */
        FULL_SHORT_CIRCUIT("full_short_circuit");

        companion object {
            fun fromString(value: String): ScanMode {
                return values().find { it.value == value } ?: SAMPLED
            }
        }
    }

    /** Result from analyzing a single frame */
    data class FrameAnalysisResult(
        val timestamp: Double,  // seconds
        val isNSFW: Boolean,
        val confidence: Float,
        val detections: List<NMS.Detection>,
        val processingTime: Int  // ms
    )

    /** Result from analyzing entire video */
    data class VideoAnalysisResult(
        val isNSFW: Boolean,
        val nsfwFrameCount: Int,
        val totalFramesAnalyzed: Int,
        val firstNSFWTimestamp: Double?,
        val nsfwTimestamps: List<Double>,
        val highestConfidence: Float,
        val totalProcessingTime: Int,  // ms
        val videoDuration: Double,  // seconds
        val scanMode: String,
        val humanFramesDetected: Int
    )

    /** Delegate for progress updates */
    interface Delegate {
        fun onProgressUpdate(progress: Float)
        fun onNSFWFound(timestamp: Double, confidence: Float)
        fun onComplete(result: VideoAnalysisResult)
    }

    var delegate: Delegate? = null
    private var isCancelled = false

    // Configuration
    private var sampleInterval = 5.0
    private val quickCheckPoints = listOf(0.0, 0.5, 1.0)  // start, middle, end (as ratio)

    fun cancel() {
        isCancelled = true
    }

    /**
     * Analyze a video from content URI or media ID
     */
    fun analyzeVideo(
        assetId: String,
        mode: ScanMode,
        sampleInterval: Double = 5.0,
        confidenceThreshold: Float = 0.6f
    ): VideoAnalysisResult {
        isCancelled = false
        this.sampleInterval = sampleInterval

        val retriever = MediaMetadataRetriever()

        try {
            // Try to open the video
            val uri = resolveAssetId(assetId)
            retriever.setDataSource(context, uri)

            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L
            val duration = durationMs / 1000.0

            Log.d(TAG, "Video: ${duration}s, mode: ${mode.value}, interval: ${this.sampleInterval}s")

            return when (mode) {
                ScanMode.QUICK_CHECK -> analyzeQuickCheck(retriever, duration, confidenceThreshold)
                ScanMode.SAMPLED, ScanMode.FULL_SHORT_CIRCUIT -> analyzeSampled(retriever, duration, mode, confidenceThreshold)
                ScanMode.THOROUGH -> analyzeThorough(retriever, duration, confidenceThreshold)
                ScanMode.BINARY_SEARCH -> analyzeBinarySearch(retriever, duration, confidenceThreshold)
            }
        } finally {
            retriever.release()
        }
    }

    /**
     * Resolve asset ID to content URI
     * Supports:
     * - content:// URIs directly
     * - MediaStore IDs (numeric)
     * - file:// URIs
     */
    private fun resolveAssetId(assetId: String): Uri {
        return when {
            assetId.startsWith("content://") -> Uri.parse(assetId)
            assetId.startsWith("file://") -> Uri.parse(assetId)
            assetId.all { it.isDigit() } -> {
                // Numeric ID - assume MediaStore video ID
                Uri.withAppendedPath(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, assetId)
            }
            else -> Uri.parse(assetId)
        }
    }

    // MARK: - Quick Check (3 frames)

    private fun analyzeQuickCheck(
        retriever: MediaMetadataRetriever,
        duration: Double,
        confidenceThreshold: Float
    ): VideoAnalysisResult {
        val startTime = System.currentTimeMillis()
        val results = mutableListOf<FrameAnalysisResult>()

        val timestamps = quickCheckPoints.map { it * max(0.1, duration - 0.1) }

        for ((index, timestamp) in timestamps.withIndex()) {
            if (isCancelled) break

            delegate?.onProgressUpdate(index.toFloat() / timestamps.size)

            val result = analyzeFrameAtTime(retriever, timestamp, confidenceThreshold)
            if (result != null) {
                results.add(result)
                if (result.isNSFW) {
                    delegate?.onNSFWFound(timestamp, result.confidence)
                }
            }
        }

        return buildResult(results, duration, ScanMode.QUICK_CHECK, startTime, 0)
    }

    // MARK: - Sampled Analysis

    private fun analyzeSampled(
        retriever: MediaMetadataRetriever,
        duration: Double,
        mode: ScanMode,
        confidenceThreshold: Float
    ): VideoAnalysisResult {
        val startTime = System.currentTimeMillis()
        val results = mutableListOf<FrameAnalysisResult>()

        // Generate sample timestamps
        val timestamps = mutableListOf<Double>()
        var t = 0.0
        while (t < duration) {
            timestamps.add(t)
            t += sampleInterval
        }
        // Include near-end frame
        if ((timestamps.lastOrNull() ?: 0.0) < duration - 1.0) {
            timestamps.add(max(0.0, duration - 0.5))
        }

        Log.d(TAG, "Sampling ${timestamps.size} frames at ${sampleInterval}s intervals")

        for ((index, timestamp) in timestamps.withIndex()) {
            if (isCancelled) break

            delegate?.onProgressUpdate(index.toFloat() / timestamps.size)

            val result = analyzeFrameAtTime(retriever, timestamp, confidenceThreshold)
            if (result != null) {
                results.add(result)

                if (result.isNSFW) {
                    delegate?.onNSFWFound(timestamp, result.confidence)

                    // Short-circuit if requested
                    if (mode == ScanMode.FULL_SHORT_CIRCUIT) {
                        Log.d(TAG, "NSFW at ${timestamp}s, short-circuiting")
                        break
                    }
                }
            }
        }

        return buildResult(results, duration, mode, startTime, 0)
    }

    // MARK: - Thorough Analysis (ML Kit human detection + ONNX)

    private fun analyzeThorough(
        retriever: MediaMetadataRetriever,
        duration: Double,
        confidenceThreshold: Float
    ): VideoAnalysisResult {
        val startTime = System.currentTimeMillis()

        // Step 1: Use ML Kit to find frames with humans (if available)
        // For now, fall back to sampled mode since ML Kit requires additional setup
        Log.d(TAG, "Phase 1: Scanning for humans (falling back to sampled)...")

        // TODO: Integrate MLKitAnalyzer for human detection
        // For now, analyze all sampled frames
        val timestamps = mutableListOf<Double>()
        var t = 0.0
        while (t < duration) {
            timestamps.add(t)
            t += sampleInterval
        }

        Log.d(TAG, "Analyzing ${timestamps.size} frames...")

        val results = mutableListOf<FrameAnalysisResult>()

        for ((index, timestamp) in timestamps.withIndex()) {
            if (isCancelled) break

            val progress = index.toFloat() / timestamps.size
            delegate?.onProgressUpdate(progress)

            val result = analyzeFrameAtTime(retriever, timestamp, confidenceThreshold)
            if (result != null) {
                results.add(result)
                if (result.isNSFW) {
                    delegate?.onNSFWFound(timestamp, result.confidence)
                }
            }
        }

        return buildResult(results, duration, ScanMode.THOROUGH, startTime, timestamps.size)
    }

    // MARK: - Binary Search

    private fun analyzeBinarySearch(
        retriever: MediaMetadataRetriever,
        duration: Double,
        confidenceThreshold: Float
    ): VideoAnalysisResult {
        val startTime = System.currentTimeMillis()
        val analyzedTimestamps = mutableSetOf<Double>()
        val results = mutableListOf<FrameAnalysisResult>()

        val binarySearchWindow = 5.0
        val binarySearchDepth = 3

        // Start with middle ± window
        val middle = duration / 2.0
        var queue = mutableListOf<Double>()
        var offset = -binarySearchWindow
        while (offset <= binarySearchWindow) {
            val t = middle + offset
            if (t >= 0 && t <= duration) {
                queue.add(t)
            }
            offset += 1.0
        }

        var depth = 0
        while (depth < binarySearchDepth && queue.isNotEmpty()) {
            if (isCancelled) break

            val nextQueue = mutableListOf<Double>()

            for (timestamp in queue) {
                val roundedTime = round(timestamp * 2) / 2
                if (roundedTime in analyzedTimestamps) continue
                analyzedTimestamps.add(roundedTime)

                delegate?.onProgressUpdate(analyzedTimestamps.size.toFloat() / 50f.coerceAtMost((duration / 2).toFloat()))

                val result = analyzeFrameAtTime(retriever, timestamp, confidenceThreshold)
                if (result != null) {
                    results.add(result)

                    if (result.isNSFW) {
                        delegate?.onNSFWFound(timestamp, result.confidence)
                        Log.d(TAG, "Binary search: NSFW at ${timestamp}s, expanding")

                        val before = timestamp - binarySearchWindow
                        val after = timestamp + binarySearchWindow
                        if (before >= 0) nextQueue.add(before)
                        if (after <= duration) nextQueue.add(after)
                    }
                }
            }

            queue = nextQueue
            depth++
        }

        return buildResult(results, duration, ScanMode.BINARY_SEARCH, startTime, 0)
    }

    // MARK: - Helper Methods

    private fun analyzeFrameAtTime(
        retriever: MediaMetadataRetriever,
        timestamp: Double,
        confidenceThreshold: Float
    ): FrameAnalysisResult? {
        val frameStart = System.currentTimeMillis()
        val timestampUs = (timestamp * 1_000_000).toLong()  // Convert to microseconds

        val bitmap = retriever.getFrameAtTime(timestampUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            ?: return null

        return try {
            val result = detector.detectBitmap(bitmap, confidenceThreshold, 0.45f)

            val nsfwDetections = result.detections.filter { it.classIndex in NSFW_CLASS_INDICES }
            val processingTime = (System.currentTimeMillis() - frameStart).toInt()

            FrameAnalysisResult(
                timestamp = timestamp,
                isNSFW = nsfwDetections.isNotEmpty(),
                confidence = nsfwDetections.maxOfOrNull { it.score } ?: 0f,
                detections = result.detections,
                processingTime = processingTime
            )
        } finally {
            bitmap.recycle()
        }
    }

    private fun buildResult(
        results: List<FrameAnalysisResult>,
        duration: Double,
        mode: ScanMode,
        startTime: Long,
        humanFrames: Int
    ): VideoAnalysisResult {
        val nsfwFrames = results.filter { it.isNSFW }
        val totalTime = (System.currentTimeMillis() - startTime).toInt()

        val result = VideoAnalysisResult(
            isNSFW = nsfwFrames.isNotEmpty(),
            nsfwFrameCount = nsfwFrames.size,
            totalFramesAnalyzed = results.size,
            firstNSFWTimestamp = nsfwFrames.firstOrNull()?.timestamp,
            nsfwTimestamps = nsfwFrames.map { it.timestamp }.sorted(),
            highestConfidence = results.maxOfOrNull { it.confidence } ?: 0f,
            totalProcessingTime = totalTime,
            videoDuration = duration,
            scanMode = mode.value,
            humanFramesDetected = humanFrames
        )

        delegate?.onComplete(result)
        delegate?.onProgressUpdate(1f)

        return result
    }
}
