package com.visionml

import android.util.Log
import kotlin.math.max
import kotlin.math.min

/**
 * YOLO v8 output parser - matches NudeNet Python implementation exactly
 * Ported from iOS YOLOParser.swift
 *
 * NudeNet uses YOLOv8 with output format:
 * - Output shape: [1, 4+numClasses, numPredictions]
 * - Box format: [cx, cy, w, h] center coordinates
 * - Class scores are direct (no objectness multiplication needed)
 */
class YOLOParser(
    private val classLabels: List<String>,
    private val inputSize: Int = 320
) {
    private val numClasses: Int = classLabels.size

    // Store debug info for last parse call
    var lastDebugInfo: Map<String, Any> = emptyMap()
        private set

    companion object {
        private const val TAG = "YOLOParser"
        private val NSFW_CLASS_INDICES = setOf(2, 3, 4, 6, 14)
    }

    /**
     * Parse YOLO v8 output tensor to detections
     * Matches NudeNet Python postprocessing exactly
     *
     * @param output ONNX output tensor data as flat Float array [1, 22, numPredictions]
     * @param confidenceThreshold Minimum confidence to include
     * @param originalWidth Original image width for coordinate scaling
     * @param originalHeight Original image height for coordinate scaling
     * @return List of parsed detections (before NMS)
     */
    fun parse(
        output: FloatArray,
        confidenceThreshold: Float,
        originalWidth: Int,
        originalHeight: Int
    ): List<NMS.Detection> {
        val detections = mutableListOf<NMS.Detection>()

        // YOLOv8 output format: [1, 4+numClasses, numPredictions]
        // For NudeNet 320n: [1, 22, 6300] where 22 = 4 box coords + 18 classes
        val valuesPerPrediction = 4 + numClasses  // 22 for NudeNet

        // Calculate number of predictions from output size
        val numPredictions = output.size / valuesPerPrediction

        Log.d(TAG, "Output size: ${output.size}, valuesPerPrediction: $valuesPerPrediction, numPredictions: $numPredictions")

        // NudeNet uses letterboxing: pad to square (right/bottom), then resize to inputSize
        // Image is at TOP-LEFT of padded square, so NO offset subtraction needed
        val maxDim = max(originalWidth, originalHeight)
        val scale = maxDim.toFloat() / inputSize.toFloat()

        Log.d(TAG, "Original: ${originalWidth}x$originalHeight, maxDim: $maxDim, scale: $scale")

        // Track max scores per NSFW class for debugging
        val maxNSFWScores = FloatArray(5) { 0f }
        val maxNSFWPredIdx = IntArray(5) { 0 }

        // Use very low threshold for debugging - we'll filter in JS
        val effectiveThreshold = 0.01f

        var detectionCount = 0

        for (i in 0 until numPredictions) {
            // Extract box coordinates (center format)
            val cx = output[0 * numPredictions + i]
            val cy = output[1 * numPredictions + i]
            val w = output[2 * numPredictions + i]
            val h = output[3 * numPredictions + i]

            // Find best class score (YOLOv8: class score IS confidence)
            var maxClassScore = 0f
            var bestClassIdx = 0

            for (c in 0 until numClasses) {
                val classScore = output[(4 + c) * numPredictions + i]
                if (classScore > maxClassScore) {
                    maxClassScore = classScore
                    bestClassIdx = c
                }

                // Track max NSFW scores for debugging
                val nsfwIdx = when (c) {
                    2 -> 0   // BUTTOCKS_EXPOSED
                    3 -> 1   // FEMALE_BREAST_EXPOSED
                    4 -> 2   // FEMALE_GENITALIA_EXPOSED
                    6 -> 3   // ANUS_EXPOSED
                    14 -> 4  // MALE_GENITALIA_EXPOSED
                    else -> -1
                }
                if (nsfwIdx >= 0 && classScore > maxNSFWScores[nsfwIdx]) {
                    maxNSFWScores[nsfwIdx] = classScore
                    maxNSFWPredIdx[nsfwIdx] = i
                }
            }

            // Skip very low confidence (just noise)
            if (maxClassScore < effectiveThreshold) continue

            // Convert center coordinates to corner format
            var x1 = cx - w / 2
            var y1 = cy - h / 2
            var x2 = cx + w / 2
            var y2 = cy + h / 2

            // NOTE: Android Bitmap has origin at top-left (standard image coordinates)
            // No Y-axis flip needed unlike iOS CGContext which has origin at bottom-left

            // Scale from model coordinates to original image coordinates
            x1 *= scale
            y1 *= scale
            x2 *= scale
            y2 *= scale

            // Clip to original image boundaries
            x1 = max(0f, min(x1, originalWidth.toFloat()))
            y1 = max(0f, min(y1, originalHeight.toFloat()))
            x2 = max(0f, min(x2, originalWidth.toFloat()))
            y2 = max(0f, min(y2, originalHeight.toFloat()))

            // Skip boxes with zero or negative area
            val boxWidth = x2 - x1
            val boxHeight = y2 - y1
            if (boxWidth <= 0 || boxHeight <= 0) continue

            // Minimum box size filter
            val minDimension = 10f
            if (boxWidth < minDimension && boxHeight < minDimension) continue

            // Log first few detections for debugging
            if (detectionCount < 10) {
                Log.d(TAG, "Detection #$detectionCount: class=${classLabels[bestClassIdx]}($bestClassIdx), " +
                          "conf=${"%.4f".format(maxClassScore)}, box=[${"%.1f".format(x1)},${"%.1f".format(y1)},${"%.1f".format(x2)},${"%.1f".format(y2)}]")
            }

            detections.add(NMS.Detection(
                box = floatArrayOf(x1, y1, x2, y2),
                score = maxClassScore,
                classIndex = bestClassIdx,
                className = classLabels[bestClassIdx]
            ))

            detectionCount++
        }

        Log.d(TAG, "Max NSFW scores: BUTTOCKS=${"%.4f".format(maxNSFWScores[0])}, " +
                  "F_BREAST=${"%.4f".format(maxNSFWScores[1])}, F_GEN=${"%.4f".format(maxNSFWScores[2])}, " +
                  "ANUS=${"%.4f".format(maxNSFWScores[3])}, M_GEN=${"%.4f".format(maxNSFWScores[4])}")

        Log.d(TAG, "Parsed ${detections.size} detections before NMS")

        // Store debug info
        lastDebugInfo = mapOf(
            "maxNSFWScores" to mapOf(
                "BUTTOCKS_EXPOSED" to maxNSFWScores[0],
                "FEMALE_BREAST_EXPOSED" to maxNSFWScores[1],
                "FEMALE_GENITALIA_EXPOSED" to maxNSFWScores[2],
                "ANUS_EXPOSED" to maxNSFWScores[3],
                "MALE_GENITALIA_EXPOSED" to maxNSFWScores[4]
            ),
            "numPredictions" to numPredictions,
            "detectionsBeforeNMS" to detections.size,
            "nativeOriginalWidth" to originalWidth,
            "nativeOriginalHeight" to originalHeight,
            "nativeMaxDim" to maxDim,
            "nativeScaleFactor" to scale,
            "letterboxStyle" to "right-bottom"
        )

        return detections
    }

    /**
     * Parse ALL classes above threshold, not just best class per prediction
     * This catches NSFW even when face scores higher
     */
    fun parseAllClasses(
        output: FloatArray,
        confidenceThreshold: Float,
        originalWidth: Int,
        originalHeight: Int
    ): List<NMS.Detection> {
        val detections = mutableListOf<NMS.Detection>()

        val valuesPerPrediction = 4 + numClasses
        val numPredictions = output.size / valuesPerPrediction

        val maxDim = max(originalWidth, originalHeight)
        val scale = maxDim.toFloat() / inputSize.toFloat()

        val minDimension = 10f

        for (i in 0 until numPredictions) {
            val cx = output[0 * numPredictions + i]
            val cy = output[1 * numPredictions + i]
            val w = output[2 * numPredictions + i]
            val h = output[3 * numPredictions + i]

            // Check EVERY class, not just the best one
            for (c in 0 until numClasses) {
                val classScore = output[(4 + c) * numPredictions + i]

                if (classScore < confidenceThreshold) continue

                // Convert to corner format
                var x1 = (cx - w / 2) * scale
                var y1 = (cy - h / 2) * scale
                var x2 = (cx + w / 2) * scale
                var y2 = (cy + h / 2) * scale

                // Clip to original image boundaries
                x1 = max(0f, min(x1, originalWidth.toFloat()))
                y1 = max(0f, min(y1, originalHeight.toFloat()))
                x2 = max(0f, min(x2, originalWidth.toFloat()))
                y2 = max(0f, min(y2, originalHeight.toFloat()))

                val boxWidth = x2 - x1
                val boxHeight = y2 - y1
                if (boxWidth < minDimension && boxHeight < minDimension) continue
                if (boxWidth <= 0 || boxHeight <= 0) continue

                detections.add(NMS.Detection(
                    box = floatArrayOf(x1, y1, x2, y2),
                    score = classScore,
                    classIndex = c,
                    className = classLabels[c]
                ))
            }
        }

        Log.d(TAG, "parseAllClasses found ${detections.size} detections")
        return detections
    }
}
