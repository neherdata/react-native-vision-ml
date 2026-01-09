package com.visionml

import kotlin.math.max
import kotlin.math.min

/**
 * Non-Maximum Suppression for filtering overlapping detections
 * Ported from iOS NMS.swift
 */
object NMS {

    data class Detection(
        val box: FloatArray,      // [x1, y1, x2, y2]
        val score: Float,
        val classIndex: Int,
        val className: String
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as Detection
            return box.contentEquals(other.box) &&
                   score == other.score &&
                   classIndex == other.classIndex &&
                   className == other.className
        }

        override fun hashCode(): Int {
            var result = box.contentHashCode()
            result = 31 * result + score.hashCode()
            result = 31 * result + classIndex
            result = 31 * result + className.hashCode()
            return result
        }
    }

    /**
     * Apply NMS to filter overlapping detections
     * @param detections Array of detections to filter
     * @param iouThreshold IoU threshold (default 0.45)
     * @return Filtered list of detections
     */
    fun apply(detections: List<Detection>, iouThreshold: Float = 0.45f): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        // Sort by confidence score (highest first)
        val sorted = detections.sortedByDescending { it.score }
        val keep = mutableListOf<Detection>()
        val suppressed = mutableSetOf<Int>()

        for ((i, detection) in sorted.withIndex()) {
            if (i in suppressed) continue
            keep.add(detection)

            // Suppress overlapping detections
            for ((j, other) in sorted.withIndex()) {
                if (j <= i || j in suppressed) continue

                val iou = calculateIoU(detection.box, other.box)
                if (iou > iouThreshold) {
                    suppressed.add(j)
                }
            }
        }

        return keep
    }

    /**
     * Calculate Intersection over Union between two boxes
     * @param box1 First box [x1, y1, x2, y2]
     * @param box2 Second box [x1, y1, x2, y2]
     * @return IoU value (0.0 to 1.0)
     */
    fun calculateIoU(box1: FloatArray, box2: FloatArray): Float {
        val x1_1 = box1[0]
        val y1_1 = box1[1]
        val x2_1 = box1[2]
        val y2_1 = box1[3]

        val x1_2 = box2[0]
        val y1_2 = box2[1]
        val x2_2 = box2[2]
        val y2_2 = box2[3]

        // Calculate intersection
        val x1_i = max(x1_1, x1_2)
        val y1_i = max(y1_1, y1_2)
        val x2_i = min(x2_1, x2_2)
        val y2_i = min(y2_1, y2_2)

        val intersectionWidth = max(0f, x2_i - x1_i)
        val intersectionHeight = max(0f, y2_i - y1_i)
        val intersection = intersectionWidth * intersectionHeight

        // Calculate union
        val area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        val area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        val union = area1 + area2 - intersection

        return if (union > 0) intersection / union else 0f
    }
}
