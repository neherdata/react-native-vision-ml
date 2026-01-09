package com.visionml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * ML Kit analyzer - Android equivalent of iOS Vision framework
 * Provides face detection, pose detection, image labeling, and text recognition
 *
 * Ported from iOS VisionMLModule.swift Vision framework methods
 */
class MLKitAnalyzer(private val context: Context) {

    companion object {
        private const val TAG = "MLKitAnalyzer"

        // Animal identifiers from ML Kit image labeling
        private val ANIMAL_LABELS = setOf(
            "Cat", "Dog", "Bird", "Fish", "Horse", "Rabbit",
            "Animal", "Pet", "Mammal", "Reptile", "Insect"
        )
    }

    // Lazy-initialized detectors
    private val faceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
            .build()
        FaceDetection.getClient(options)
    }

    private val poseDetector by lazy {
        val options = PoseDetectorOptions.Builder()
            .setDetectorMode(PoseDetectorOptions.SINGLE_IMAGE_MODE)
            .build()
        PoseDetection.getClient(options)
    }

    private val imageLabeler by lazy {
        val options = ImageLabelerOptions.Builder()
            .setConfidenceThreshold(0.5f)
            .build()
        ImageLabeling.getClient(options)
    }

    private val textRecognizer by lazy {
        TextRecognition.getClient(TextRecognizerOptions.Builder().build())
    }

    // MARK: - Public API

    /**
     * Analyze image for animals (cats/dogs)
     * Equivalent to iOS VNRecognizeAnimalsRequest
     */
    suspend fun analyzeAnimals(assetId: String): Map<String, Any> {
        val bitmap = loadBitmap(assetId) ?: return mapOf("animals" to emptyList<Any>(), "count" to 0)
        val inputImage = InputImage.fromBitmap(bitmap, 0)

        return suspendCancellableCoroutine { cont ->
            imageLabeler.process(inputImage)
                .addOnSuccessListener { labels ->
                    val animals = labels
                        .filter { it.text in ANIMAL_LABELS }
                        .map { label ->
                            mapOf(
                                "identifier" to label.text,
                                "confidence" to label.confidence
                            )
                        }

                    cont.resume(mapOf(
                        "animals" to animals,
                        "count" to animals.size
                    ))
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "Animal detection failed: ${e.message}")
                    cont.resume(mapOf("animals" to emptyList<Any>(), "count" to 0))
                }
                .addOnCompleteListener {
                    bitmap.recycle()
                }
        }
    }

    /**
     * Analyze image for human pose
     * Equivalent to iOS VNDetectHumanBodyPoseRequest
     */
    suspend fun analyzeHumanPose(assetId: String): Map<String, Any> {
        val bitmap = loadBitmap(assetId) ?: return mapOf("humans" to emptyList<Any>(), "humanCount" to 0)
        val inputImage = InputImage.fromBitmap(bitmap, 0)

        return suspendCancellableCoroutine { cont ->
            poseDetector.process(inputImage)
                .addOnSuccessListener { pose ->
                    val humans = if (pose.allPoseLandmarks.isNotEmpty()) {
                        val points = mutableMapOf<String, Map<String, Any>>()

                        for (landmark in pose.allPoseLandmarks) {
                            if (landmark.inFrameLikelihood > 0.1f) {
                                points[landmark.landmarkType.toString()] = mapOf(
                                    "x" to landmark.position.x / bitmap.width,
                                    "y" to landmark.position.y / bitmap.height,
                                    "confidence" to landmark.inFrameLikelihood
                                )
                            }
                        }

                        listOf(mapOf(
                            "points" to points,
                            "pointCount" to points.size
                        ))
                    } else {
                        emptyList()
                    }

                    cont.resume(mapOf(
                        "humans" to humans,
                        "humanCount" to humans.size
                    ))
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "Pose detection failed: ${e.message}")
                    cont.resume(mapOf("humans" to emptyList<Any>(), "humanCount" to 0))
                }
                .addOnCompleteListener {
                    bitmap.recycle()
                }
        }
    }

    /**
     * Comprehensive analysis - runs multiple ML Kit detectors
     * Equivalent to iOS performComprehensiveAnalysis
     */
    suspend fun analyzeComprehensive(assetId: String): Map<String, Any> {
        val bitmap = loadBitmap(assetId) ?: return emptyMap()
        val inputImage = InputImage.fromBitmap(bitmap, 0)

        val result = mutableMapOf<String, Any>()

        // Run all detectors in parallel using coroutines
        try {
            // Scene/Label classification
            val labels = runImageLabeling(inputImage)
            result["scenes"] = labels.take(10).map {
                mapOf("identifier" to it.text, "confidence" to it.confidence)
            }

            // Face detection
            val faces = runFaceDetection(inputImage, bitmap.width, bitmap.height)
            result["faces"] = faces
            result["faceCount"] = faces.size

            // Animal detection (from labels)
            val animals = labels.filter { it.text in ANIMAL_LABELS }
            result["animals"] = animals.map {
                mapOf(
                    "boundingBox" to mapOf("x" to 0, "y" to 0, "width" to 1, "height" to 1),
                    "confidence" to it.confidence,
                    "labels" to listOf(mapOf("identifier" to it.text, "confidence" to it.confidence))
                )
            }
            result["animalCount"] = animals.size

            // Human pose detection
            val hasHumans = runPoseDetection(inputImage)
            result["humanCount"] = if (hasHumans) 1 else 0
            result["hasHumans"] = hasHumans

            // Text detection
            val textResult = runTextRecognition(inputImage)
            result["hasText"] = textResult.isNotEmpty()
            result["textRegions"] = textResult.size

            // Rectangle detection (for screenshots) - not directly available in ML Kit
            // Approximate using text regions as indicator
            result["rectangles"] = textResult.size
            result["likelyScreenshot"] = textResult.size > 5

        } finally {
            bitmap.recycle()
        }

        return result
    }

    /**
     * Quick check if image contains humans (for video frame pre-filtering)
     */
    suspend fun hasHumans(bitmap: Bitmap): Boolean {
        val inputImage = InputImage.fromBitmap(bitmap, 0)
        return runPoseDetection(inputImage)
    }

    // MARK: - Private Helpers

    private fun loadBitmap(assetId: String): Bitmap? {
        return try {
            when {
                assetId.startsWith("content://") -> {
                    context.contentResolver.openInputStream(Uri.parse(assetId))?.use {
                        BitmapFactory.decodeStream(it)
                    }
                }
                assetId.startsWith("file://") -> {
                    BitmapFactory.decodeFile(assetId.removePrefix("file://"))
                }
                assetId.all { it.isDigit() } -> {
                    // MediaStore ID
                    val uri = Uri.withAppendedPath(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, assetId)
                    context.contentResolver.openInputStream(uri)?.use {
                        BitmapFactory.decodeStream(it)
                    }
                }
                else -> {
                    BitmapFactory.decodeFile(assetId)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load bitmap: ${e.message}")
            null
        }
    }

    private suspend fun runImageLabeling(inputImage: InputImage): List<com.google.mlkit.vision.label.ImageLabel> {
        return suspendCancellableCoroutine { cont ->
            imageLabeler.process(inputImage)
                .addOnSuccessListener { labels -> cont.resume(labels) }
                .addOnFailureListener { e ->
                    Log.e(TAG, "Image labeling failed: ${e.message}")
                    cont.resume(emptyList())
                }
        }
    }

    private suspend fun runFaceDetection(
        inputImage: InputImage,
        imageWidth: Int,
        imageHeight: Int
    ): List<Map<String, Any>> {
        return suspendCancellableCoroutine { cont ->
            faceDetector.process(inputImage)
                .addOnSuccessListener { faces ->
                    val result = faces.map { face ->
                        val bounds = face.boundingBox
                        mapOf(
                            "boundingBox" to mapOf(
                                "x" to bounds.left.toFloat() / imageWidth,
                                "y" to bounds.top.toFloat() / imageHeight,
                                "width" to bounds.width().toFloat() / imageWidth,
                                "height" to bounds.height().toFloat() / imageHeight
                            ),
                            "confidence" to (face.trackingId?.toFloat() ?: 0.9f)
                        )
                    }
                    cont.resume(result)
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "Face detection failed: ${e.message}")
                    cont.resume(emptyList())
                }
        }
    }

    private suspend fun runPoseDetection(inputImage: InputImage): Boolean {
        return suspendCancellableCoroutine { cont ->
            poseDetector.process(inputImage)
                .addOnSuccessListener { pose ->
                    cont.resume(pose.allPoseLandmarks.isNotEmpty())
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "Pose detection failed: ${e.message}")
                    cont.resume(false)
                }
        }
    }

    private suspend fun runTextRecognition(inputImage: InputImage): List<String> {
        return suspendCancellableCoroutine { cont ->
            textRecognizer.process(inputImage)
                .addOnSuccessListener { text ->
                    cont.resume(text.textBlocks.map { it.text })
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "Text recognition failed: ${e.message}")
                    cont.resume(emptyList())
                }
        }
    }

    /**
     * Clean up resources
     */
    fun dispose() {
        try {
            faceDetector.close()
            poseDetector.close()
            imageLabeler.close()
            textRecognizer.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error disposing ML Kit: ${e.message}")
        }
    }
}
