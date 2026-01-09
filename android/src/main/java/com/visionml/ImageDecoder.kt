package com.visionml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import kotlin.math.max

/**
 * Image decoder with letterbox resize for YOLO preprocessing
 * Ported from iOS ImageDecoder.swift
 */
class ImageDecoder(private val context: Context) {

    companion object {
        private const val TAG = "ImageDecoder"
        private const val MAX_ALLOWED_DIMENSION = 8192
    }

    sealed class DecodeError : Exception() {
        object InvalidUri : DecodeError()
        object FailedToLoadData : DecodeError()
        object FailedToCreateBitmap : DecodeError()
        object FailedToResize : DecodeError()
        object ImageTooLarge : DecodeError()
    }

    data class DecodedImage(
        val width: Int,           // Resized width (or original if no resize)
        val height: Int,          // Resized height (or original if no resize)
        val originalWidth: Int,   // Original image width before resize
        val originalHeight: Int,  // Original image height before resize
        val data: FloatArray      // Normalized RGB pixel data (0.0-1.0), HWC format
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as DecodedImage
            return width == other.width && height == other.height &&
                   originalWidth == other.originalWidth && originalHeight == other.originalHeight &&
                   data.contentEquals(other.data)
        }

        override fun hashCode(): Int {
            var result = width
            result = 31 * result + height
            result = 31 * result + originalWidth
            result = 31 * result + originalHeight
            result = 31 * result + data.contentHashCode()
            return result
        }
    }

    /**
     * Decode and optionally resize image to target size
     * @param imageUri file:// URI or content:// URI to image
     * @param targetSize Target size for square resize (0 = no resize)
     * @return Decoded image with normalized pixel data
     */
    fun decode(imageUri: String, targetSize: Int = 0): DecodedImage {
        Log.d(TAG, "START: decode called with URI: $imageUri targetSize: $targetSize")

        val bitmap = loadBitmap(imageUri)
            ?: throw DecodeError.FailedToLoadData

        val originalWidth = bitmap.width
        val originalHeight = bitmap.height
        Log.d(TAG, "Original dimensions: ${originalWidth}x$originalHeight")

        // Safety check for extremely large images
        if (originalWidth > MAX_ALLOWED_DIMENSION || originalHeight > MAX_ALLOWED_DIMENSION) {
            Log.e(TAG, "Image too large: ${originalWidth}x$originalHeight (max $MAX_ALLOWED_DIMENSION)")
            bitmap.recycle()
            throw DecodeError.ImageTooLarge
        }

        // Letterbox resize if target size specified
        val processedBitmap: Bitmap
        val finalWidth: Int
        val finalHeight: Int

        if (targetSize > 0) {
            Log.d(TAG, "Letterbox resizing to ${targetSize}x$targetSize...")
            processedBitmap = letterboxResize(bitmap, targetSize)
                ?: run {
                    Log.w(TAG, "Letterbox failed, trying simple resize...")
                    simpleResize(bitmap, targetSize) ?: run {
                        Log.e(TAG, "All resize methods failed")
                        bitmap.recycle()
                        throw DecodeError.FailedToResize
                    }
                }
            finalWidth = targetSize
            finalHeight = targetSize

            // Recycle original if we created a new bitmap
            if (processedBitmap !== bitmap) {
                bitmap.recycle()
            }
        } else {
            processedBitmap = bitmap
            finalWidth = originalWidth
            finalHeight = originalHeight
        }

        Log.d(TAG, "Final dimensions: ${finalWidth}x$finalHeight")

        // Extract normalized pixel data
        val normalizedData = extractNormalizedPixels(processedBitmap)

        // Recycle bitmap if we created it during resize
        if (targetSize > 0) {
            processedBitmap.recycle()
        }

        Log.d(TAG, "Decode complete: ${normalizedData.size} floats")

        return DecodedImage(
            width = finalWidth,
            height = finalHeight,
            originalWidth = originalWidth,
            originalHeight = originalHeight,
            data = normalizedData
        )
    }

    /**
     * Decode directly to Bitmap (for video frames that are already Bitmap)
     */
    fun decodeBitmap(bitmap: Bitmap, targetSize: Int = 0): DecodedImage {
        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        val processedBitmap: Bitmap
        val finalWidth: Int
        val finalHeight: Int

        if (targetSize > 0) {
            processedBitmap = letterboxResize(bitmap, targetSize)
                ?: simpleResize(bitmap, targetSize)
                ?: throw DecodeError.FailedToResize
            finalWidth = targetSize
            finalHeight = targetSize
        } else {
            processedBitmap = bitmap
            finalWidth = originalWidth
            finalHeight = originalHeight
        }

        val normalizedData = extractNormalizedPixels(processedBitmap)

        if (processedBitmap !== bitmap && targetSize > 0) {
            processedBitmap.recycle()
        }

        return DecodedImage(
            width = finalWidth,
            height = finalHeight,
            originalWidth = originalWidth,
            originalHeight = originalHeight,
            data = normalizedData
        )
    }

    private fun loadBitmap(imageUri: String): Bitmap? {
        return try {
            val inputStream: InputStream? = when {
                imageUri.startsWith("file://") -> {
                    val path = imageUri.removePrefix("file://")
                    FileInputStream(File(path))
                }
                imageUri.startsWith("content://") -> {
                    context.contentResolver.openInputStream(Uri.parse(imageUri))
                }
                imageUri.startsWith("/") -> {
                    FileInputStream(File(imageUri))
                }
                else -> {
                    Log.e(TAG, "Unsupported URI scheme: $imageUri")
                    null
                }
            }

            inputStream?.use { stream ->
                // First decode to get dimensions without loading full bitmap
                val options = BitmapFactory.Options().apply {
                    inJustDecodeBounds = true
                }
                BitmapFactory.decodeStream(stream, null, options)

                // Reset stream and decode actual bitmap
                inputStream.close()
                val newStream = when {
                    imageUri.startsWith("file://") -> FileInputStream(File(imageUri.removePrefix("file://")))
                    imageUri.startsWith("content://") -> context.contentResolver.openInputStream(Uri.parse(imageUri))
                    imageUri.startsWith("/") -> FileInputStream(File(imageUri))
                    else -> null
                }

                newStream?.use { s ->
                    val bitmap = BitmapFactory.decodeStream(s)

                    // Handle EXIF rotation
                    if (bitmap != null) {
                        val rotated = applyExifRotation(imageUri, bitmap)
                        if (rotated !== bitmap) {
                            bitmap.recycle()
                        }
                        rotated
                    } else {
                        null
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load bitmap: ${e.message}")
            null
        }
    }

    private fun applyExifRotation(imageUri: String, bitmap: Bitmap): Bitmap {
        try {
            val path = when {
                imageUri.startsWith("file://") -> imageUri.removePrefix("file://")
                imageUri.startsWith("/") -> imageUri
                else -> return bitmap  // Can't read EXIF from content:// easily
            }

            val exif = ExifInterface(path)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )

            val matrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
                ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
                ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1f, -1f)
                else -> return bitmap  // No rotation needed
            }

            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to apply EXIF rotation: ${e.message}")
            return bitmap
        }
    }

    /**
     * Letterbox resize: pad image to square, then resize to target size
     * NudeNet pads on RIGHT and BOTTOM (image at top-left corner)
     */
    private fun letterboxResize(image: Bitmap, targetSize: Int): Bitmap? {
        val originalWidth = image.width
        val originalHeight = image.height

        // Find the max dimension for square padding
        val maxDim = max(originalWidth, originalHeight)

        // NudeNet pads on right and bottom
        val xPad = maxDim - originalWidth
        val yPad = maxDim - originalHeight

        Log.d(TAG, "Letterbox: original ${originalWidth}x$originalHeight, maxDim $maxDim, pad right=$xPad bottom=$yPad")

        return try {
            // Step 1: Create padded square image (black background)
            val paddedBitmap = Bitmap.createBitmap(maxDim, maxDim, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(paddedBitmap)
            canvas.drawColor(Color.BLACK)

            // Draw original image at TOP-LEFT (NudeNet style)
            canvas.drawBitmap(image, 0f, 0f, null)

            // Step 2: Resize padded square to target size
            val resized = Bitmap.createScaledBitmap(paddedBitmap, targetSize, targetSize, true)

            // Recycle intermediate bitmap
            if (resized !== paddedBitmap) {
                paddedBitmap.recycle()
            }

            resized
        } catch (e: Exception) {
            Log.e(TAG, "Letterbox resize failed: ${e.message}")
            null
        }
    }

    /**
     * Simple resize without letterboxing - fallback
     */
    private fun simpleResize(image: Bitmap, targetSize: Int): Bitmap? {
        return try {
            Bitmap.createScaledBitmap(image, targetSize, targetSize, true)
        } catch (e: Exception) {
            Log.e(TAG, "Simple resize failed: ${e.message}")
            null
        }
    }

    /**
     * Extract normalized RGB pixel data from bitmap
     * Returns HWC format (height, width, channels) with values 0.0-1.0
     */
    private fun extractNormalizedPixels(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Convert to normalized float RGB (HWC format)
        val normalized = FloatArray(width * height * 3)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            normalized[i * 3] = r / 255f
            normalized[i * 3 + 1] = g / 255f
            normalized[i * 3 + 2] = b / 255f
        }

        // Debug: Log sample pixel values
        if (width > 0 && height > 0) {
            val centerIdx = (height / 2) * width + (width / 2)
            Log.d(TAG, "DEBUG center pixel RGB: ${normalized[centerIdx * 3]}, ${normalized[centerIdx * 3 + 1]}, ${normalized[centerIdx * 3 + 2]}")
        }

        return normalized
    }

    /**
     * Convert HWC (Height, Width, Channels) to NCHW (Batch, Channels, Height, Width)
     * Required for ONNX input tensor
     */
    fun convertToNCHW(hwcData: FloatArray, width: Int, height: Int): FloatArray {
        val nchwData = FloatArray(3 * height * width)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val hwcIndex = (y * width + x) * 3
                val baseIndex = y * width + x

                nchwData[baseIndex] = hwcData[hwcIndex]                          // R channel
                nchwData[height * width + baseIndex] = hwcData[hwcIndex + 1]     // G channel
                nchwData[2 * height * width + baseIndex] = hwcData[hwcIndex + 2] // B channel
            }
        }

        return nchwData
    }
}
