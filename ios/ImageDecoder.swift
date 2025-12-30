import Foundation
import UIKit

/// Image decoder with hardware-accelerated resize
class ImageDecoder {

  enum DecodeError: Error {
    case invalidURI
    case unsupportedScheme
    case failedToLoadData
    case failedToCreateImage
    case failedToGetCGImage
    case failedToResize
    case failedToCreateColorSpace
    case failedToCreateContext
  }

  struct DecodedImage {
    let width: Int           // Resized width (or original if no resize)
    let height: Int          // Resized height (or original if no resize)
    let originalWidth: Int   // Original image width before any resize
    let originalHeight: Int  // Original image height before any resize
    let data: [Float]        // Normalized RGB pixel data (0.0-1.0)
  }

  /// Decode and optionally resize image to target size
  /// - Parameters:
  ///   - imageUri: file:// URI to local image
  ///   - targetSize: Target size for square resize (0 = no resize)
  /// - Returns: Decoded image with normalized pixel data
  /// - Throws: DecodeError if any step fails
  static func decode(imageUri: String, targetSize: Int = 0) throws -> DecodedImage {
    NSLog("[ImageDecoder] START: decode called with URI: %@ targetSize: %d", imageUri, targetSize)

    guard let url = URL(string: imageUri) else {
      NSLog("[ImageDecoder] ERROR: Invalid URI")
      throw DecodeError.invalidURI
    }

    // Handle file:// URIs
    let fileURL: URL
    if url.scheme == "file" {
      fileURL = url
      NSLog("[ImageDecoder] Using file:// URL: %@", fileURL.path)
    } else if url.scheme == nil {
      fileURL = URL(fileURLWithPath: imageUri)
      NSLog("[ImageDecoder] Using local path: %@", fileURL.path)
    } else {
      NSLog("[ImageDecoder] ERROR: Unsupported scheme: %@", url.scheme ?? "nil")
      throw DecodeError.unsupportedScheme
    }

    NSLog("[ImageDecoder] Loading image data from: %@", fileURL.path)
    guard let imageData = try? Data(contentsOf: fileURL) else {
      NSLog("[ImageDecoder] ERROR: Failed to load image data")
      throw DecodeError.failedToLoadData
    }

    NSLog("[ImageDecoder] Image data loaded, size: %lu bytes", imageData.count)
    guard let image = UIImage(data: imageData) else {
      NSLog("[ImageDecoder] ERROR: Failed to create UIImage from data")
      throw DecodeError.failedToCreateImage
    }

    NSLog("[ImageDecoder] UIImage created successfully")
    guard let cgImage = image.cgImage else {
      NSLog("[ImageDecoder] ERROR: Failed to get CGImage from UIImage")
      throw DecodeError.failedToGetCGImage
    }

    let originalWidth = cgImage.width
    let originalHeight = cgImage.height
    NSLog("[ImageDecoder] Original image dimensions: %d x %d", originalWidth, originalHeight)

    // Letterbox resize (pad to square, then resize) - matches NudeNet preprocessing
    let resizedImage: CGImage
    let width: Int
    let height: Int

    if targetSize > 0 {
      NSLog("[ImageDecoder] Letterbox resizing to %d x %d...", targetSize, targetSize)
      guard let letterboxed = letterboxResize(image: cgImage, targetSize: targetSize) else {
        NSLog("[ImageDecoder] ERROR: Failed to letterbox resize image")
        throw DecodeError.failedToResize
      }
      resizedImage = letterboxed
      width = targetSize
      height = targetSize
      NSLog("[ImageDecoder] ✓ Letterbox resize complete: %d x %d", width, height)
    } else {
      resizedImage = cgImage
      width = originalWidth
      height = originalHeight
      NSLog("[ImageDecoder] Using original dimensions: %d x %d", width, height)
    }

    // Extract pixel data
    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel
    let pixelCount = width * height * bytesPerPixel
    NSLog("[ImageDecoder] Allocating pixel buffer: %d bytes", pixelCount)
    var pixelData = [UInt8](repeating: 0, count: pixelCount)

    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
      NSLog("[ImageDecoder] ERROR: Failed to create sRGB color space")
      throw DecodeError.failedToCreateColorSpace
    }

    let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

    guard let context = CGContext(
      data: &pixelData,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      NSLog("[ImageDecoder] ERROR: Failed to create graphics context")
      throw DecodeError.failedToCreateContext
    }

    NSLog("[ImageDecoder] Drawing image into context...")
    context.draw(resizedImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    NSLog("[ImageDecoder] Image drawn into context")

    // Convert UInt8 RGBA to Float32 RGB and normalize
    let normalizedCount = width * height * 3
    NSLog("[ImageDecoder] Converting to normalized Float32 array (%d floats)...", normalizedCount)
    var normalizedPixels = [Float](repeating: 0, count: normalizedCount)

    for i in 0..<(width * height) {
      let offset = i * bytesPerPixel
      normalizedPixels[i * 3] = Float(pixelData[offset]) / 255.0      // R
      normalizedPixels[i * 3 + 1] = Float(pixelData[offset + 1]) / 255.0  // G
      normalizedPixels[i * 3 + 2] = Float(pixelData[offset + 2]) / 255.0  // B
    }

    NSLog("[ImageDecoder] ✓ Decode complete")
    return DecodedImage(
      width: width,
      height: height,
      originalWidth: originalWidth,
      originalHeight: originalHeight,
      data: normalizedPixels
    )
  }

  /// Letterbox resize: pad image to square, then resize to target size
  /// This matches NudeNet's preprocessing which preserves aspect ratio
  private static func letterboxResize(image: CGImage, targetSize: Int) -> CGImage? {
    let originalWidth = image.width
    let originalHeight = image.height

    // Find the max dimension for square padding
    let maxDim = max(originalWidth, originalHeight)

    // Calculate padding (NudeNet pads on right and bottom)
    let xPad = maxDim - originalWidth
    let yPad = maxDim - originalHeight

    NSLog("[ImageDecoder] Letterbox: original %dx%d, maxDim %d, pad x=%d y=%d",
          originalWidth, originalHeight, maxDim, xPad, yPad)

    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
      NSLog("[ImageDecoder] ERROR: Failed to create sRGB colorspace")
      return nil
    }

    let bytesPerPixel = 4
    let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue

    // Step 1: Create padded square image (black background)
    guard let paddedContext = CGContext(
      data: nil,
      width: maxDim,
      height: maxDim,
      bitsPerComponent: 8,
      bytesPerRow: maxDim * bytesPerPixel,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      NSLog("[ImageDecoder] ERROR: Failed to create padded context")
      return nil
    }

    // Fill with black (padding color)
    paddedContext.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    paddedContext.fill(CGRect(x: 0, y: 0, width: maxDim, height: maxDim))

    // Draw original image at top-left (NudeNet style: padding on right and bottom)
    // Note: CGContext has origin at bottom-left, so we draw at (0, yPad) to pad bottom
    paddedContext.draw(image, in: CGRect(x: 0, y: yPad, width: originalWidth, height: originalHeight))

    guard let paddedImage = paddedContext.makeImage() else {
      NSLog("[ImageDecoder] ERROR: Failed to create padded image")
      return nil
    }

    // Step 2: Resize padded square to target size
    guard let resizeContext = CGContext(
      data: nil,
      width: targetSize,
      height: targetSize,
      bitsPerComponent: 8,
      bytesPerRow: targetSize * bytesPerPixel,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      NSLog("[ImageDecoder] ERROR: Failed to create resize context")
      return nil
    }

    resizeContext.interpolationQuality = .high
    resizeContext.draw(paddedImage, in: CGRect(x: 0, y: 0, width: targetSize, height: targetSize))

    return resizeContext.makeImage()
  }
}
