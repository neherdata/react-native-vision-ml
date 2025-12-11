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

    // Resize if needed
    let resizedImage: CGImage
    let width: Int
    let height: Int

    if targetSize > 0 && (originalWidth != targetSize || originalHeight != targetSize) {
      NSLog("[ImageDecoder] Resizing to %d x %d using hardware-accelerated CGContext...", targetSize, targetSize)
      guard let resized = resize(image: cgImage, targetWidth: targetSize, targetHeight: targetSize) else {
        NSLog("[ImageDecoder] ERROR: Failed to resize image")
        throw DecodeError.failedToResize
      }
      resizedImage = resized
      width = targetSize
      height = targetSize
      NSLog("[ImageDecoder] ✓ Resize complete: %d x %d", width, height)
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

  /// Hardware-accelerated image resize using CGContext
  private static func resize(image: CGImage, targetWidth: Int, targetHeight: Int) -> CGImage? {
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
      NSLog("[ImageDecoder] ERROR: Failed to create sRGB colorspace for resize")
      return nil
    }

    let bytesPerPixel = 4
    let bytesPerRow = targetWidth * bytesPerPixel
    let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue

    guard let context = CGContext(
      data: nil,
      width: targetWidth,
      height: targetHeight,
      bitsPerComponent: 8,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      NSLog("[ImageDecoder] ERROR: Failed to create resize context")
      return nil
    }

    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

    return context.makeImage()
  }
}
