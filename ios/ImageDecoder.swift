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

    // IMPORTANT: Use UIImage.size for dimensions, NOT cgImage.width/height!
    // cgImage gives raw pixel dimensions ignoring EXIF orientation
    // UIImage.size gives display dimensions with orientation applied
    // MediaLibrary also returns orientation-aware dimensions, so we must match
    let originalWidth = Int(image.size.width)
    let originalHeight = Int(image.size.height)
    let rawWidth = cgImage.width
    let rawHeight = cgImage.height
    NSLog("[ImageDecoder] Original dimensions (display): %d x %d, raw CGImage: %d x %d",
          originalWidth, originalHeight, rawWidth, rawHeight)

    // Check if orientation differs from raw
    if originalWidth != rawWidth || originalHeight != rawHeight {
      NSLog("[ImageDecoder] ⚠️ Image has EXIF rotation - using display dimensions for coordinates")
    }

    // First, render the UIImage to a new CGImage with orientation applied
    // This ensures the pixel data matches the display orientation
    let orientedImage: CGImage
    if originalWidth != rawWidth || originalHeight != rawHeight {
      NSLog("[ImageDecoder] Rendering with orientation correction...")
      guard let oriented = renderWithOrientation(image: image) else {
        NSLog("[ImageDecoder] ERROR: Failed to render with orientation")
        throw DecodeError.failedToResize
      }
      orientedImage = oriented
      NSLog("[ImageDecoder] ✓ Orientation corrected: %d x %d", orientedImage.width, orientedImage.height)
    } else {
      orientedImage = cgImage
    }

    // Letterbox resize (pad to square, then resize) - matches NudeNet preprocessing
    let resizedImage: CGImage
    let width: Int
    let height: Int

    if targetSize > 0 {
      NSLog("[ImageDecoder] Letterbox resizing to %d x %d...", targetSize, targetSize)
      guard let letterboxed = letterboxResize(image: orientedImage, targetSize: targetSize) else {
        NSLog("[ImageDecoder] ERROR: Failed to letterbox resize image")
        throw DecodeError.failedToResize
      }
      resizedImage = letterboxed
      width = targetSize
      height = targetSize
      NSLog("[ImageDecoder] ✓ Letterbox resize complete: %d x %d", width, height)
    } else {
      resizedImage = orientedImage
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

    // DEBUG: Log sample pixel values to verify preprocessing
    // Log center pixel
    let centerIdx = (height / 2) * width + (width / 2)
    NSLog("[ImageDecoder] DEBUG center pixel RGB: %.3f, %.3f, %.3f",
          normalizedPixels[centerIdx * 3], normalizedPixels[centerIdx * 3 + 1], normalizedPixels[centerIdx * 3 + 2])
    // Log corner pixel (should be black padding area for non-square images)
    let bottomRightIdx = (height - 1) * width + (width - 1)
    NSLog("[ImageDecoder] DEBUG bottom-right pixel RGB: %.3f, %.3f, %.3f",
          normalizedPixels[bottomRightIdx * 3], normalizedPixels[bottomRightIdx * 3 + 1], normalizedPixels[bottomRightIdx * 3 + 2])
    // Log top-left pixel
    NSLog("[ImageDecoder] DEBUG top-left pixel RGB: %.3f, %.3f, %.3f",
          normalizedPixels[0], normalizedPixels[1], normalizedPixels[2])

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
  /// NudeNet pads on RIGHT and BOTTOM (image at top-left corner)
  private static func letterboxResize(image: CGImage, targetSize: Int) -> CGImage? {
    let originalWidth = image.width
    let originalHeight = image.height

    // Safety check for extremely large images that might cause memory issues
    let maxAllowedDimension = 8192
    guard originalWidth > 0 && originalHeight > 0 &&
          originalWidth <= maxAllowedDimension && originalHeight <= maxAllowedDimension else {
      NSLog("[ImageDecoder] ERROR: Image dimensions invalid or too large: %dx%d (max %d)",
            originalWidth, originalHeight, maxAllowedDimension)
      return nil
    }

    // Find the max dimension for square padding
    let maxDim = max(originalWidth, originalHeight)

    // NudeNet pads on right and bottom: cv2.copyMakeBorder(img, 0, y_pad, 0, x_pad, ...)
    // This means image is at TOP-LEFT, padding on RIGHT and BOTTOM
    let xPad = maxDim - originalWidth  // Padding on right
    let yPad = maxDim - originalHeight  // Padding on bottom

    NSLog("[ImageDecoder] Letterbox (right/bottom pad): original %dx%d, maxDim %d, pad right=%d bottom=%d",
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

    // Draw original image at TOP-LEFT (NudeNet style: padding on right and bottom)
    // CGContext has origin at BOTTOM-left, so to put image at TOP, draw at y=yPad
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

  /// Render UIImage to CGImage with orientation applied
  /// This ensures the CGImage pixel data matches the display orientation
  private static func renderWithOrientation(image: UIImage) -> CGImage? {
    let width = Int(image.size.width)
    let height = Int(image.size.height)

    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
      NSLog("[ImageDecoder] ERROR: Failed to create sRGB colorspace for orientation")
      return nil
    }

    let bytesPerPixel = 4
    let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue

    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * bytesPerPixel,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      NSLog("[ImageDecoder] ERROR: Failed to create context for orientation")
      return nil
    }

    // Draw UIImage - this automatically applies the orientation transform
    UIGraphicsPushContext(context)
    image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
    UIGraphicsPopContext()

    return context.makeImage()
  }
}
