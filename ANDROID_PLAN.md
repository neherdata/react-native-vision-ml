# Android Implementation for react-native-vision-ml

## Status: ✅ Core Implementation Complete

All major components have been implemented. Ready for integration testing.

## Implementation Summary

### ✅ Completed Components

| Component | File | Status |
|-----------|------|--------|
| NMS (Non-Maximum Suppression) | `NMS.kt` | ✅ Complete |
| YOLO Parser | `YOLOParser.kt` | ✅ Complete |
| Image Decoder | `ImageDecoder.kt` | ✅ Complete |
| ONNX Inference | `ONNXInference.kt` | ✅ Complete |
| Video Analyzer | `VideoAnalyzer.kt` | ✅ Complete |
| ML Kit Analyzer | `MLKitAnalyzer.kt` | ✅ Complete |
| RN Bridge Module | `VisionMLModule.kt` | ✅ Complete |
| Package Registration | `VisionMLPackage.kt` | ✅ Complete |
| Build Configuration | `build.gradle` | ✅ Complete |
| Manifest | `AndroidManifest.xml` | ✅ Complete |

### Feature Mapping: iOS → Android

| iOS Feature | Android Implementation | Status |
|-------------|----------------------|--------|
| ONNX Runtime (Swift/ObjC) | ONNX Runtime Android (Kotlin) | ✅ |
| Vision Framework | Google ML Kit | ✅ |
| AVFoundation (video) | MediaMetadataRetriever | ✅ |
| PHAsset | MediaStore/ContentResolver | ✅ |
| Live Activity | Foreground Service (stub) | 🔶 Stub |
| CoreML acceleration | NNAPI delegate | ✅ |
| SensitiveContentAnalysis | N/A (iOS 17+ only) | ❌ iOS-only |

## Architecture

```
VisionMLModule.kt (React Native Bridge)
    ├── ONNXInference.kt
    │   ├── ImageDecoder.kt (letterbox resize, EXIF handling)
    │   ├── YOLOParser.kt (output parsing)
    │   └── NMS.kt (non-maximum suppression)
    │
    ├── VideoAnalyzer.kt
    │   ├── ScanMode: quick_check, sampled, thorough, binary_search, full_short_circuit
    │   └── MediaMetadataRetriever for frame extraction
    │
    └── MLKitAnalyzer.kt
        ├── Face Detection
        ├── Pose Detection
        ├── Image Labeling (animal detection)
        └── Text Recognition
```

## API Parity with iOS

All iOS methods are implemented:

```kotlin
// Detector Management
createDetector(modelPath, classLabels, inputSize) → detectorId
detect(detectorId, imageUri, confThreshold, iouThreshold) → detections
disposeDetector(detectorId)
disposeAllDetectors()

// Video Analysis
analyzeVideo(detectorId, assetId, mode, sampleInterval, confThreshold)
quickCheckVideo(detectorId, assetId, confThreshold)

// ML Kit (Vision Framework equivalent)
analyzeAnimals(assetId)
analyzeHumanPose(assetId)
analyzeComprehensive(assetId)

// Progress Notification (Live Activity equivalent)
isLiveActivityAvailable() → true (always available on Android)
startVideoScanActivity(videoName, duration, mode)
updateVideoScanActivity(progress, phase, nsfwCount, framesAnalyzed)
endVideoScanActivity(nsfwCount, framesAnalyzed, isNSFW)
```

## Dependencies

```gradle
// ONNX Runtime
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'

// ML Kit
implementation 'com.google.mlkit:pose-detection:18.0.0-beta4'
implementation 'com.google.mlkit:face-detection:16.1.6'
implementation 'com.google.mlkit:image-labeling:17.0.8'
implementation 'com.google.mlkit:text-recognition:16.0.0'

// Kotlin Coroutines
implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3'
implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
```

## Key Differences from iOS

1. **Y-axis orientation**: Android Bitmap has origin at top-left (same as standard image coordinates), so no Y-flip needed in YOLOParser (unlike iOS CGContext which has origin at bottom-left)

2. **EXIF handling**: Android requires explicit ExifInterface for rotation, handled in ImageDecoder

3. **Video frame extraction**: Uses `MediaMetadataRetriever.getFrameAtTime()` instead of `AVAssetImageGenerator`

4. **ML Kit vs Vision**: Similar APIs but different class names and result formats

5. **Live Activity**: iOS-specific feature - Android uses foreground service with notification (stubbed)

## Testing

```bash
# Build the Android library
cd android
./gradlew assembleDebug

# In the consuming app
npx react-native run-android
```

## Remaining Work

1. **Foreground Service**: Implement actual progress notification for video scanning
2. **Integration Testing**: Test with actual ONNX models on Android device
3. **Performance Tuning**: Profile NNAPI acceleration on various devices
4. **Error Handling**: Add more detailed error messages for debugging

## Notes

- ONNX models are cross-platform - same .onnx file works on both iOS and Android
- NNAPI acceleration is attempted automatically, falls back to CPU if unavailable
- ML Kit models are downloaded on first use (requires network)
- Video frame extraction is simpler on Android than iOS
