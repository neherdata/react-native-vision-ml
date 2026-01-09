# Android Implementation Plan for react-native-vision-ml

## Overview

This document outlines the plan to add Android support to the react-native-vision-ml native module. The iOS implementation provides on-device ONNX inference with Vision framework integration for human/pose/face detection and video analysis.

## Feature Mapping: iOS → Android

| iOS Feature | Android Equivalent | Status |
|-------------|-------------------|--------|
| ONNX Runtime (Swift/ObjC) | ONNX Runtime Android (Kotlin) | 📋 Planned |
| Vision Framework | Google ML Kit | 📋 Planned |
| AVFoundation (video) | MediaExtractor + ImageReader | 📋 Planned |
| PHAsset | MediaStore/ContentResolver | 📋 Planned |
| Live Activity | Foreground Service + Notification | 📋 Planned |
| CoreML acceleration | NNAPI delegate | 📋 Planned |

## Implementation Phases

### Phase 1: Core ONNX Inference ⏱️ ~2-3 days

**Files to create:**
- `android/src/main/java/com/visionml/ONNXInference.kt`
- `android/src/main/java/com/visionml/ImageDecoder.kt`
- `android/src/main/java/com/visionml/YOLOParser.kt`
- `android/src/main/java/com/visionml/NMS.kt`

**Dependencies:**
```gradle
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'
```

**Key implementation notes:**
1. ONNX Runtime Android has nearly identical API to iOS:
   - `OrtEnvironment` → iOS `ONNXWrapper`
   - `OrtSession` → Same concept
   - Input/output tensors work the same way

2. Image preprocessing (letterbox resize, NCHW conversion) is identical logic

3. YOLO output parsing and NMS are pure math - can port directly

### Phase 2: ML Kit Integration ⏱️ ~2 days

**Files to create:**
- `android/src/main/java/com/visionml/VisionAnalyzer.kt`

**Dependencies:**
```gradle
implementation 'com.google.mlkit:pose-detection:18.0.0-beta4'
implementation 'com.google.mlkit:face-detection:16.1.6'
implementation 'com.google.mlkit:image-labeling:17.0.8'
```

**ML Kit equivalents:**
| iOS Vision Request | ML Kit API |
|-------------------|------------|
| `VNDetectHumanBodyPoseRequest` | `PoseDetector` |
| `VNDetectFaceRectanglesRequest` | `FaceDetector` |
| `VNRecognizeAnimalsRequest` | `ImageLabeler` (animals subset) |
| `VNClassifyImageRequest` | `ImageLabeler` |
| `VNDetectHumanRectanglesRequest` | `PoseDetector` (check if any poses) |
| `VNRecognizeTextRequest` | `TextRecognizer` |

### Phase 3: Video Analysis ⏱️ ~3-4 days

**Files to create:**
- `android/src/main/java/com/visionml/VideoAnalyzer.kt`
- `android/src/main/java/com/visionml/VideoFrameExtractor.kt`

**Implementation approach:**

```kotlin
class VideoFrameExtractor(private val context: Context) {

    fun extractFrameAtTime(uri: Uri, timestampMs: Long): Bitmap? {
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, uri)
        return retriever.getFrameAtTime(timestampMs * 1000) // microseconds
    }

    fun getVideoDuration(uri: Uri): Long {
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, uri)
        return retriever.extractMetadata(
            MediaMetadataRetriever.METADATA_KEY_DURATION
        )?.toLong() ?: 0
    }
}
```

**Scan modes - all work the same way:**
- `quick_check` - Extract 3 frames (start, middle, end)
- `sampled` - Extract frames at regular intervals
- `thorough` - ML Kit pose detection → ONNX on human frames
- `binary_search` - Same algorithm, different frame extraction API
- `full_short_circuit` - Same as sampled but stops on first detection

### Phase 4: Progress Notification (Live Activity equivalent) ⏱️ ~1 day

**Files to create:**
- `android/src/main/java/com/visionml/ScanProgressService.kt`

**Implementation:**

iOS Live Activity shows progress on Dynamic Island. Android equivalent is a Foreground Service with a progress notification:

```kotlin
class ScanProgressService : Service() {
    private val NOTIFICATION_ID = 1
    private val CHANNEL_ID = "video_scan_progress"

    fun updateProgress(progress: Float, phase: String, nsfwCount: Int) {
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Scanning Video")
            .setContentText(phase)
            .setProgress(100, (progress * 100).toInt(), false)
            .setSmallIcon(R.drawable.ic_scan)
            .setOngoing(true)
            .build()

        startForeground(NOTIFICATION_ID, notification)
    }
}
```

### Phase 5: React Native Bridge ⏱️ ~1 day

**Files to create:**
- `android/src/main/java/com/visionml/VisionMLModule.kt`
- `android/src/main/java/com/visionml/VisionMLPackage.kt`

**Bridge methods to implement:**
```kotlin
@ReactMethod
fun createDetector(modelPath: String, classLabels: ReadableArray, inputSize: Int, promise: Promise)

@ReactMethod
fun detect(detectorId: String, imageUri: String, confThreshold: Double, iouThreshold: Double, promise: Promise)

@ReactMethod
fun disposeDetector(detectorId: String, promise: Promise)

@ReactMethod
fun analyzeVideo(detectorId: String, assetId: String, mode: String, sampleInterval: Double, confThreshold: Double, promise: Promise)

@ReactMethod
fun analyzeAnimals(assetId: String, promise: Promise)

@ReactMethod
fun analyzeHumanPose(assetId: String, promise: Promise)

@ReactMethod
fun analyzeComprehensive(assetId: String, promise: Promise)

// Progress notification methods (instead of Live Activity)
@ReactMethod
fun startScanProgress(videoName: String, duration: Double, mode: String, promise: Promise)

@ReactMethod
fun updateScanProgress(progress: Double, phase: String, nsfwCount: Int, framesAnalyzed: Int, promise: Promise)

@ReactMethod
fun endScanProgress(nsfwCount: Int, framesAnalyzed: Int, isNSFW: Boolean, promise: Promise)
```

## TypeScript Changes

Update `src/index.ts` to handle platform differences:

```typescript
import { Platform, NativeModules } from 'react-native';

// Live Activity is iOS-only, Android uses notification
export async function startVideoScanActivity(
  videoName: string,
  videoDuration: number,
  scanMode: string
): Promise<{ activityId: string | null; success: boolean }> {
  if (Platform.OS === 'ios') {
    return VisionML.startVideoScanActivity(videoName, videoDuration, scanMode);
  } else {
    // Android: start foreground service notification
    return VisionML.startScanProgress(videoName, videoDuration, scanMode);
  }
}

export async function isLiveActivityAvailable(): Promise<boolean> {
  if (Platform.OS === 'ios') {
    return VisionML.isLiveActivityAvailable();
  }
  // Android always returns true (notification always available)
  return true;
}
```

## Hardware Acceleration

| Platform | Accelerator | Implementation |
|----------|-------------|----------------|
| iOS | CoreML (ANE) | Built into ONNX Runtime iOS |
| Android | NNAPI | Add NNAPI execution provider |

**Android NNAPI setup:**
```kotlin
val sessionOptions = OrtSession.SessionOptions()
sessionOptions.addNnapi() // Enable NNAPI for GPU/NPU acceleration
```

## Testing Strategy

1. **Unit tests** for pure logic (YOLOParser, NMS)
2. **Instrumentation tests** for ONNX inference
3. **Integration tests** with sample videos
4. **Cross-platform parity tests** - same image should produce same detections (within tolerance)

## Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: ONNX Core | 2-3 days | None |
| Phase 2: ML Kit | 2 days | Phase 1 |
| Phase 3: Video Analysis | 3-4 days | Phase 1, 2 |
| Phase 4: Progress Service | 1 day | None |
| Phase 5: RN Bridge | 1 day | Phase 1-4 |
| **Total** | **~10 days** | |

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| NNAPI performance varies by device | Medium | Fall back to CPU, test on multiple devices |
| ML Kit results differ from Vision | Low | Both are mature APIs, minor normalization needed |
| MediaMetadataRetriever limitations | Low | Can use MediaCodec for more control if needed |
| Large APK size (ONNX Runtime) | Medium | Use ABI splits, consider lite runtime |

## Notes

- **No architectural changes needed** - the TypeScript API stays the same
- **ONNX models are cross-platform** - same .onnx file works on both
- **ML Kit is well-documented** - Google provides extensive examples
- **Video frame extraction is simpler** on Android than iOS (MediaMetadataRetriever is easier than AVAssetImageGenerator)

## Getting Started

```bash
# After implementation, test with:
cd android
./gradlew assembleDebug

# Or test in example app:
cd example
npx react-native run-android
```
