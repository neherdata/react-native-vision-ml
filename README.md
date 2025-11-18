# react-native-vision-ml

Native ONNX Runtime inference with integrated YOLO post-processing for React Native iOS.

## Features

- **Fully Native Pipeline**: Entire ML inference pipeline runs in native Swift
- **Hardware Acceleration**: CoreML execution provider for ONNX Runtime
- **Complete Processing**: Image decode → resize → inference → YOLO parse → NMS
- **Original Image Coordinates**: Bounding boxes scaled to original image dimensions
- **High Performance**: Eliminates bridge overhead by processing 2.1M floats natively
- **HEIC/JPEG Support**: Hardware-accelerated image decoding and resizing

## Architecture

### Full Native Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                   Native Swift Layer                     │
├─────────────────────────────────────────────────────────┤
│ 1. ImageDecoder                                          │
│    • Decode HEIC/JPEG (UIImage)                         │
│    • Hardware-accelerated resize (CGContext)            │
│    • Normalize to Float32 RGB (0.0-1.0)                 │
│    • Capture original dimensions                        │
├─────────────────────────────────────────────────────────┤
│ 2. ONNX Inference                                        │
│    • Convert HWC → NCHW tensor format                   │
│    • CoreML-accelerated inference                       │
│    • Output: [1, 25200, 85] tensor                      │
├─────────────────────────────────────────────────────────┤
│ 3. YOLO Parser                                           │
│    • Parse 25,200 predictions                           │
│    • Filter by confidence threshold                     │
│    • Scale boxes to original image dimensions           │
├─────────────────────────────────────────────────────────┤
│ 4. Non-Maximum Suppression                              │
│    • Sort by confidence score                           │
│    • Calculate IoU for overlapping boxes                │
│    • Suppress duplicates                                │
└─────────────────────────────────────────────────────────┘
                           ↓
              Return ~5-10 final detections
                    (original image coordinates)
```

### Performance Benefits

**Before (JS post-processing):**
- Bridge: 307,200 floats → JS
- Bridge: 2,142,000 floats → JS
- JS: Parse 25,200 predictions → **TIMEOUT**

**After (native post-processing):**
- Native: All processing
- Bridge: ~5-10 detection objects → JS
- **Total: <170ms** (decode ~30ms + inference ~73ms + post-process ~20ms)

## Installation

```bash
npm install react-native-vision-ml
# or
yarn add react-native-vision-ml
```

### iOS Setup

```bash
cd ios
pod install
```

The module requires ONNX Runtime Objective-C library, which will be installed automatically via CocoaPods.

## Usage

### 1. Load Model

```typescript
import { loadModel, detect } from 'react-native-vision-ml';

// NSFW detection class labels
const classLabels = [
  'FEMALE_GENITALIA_COVERED',
  'FACE_FEMALE',
  'BUTTOCKS_EXPOSED',
  'FEMALE_BREAST_EXPOSED',
  'FEMALE_GENITALIA_EXPOSED',
  'MALE_BREAST_EXPOSED',
  'ANUS_EXPOSED',
  'FEET_EXPOSED',
  'BELLY_COVERED',
  'FEET_COVERED',
  'ARMPITS_COVERED',
  'ARMPITS_EXPOSED',
  'FACE_MALE',
  'BELLY_EXPOSED',
  'MALE_GENITALIA_EXPOSED',
  'ANUS_COVERED',
  'FEMALE_BREAST_COVERED',
  'BUTTOCKS_COVERED'
];

// Load ONNX model
await loadModel(
  '/path/to/model.onnx',
  classLabels,
  320  // input size
);
```

### 2. Run Detection

```typescript
const result = await detect(
  'file:///path/to/image.jpg',
  {
    confidenceThreshold: 0.6,  // minimum confidence
    iouThreshold: 0.45          // NMS IoU threshold
  }
);

console.log('Total time:', result.totalTime, 'ms');
console.log('Inference time:', result.inferenceTime, 'ms');
console.log('Post-process time:', result.postProcessTime, 'ms');

// Draw bounding boxes on original image
result.detections.forEach(detection => {
  const [x1, y1, x2, y2] = detection.box;  // Already in original image coordinates
  console.log(`${detection.className}: ${(detection.score * 100).toFixed(1)}% at [${x1},${y1}] → [${x2},${y2}]`);
});
```

### 3. Dispose Resources

```typescript
await dispose();
```

## API Reference

### `loadModel(modelPath, classLabels, inputSize)`

Load ONNX model for inference.

**Parameters:**
- `modelPath` (string): Absolute path to .onnx model file
- `classLabels` (string[]): Array of class label strings
- `inputSize` (number): Model input size (default: 320)

**Returns:** `Promise<{ success: boolean, message: string }>`

### `detect(imageUri, options)`

Detect objects in image using loaded ONNX model.

**Parameters:**
- `imageUri` (string): file:// URI to image
- `options` (object):
  - `confidenceThreshold` (number): Minimum confidence (0.0-1.0, default: 0.6)
  - `iouThreshold` (number): NMS IoU threshold (0.0-1.0, default: 0.45)

**Returns:** `Promise<InferenceResult>`

```typescript
interface InferenceResult {
  detections: Detection[];
  inferenceTime: number;      // ONNX inference time (ms)
  postProcessTime: number;    // YOLO parse + NMS time (ms)
  totalTime: number;          // Total pipeline time (ms)
}

interface Detection {
  box: [number, number, number, number];  // [x1, y1, x2, y2] in original image space
  score: number;              // Confidence (0.0-1.0)
  classIndex: number;         // Class index
  className: string;          // Class label
}
```

### `dispose()`

Dispose of ONNX session and free resources.

**Returns:** `Promise<{ success: boolean }>`

## Bounding Box Coordinates

⚠️ **Important**: All bounding box coordinates are returned in **original image space**, not model input space.

The module handles all scaling internally:
1. Original image is resized to model input size (e.g., 320x320)
2. ONNX inference runs on resized tensor
3. YOLO parser scales predictions back to original dimensions
4. You receive boxes ready to draw on the original image

Example:
```typescript
// Original image: 1920x1080
// Model input: 320x320
const result = await detect('file:///image.jpg');

result.detections.forEach(det => {
  const [x1, y1, x2, y2] = det.box;
  // Coordinates are in 1920x1080 space, not 320x320
  drawBox(originalImage, x1, y1, x2, y2);
});
```

## Performance

Typical performance on iPhone (iOS 17+):
- Image decode + resize (1920x1080 → 320x320): ~30ms
- ONNX inference (CoreML): ~70ms
- YOLO parsing (25,200 predictions): ~15ms
- NMS: ~5ms
- **Total: ~120ms**

## Technical Details

### Components

- **ImageDecoder**: UIImage-based decoder with CGContext hardware-accelerated resize
- **ONNXInference**: ONNX Runtime wrapper with CoreML execution provider
- **YOLOParser**: Parses YOLO v5/v8 output format [1, 25200, 85]
- **NMS**: Non-Maximum Suppression with IoU calculation

### Dependencies

- `onnxruntime-objc`: ~1.19.0
- iOS: 13.0+
- Swift: 5.0+

### Frameworks

- CoreML (hardware acceleration)
- Accelerate (SIMD operations)
- UIKit (image decoding)
- CoreGraphics (image rendering)

## License

MIT

## Author

Neher Data
