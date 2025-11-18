import { NativeModules } from 'react-native';

const { VisionML } = NativeModules;

/**
 * Detection result with bounding box coordinates
 * Note: Box coordinates are in original image space, not model input space
 */
export interface Detection {
  /** Bounding box [x1, y1, x2, y2] in original image coordinates */
  box: [number, number, number, number];
  /** Confidence score (0.0 - 1.0) */
  score: number;
  /** Class index */
  classIndex: number;
  /** Class label */
  className: string;
}

/**
 * Inference result with detections and timing information
 */
export interface InferenceResult {
  /** Array of detected objects with bounding boxes in original image space */
  detections: Detection[];
  /** ONNX inference time in milliseconds */
  inferenceTime: number;
  /** Post-processing time (YOLO parse + NMS) in milliseconds */
  postProcessTime: number;
  /** Total pipeline time in milliseconds */
  totalTime: number;
}

/**
 * Detection options
 */
export interface DetectionOptions {
  /** Minimum confidence threshold (default: 0.6) */
  confidenceThreshold?: number;
  /** IoU threshold for NMS (default: 0.45) */
  iouThreshold?: number;
}

/**
 * Load ONNX model for inference
 * @param modelPath - Absolute path to .onnx model file
 * @param classLabels - Array of class label strings
 * @param inputSize - Model input size (default: 320)
 */
export async function loadModel(
  modelPath: string,
  classLabels: string[],
  inputSize: number = 320
): Promise<{ success: boolean; message: string }> {
  return VisionML.loadModel(modelPath, classLabels, inputSize);
}

/**
 * Detect objects in image using ONNX model
 *
 * Full native pipeline:
 * 1. Decode image (HEIC/JPEG supported)
 * 2. Resize to model input size (hardware-accelerated)
 * 3. Convert to NCHW tensor format
 * 4. Run ONNX inference with CoreML acceleration
 * 5. Parse YOLO output (25,200 predictions)
 * 6. Apply Non-Maximum Suppression
 *
 * @param imageUri - file:// URI to image
 * @param options - Detection options (confidence/IoU thresholds)
 * @returns Inference result with detections in original image coordinates
 */
export async function detect(
  imageUri: string,
  options: DetectionOptions = {}
): Promise<InferenceResult> {
  const {
    confidenceThreshold = 0.6,
    iouThreshold = 0.45
  } = options;

  return VisionML.detect(imageUri, confidenceThreshold, iouThreshold);
}

/**
 * Dispose of ONNX session and free resources
 */
export async function dispose(): Promise<{ success: boolean }> {
  return VisionML.dispose();
}

export default {
  loadModel,
  detect,
  dispose
};
