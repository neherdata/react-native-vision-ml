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

// MARK: - Vision Framework Types

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Label {
  identifier: string;
  confidence: number;
}

export interface AnimalDetection {
  boundingBox: BoundingBox;
  confidence: number;
  labels: Label[];
}

export interface AnimalAnalysis {
  animals: AnimalDetection[];
  count: number;
}

export interface BodyPoint {
  x: number;
  y: number;
  confidence: number;
}

export interface HumanPose {
  points: Record<string, BodyPoint>;
  pointCount: number;
}

export interface HumanPoseAnalysis {
  humans: HumanPose[];
  humanCount: number;
}

export interface FaceDetection {
  boundingBox: BoundingBox;
  confidence: number;
}

export interface ComprehensiveAnalysis {
  // Scene Classification (VNClassifyImageRequest)
  scenes?: Array<{ identifier: string; confidence: number }>;

  // Face Detection
  faces?: FaceDetection[];
  faceCount?: number;

  // Animal Detection (VNRecognizeAnimalsRequest)
  animals?: AnimalDetection[];
  animalCount?: number;

  // Human Pose Detection (VNDetectHumanBodyPoseRequest)
  humanCount?: number;
  hasHumans?: boolean;

  // Text Detection
  hasText?: boolean;
  textRegions?: number;

  // Rectangle Detection (screenshots, documents)
  rectangles?: number;
  likelyScreenshot?: boolean;
}

/**
 * Detect animals in photos using VNRecognizeAnimalsRequest
 * Currently supports cats and dogs with bounding boxes
 *
 * Use cases:
 * - Filter pet photos for batch scanning
 * - Identify photos that are unlikely to be NSFW
 * - Provide additional metadata for photo organization
 *
 * @param assetId - Photo asset identifier from MediaLibrary
 * @returns Animal detection results with bounding boxes and labels
 */
export async function analyzeAnimals(assetId: string): Promise<AnimalAnalysis> {
  return VisionML.analyzeAnimals(assetId);
}

/**
 * Detect human body poses using VNDetectHumanBodyPoseRequest
 * Returns up to 19 body joint points with confidence scores
 *
 * Use cases:
 * - Detect fitness/exercise photos
 * - Identify dance or sports photos
 * - Provide additional context for scene understanding
 *
 * Note: Does not work in iOS Simulator, requires physical device
 *
 * @param assetId - Photo asset identifier from MediaLibrary
 * @returns Human pose analysis with joint points
 */
export async function analyzeHumanPose(assetId: string): Promise<HumanPoseAnalysis> {
  return VisionML.analyzeHumanPose(assetId);
}

/**
 * Comprehensive Vision framework analysis
 * Runs all available Vision detectors in parallel:
 * - VNClassifyImageRequest (scene classification, ~1000 categories)
 * - VNDetectFaceRectanglesRequest (face detection)
 * - VNRecognizeAnimalsRequest (cat/dog detection)
 * - VNDetectHumanBodyPoseRequest (pose estimation)
 * - VNRecognizeTextRequest (text detection)
 * - VNDetectRectanglesRequest (screenshot detection)
 *
 * Use cases:
 * - Deep photo analysis for categorization
 * - Pre-screening before API calls
 * - Rich metadata extraction for search
 *
 * Performance: ~200-400ms on modern devices
 *
 * @param assetId - Photo asset identifier from MediaLibrary
 * @returns Comprehensive analysis results
 */
export async function analyzeComprehensive(assetId: string): Promise<ComprehensiveAnalysis> {
  return VisionML.analyzeComprehensive(assetId);
}

export default {
  loadModel,
  detect,
  dispose,
  analyzeAnimals,
  analyzeHumanPose,
  analyzeComprehensive
};
