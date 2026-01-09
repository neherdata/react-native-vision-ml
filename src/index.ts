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
 * Result from createDetector
 */
export interface CreateDetectorResult {
  /** Unique detector instance ID */
  detectorId: string;
  /** Success status */
  success: boolean;
  /** Status message */
  message: string;
}

/**
 * Create a new ONNX detector instance
 * @param modelPath - Absolute path to .onnx model file
 * @param classLabels - Array of class label strings
 * @param inputSize - Model input size (default: 320)
 * @returns Detector instance with unique ID
 */
export async function createDetector(
  modelPath: string,
  classLabels: string[],
  inputSize: number = 320
): Promise<CreateDetectorResult> {
  return VisionML.createDetector(modelPath, classLabels, inputSize);
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
 * @param detectorId - Detector instance ID from createDetector()
 * @param imageUri - file:// URI to image
 * @param options - Detection options (confidence/IoU thresholds)
 * @returns Inference result with detections in original image coordinates
 */
export async function detect(
  detectorId: string,
  imageUri: string,
  options: DetectionOptions = {}
): Promise<InferenceResult> {
  const {
    confidenceThreshold = 0.6,
    iouThreshold = 0.45
  } = options;

  return VisionML.detect(detectorId, imageUri, confidenceThreshold, iouThreshold);
}

/**
 * Dispose of a specific detector instance and free resources
 * @param detectorId - Detector instance ID to dispose
 */
export async function disposeDetector(detectorId: string): Promise<{ success: boolean }> {
  return VisionML.disposeDetector(detectorId);
}

/**
 * Dispose of all detector instances and free resources
 */
export async function disposeAllDetectors(): Promise<{ success: boolean }> {
  return VisionML.disposeAllDetectors();
}

// MARK: - Live Activity Types

/**
 * Result from starting a Live Activity
 */
export interface LiveActivityResult {
  /** Activity ID if started successfully */
  activityId: string | null;
  /** Whether the activity was started */
  success: boolean;
}

/**
 * Check if Live Activities are available on this device
 * Requires iOS 16.1+ and user permission
 */
export async function isLiveActivityAvailable(): Promise<boolean> {
  return VisionML.isLiveActivityAvailable();
}

/**
 * Start a Live Activity for video scanning
 * Shows progress on Dynamic Island and Lock Screen
 *
 * @param videoName - Display name of the video
 * @param videoDuration - Duration in seconds
 * @param scanMode - Scan mode being used
 * @returns Activity ID and success status
 */
export async function startVideoScanActivity(
  videoName: string,
  videoDuration: number,
  scanMode: VideoScanMode
): Promise<LiveActivityResult> {
  return VisionML.startVideoScanActivity(videoName, videoDuration, scanMode);
}

/**
 * Update Live Activity progress
 *
 * @param progress - Progress from 0.0 to 1.0
 * @param phase - Current phase description
 * @param nsfwCount - Number of NSFW frames found so far
 * @param framesAnalyzed - Total frames analyzed so far
 */
export async function updateVideoScanActivity(
  progress: number,
  phase: string,
  nsfwCount: number,
  framesAnalyzed: number
): Promise<boolean> {
  return VisionML.updateVideoScanActivity(progress, phase, nsfwCount, framesAnalyzed);
}

/**
 * End Live Activity with final results
 *
 * @param nsfwCount - Final count of NSFW frames
 * @param framesAnalyzed - Total frames analyzed
 * @param isNSFW - Whether any NSFW content was found
 */
export async function endVideoScanActivity(
  nsfwCount: number,
  framesAnalyzed: number,
  isNSFW: boolean
): Promise<boolean> {
  return VisionML.endVideoScanActivity(nsfwCount, framesAnalyzed, isNSFW);
}

// MARK: - Video Analysis Types

/**
 * Video scan mode determines sampling strategy
 */
export type VideoScanMode =
  | 'quick_check'         // Just check start, middle, end (3 frames, fastest)
  | 'sampled'             // Check at regular intervals
  | 'thorough'            // Vision human detection first, then ONNX only on human frames (best for newer devices)
  | 'binary_search'       // Start at middle, expand outward (fallback for older devices)
  | 'full_short_circuit'; // Check every N seconds, stop at first detection

/**
 * Result from video analysis
 */
export interface VideoAnalysisResult {
  /** Whether any NSFW content was detected */
  isNSFW: boolean;
  /** Number of frames with NSFW content */
  nsfwFrameCount: number;
  /** Total frames that were analyzed */
  totalFramesAnalyzed: number;
  /** Timestamp of first NSFW frame (seconds), null if none found */
  firstNSFWTimestamp: number | null;
  /** All timestamps where NSFW was detected (seconds) */
  nsfwTimestamps: number[];
  /** Highest NSFW confidence score found */
  highestConfidence: number;
  /** Total processing time in milliseconds */
  totalProcessingTime: number;
  /** Video duration in seconds */
  videoDuration: number;
  /** Scan mode that was used */
  scanMode: string;
  /** Frames where Vision detected humans (thorough mode only) */
  humanFramesDetected: number;
}

/**
 * Options for video analysis
 */
export interface VideoAnalysisOptions {
  /** Scan mode (default: 'sampled') */
  mode?: VideoScanMode;
  /** Seconds between samples for 'sampled' and 'full_short_circuit' modes (default: 5.0) */
  sampleInterval?: number;
  /** Minimum confidence threshold (default: 0.6) */
  confidenceThreshold?: number;
}

/**
 * Analyze a video for NSFW content
 *
 * Uses Apple Vision framework for human detection pre-filtering,
 * then runs ONNX NSFW detection only on frames with humans.
 * Memory-efficient: processes one frame at a time.
 *
 * Scan Modes:
 * - 'quick_check': Just check start, middle, end (3 frames, fastest)
 * - 'sampled': Sample at regular intervals, analyze all samples
 * - 'thorough': Use Vision to find human frames first, then ONNX those (best for newer devices)
 * - 'binary_search': Start at middle, expand when NSFW found (fallback for older devices)
 * - 'full_short_circuit': Sample every N seconds, stop at first detection
 *
 * The 'thorough' mode is recommended for newer devices as it:
 * 1. Uses fast Vision framework to scan all frames for humans (~10ms/frame)
 * 2. Only runs expensive ONNX on frames with humans (~200ms/frame)
 * 3. Can skip 60-90% of frames in typical videos
 *
 * Use 'binary_search' on older devices where Vision framework is slow.
 *
 * @param detectorId - Detector instance ID from createDetector()
 * @param assetId - Video asset identifier from MediaLibrary
 * @param options - Analysis options
 * @returns Video analysis result with humanFramesDetected count
 */
export async function analyzeVideo(
  detectorId: string,
  assetId: string,
  options: VideoAnalysisOptions = {}
): Promise<VideoAnalysisResult> {
  const {
    mode = 'sampled',
    sampleInterval = 5.0,
    confidenceThreshold = 0.6
  } = options;

  return VisionML.analyzeVideo(
    detectorId,
    assetId,
    mode,
    sampleInterval,
    confidenceThreshold
  );
}

/**
 * Quick check a video (start, middle, end only)
 *
 * Fastest video analysis - only checks 3 frames.
 * Good for initial screening, may miss content in between.
 *
 * @param detectorId - Detector instance ID from createDetector()
 * @param assetId - Video asset identifier from MediaLibrary
 * @param confidenceThreshold - Minimum confidence threshold (default: 0.6)
 * @returns Video analysis result
 */
export async function quickCheckVideo(
  detectorId: string,
  assetId: string,
  confidenceThreshold: number = 0.6
): Promise<VideoAnalysisResult> {
  return VisionML.quickCheckVideo(detectorId, assetId, confidenceThreshold);
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

// MARK: - Sensitive Content Analysis Types (iOS 17+)

/**
 * Status of Apple's Sensitive Content Analysis feature
 */
export interface SensitiveContentAnalysisStatus {
  /** Whether SCA is available on this iOS version */
  available: boolean;
  /** Whether the user has enabled Sensitive Content Warning in Settings */
  enabled: boolean;
  /** Policy level: 'disabled', 'simple_interventions', 'descriptive_interventions', 'unsupported' */
  policy: 'disabled' | 'simple_interventions' | 'descriptive_interventions' | 'unsupported' | 'unknown';
  /** URL scheme to open Settings (may not work on all iOS versions) */
  settingsURL?: string;
  /** iOS version string */
  iosVersion: string;
  /** Reason if not available */
  reason?: string;
}

/**
 * Result from opening Settings
 */
export interface OpenSettingsResult {
  /** Whether Settings was opened */
  opened: boolean;
  /** Which URL was used */
  url: string | null;
}

/**
 * Result from single image sensitive content analysis
 */
export interface SensitiveContentResult {
  /** Whether SCA was available to use */
  available: boolean;
  /** Whether the image contains sensitive content */
  isSensitive: boolean;
  /** Asset ID that was analyzed */
  assetId?: string;
  /** Reason if not available: 'disabled_by_user', 'ios_version', 'framework_unavailable' */
  reason?: 'disabled_by_user' | 'ios_version' | 'framework_unavailable';
}

/**
 * Single result in batch analysis
 */
export interface BatchSensitiveContentItem {
  /** Asset ID */
  assetId: string;
  /** Whether sensitive content was detected */
  isSensitive: boolean;
  /** Error message if analysis failed */
  error?: string;
}

/**
 * Result from batch sensitive content analysis
 */
export interface BatchSensitiveContentResult {
  /** Whether SCA was available to use */
  available: boolean;
  /** Array of results for each asset */
  results: BatchSensitiveContentItem[];
  /** Total number of assets analyzed */
  totalAnalyzed?: number;
  /** Number of sensitive items found */
  sensitiveCount?: number;
  /** Reason if not available */
  reason?: 'disabled_by_user' | 'ios_version' | 'framework_unavailable';
}

/**
 * Get the status of Apple's Sensitive Content Analysis feature
 *
 * Returns whether SCA is available and enabled. The user must enable
 * "Sensitive Content Warning" in iOS Settings > Privacy & Security > Sensitive Content Warning
 *
 * @returns Status object with availability, policy, and settings URL
 */
export async function getSensitiveContentAnalysisStatus(): Promise<SensitiveContentAnalysisStatus> {
  return VisionML.getSensitiveContentAnalysisStatus();
}

/**
 * Open iOS Settings to the Sensitive Content Warning section
 *
 * Attempts to deep link directly to the Sensitive Content Warning toggle.
 * Falls back to Privacy settings, then general Settings.
 *
 * @returns Whether Settings was opened and which URL was used
 */
export async function openSensitiveContentSettings(): Promise<OpenSettingsResult> {
  return VisionML.openSensitiveContentSettings();
}

/**
 * Analyze a single image for sensitive content using Apple's SCA
 *
 * This is MUCH faster than ONNX (~20ms vs ~150ms) but only returns
 * a boolean result - no bounding boxes or confidence scores.
 *
 * Perfect for pre-filtering: if SCA says "not sensitive", skip ONNX.
 * If SCA says "sensitive", run ONNX for detailed detection.
 *
 * Requires iOS 17+ and user to enable Sensitive Content Warning in Settings.
 *
 * @param assetId - Photo asset identifier from MediaLibrary
 * @returns Analysis result with isSensitive boolean
 */
export async function analyzeSensitiveContent(assetId: string): Promise<SensitiveContentResult> {
  return VisionML.analyzeSensitiveContent(assetId);
}

/**
 * Batch analyze multiple images for sensitive content
 *
 * Uses Apple's SCA for fast pre-filtering. Process results to identify
 * which photos need detailed ONNX scanning.
 *
 * Typical workflow:
 * 1. Get all photo IDs from library
 * 2. Run batchAnalyzeSensitiveContent() - very fast
 * 3. Filter to only sensitiveCount photos
 * 4. Run ONNX only on those for bounding boxes
 *
 * This can skip 60-80% of photos that are clearly clean.
 *
 * @param assetIds - Array of photo asset identifiers
 * @returns Batch results with per-photo analysis
 */
export async function batchAnalyzeSensitiveContent(
  assetIds: string[]
): Promise<BatchSensitiveContentResult> {
  return VisionML.batchAnalyzeSensitiveContent(assetIds);
}

export default {
  createDetector,
  detect,
  disposeDetector,
  disposeAllDetectors,
  // Live Activity
  isLiveActivityAvailable,
  startVideoScanActivity,
  updateVideoScanActivity,
  endVideoScanActivity,
  // Video Analysis
  analyzeVideo,
  quickCheckVideo,
  // Vision Framework
  analyzeAnimals,
  analyzeHumanPose,
  analyzeComprehensive,
  // Sensitive Content Analysis (iOS 17+)
  getSensitiveContentAnalysisStatus,
  openSensitiveContentSettings,
  analyzeSensitiveContent,
  batchAnalyzeSensitiveContent
};
