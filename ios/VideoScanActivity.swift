import Foundation
import ActivityKit

/// Live Activity for video scanning progress
/// Shows on Dynamic Island and Lock Screen during video analysis
@available(iOS 16.1, *)
struct VideoScanActivityAttributes: ActivityAttributes {

  /// Static content that doesn't change during the activity
  public struct ContentState: Codable, Hashable {
    /// Current progress (0.0 to 1.0)
    var progress: Float
    /// Current phase description
    var phase: String
    /// Number of NSFW frames found so far
    var nsfwCount: Int
    /// Total frames analyzed so far
    var framesAnalyzed: Int
    /// Whether scanning is complete
    var isComplete: Bool
    /// Final result message (when complete)
    var resultMessage: String?
  }

  /// Video filename or identifier
  var videoName: String
  /// Video duration in seconds
  var videoDuration: Double
  /// Scan mode being used
  var scanMode: String
}

/// Manager for Video Scan Live Activities
@available(iOS 16.1, *)
class VideoScanActivityManager {

  static let shared = VideoScanActivityManager()

  private var currentActivity: Activity<VideoScanActivityAttributes>?

  private init() {}

  /// Start a new Live Activity for video scanning
  /// - Parameters:
  ///   - videoName: Display name of the video
  ///   - videoDuration: Duration in seconds
  ///   - scanMode: Scan mode being used
  /// - Returns: Activity ID if successful
  func startActivity(videoName: String, videoDuration: Double, scanMode: String) -> String? {

    // Check if Live Activities are supported
    guard ActivityAuthorizationInfo().areActivitiesEnabled else {
      NSLog("[VideoScanActivity] Live Activities not enabled")
      return nil
    }

    // End any existing activity
    endActivity()

    let attributes = VideoScanActivityAttributes(
      videoName: videoName,
      videoDuration: videoDuration,
      scanMode: scanMode
    )

    let initialState = VideoScanActivityAttributes.ContentState(
      progress: 0,
      phase: "Starting scan...",
      nsfwCount: 0,
      framesAnalyzed: 0,
      isComplete: false,
      resultMessage: nil
    )

    do {
      let activity = try Activity.request(
        attributes: attributes,
        contentState: initialState,
        pushType: nil  // No push updates, we update locally
      )

      currentActivity = activity
      NSLog("[VideoScanActivity] Started activity: %@", activity.id)
      return activity.id

    } catch {
      NSLog("[VideoScanActivity] Failed to start: %@", error.localizedDescription)
      return nil
    }
  }

  /// Update the Live Activity with current progress
  func updateProgress(
    progress: Float,
    phase: String,
    nsfwCount: Int,
    framesAnalyzed: Int
  ) {
    guard let activity = currentActivity else { return }

    let state = VideoScanActivityAttributes.ContentState(
      progress: progress,
      phase: phase,
      nsfwCount: nsfwCount,
      framesAnalyzed: framesAnalyzed,
      isComplete: false,
      resultMessage: nil
    )

    Task {
      await activity.update(using: state)
    }
  }

  /// Complete the Live Activity with results
  func completeActivity(
    nsfwCount: Int,
    framesAnalyzed: Int,
    isNSFW: Bool
  ) {
    guard let activity = currentActivity else { return }

    let resultMessage = isNSFW
      ? "Found \(nsfwCount) sensitive frame\(nsfwCount == 1 ? "" : "s")"
      : "No sensitive content found"

    let finalState = VideoScanActivityAttributes.ContentState(
      progress: 1.0,
      phase: "Complete",
      nsfwCount: nsfwCount,
      framesAnalyzed: framesAnalyzed,
      isComplete: true,
      resultMessage: resultMessage
    )

    Task {
      // Update to final state
      await activity.update(using: finalState)

      // End after a brief delay so user can see result
      try? await Task.sleep(nanoseconds: 3_000_000_000)  // 3 seconds
      await activity.end(dismissalPolicy: .immediate)

      await MainActor.run {
        self.currentActivity = nil
      }
    }
  }

  /// End the Live Activity immediately
  func endActivity() {
    guard let activity = currentActivity else { return }

    Task {
      await activity.end(dismissalPolicy: .immediate)
      await MainActor.run {
        self.currentActivity = nil
      }
    }
  }

  /// Check if Live Activities are available
  static var isAvailable: Bool {
    if #available(iOS 16.1, *) {
      return ActivityAuthorizationInfo().areActivitiesEnabled
    }
    return false
  }
}

// MARK: - VideoAnalyzer Delegate Integration

@available(iOS 16.1, *)
extension VideoScanActivityManager: VideoAnalyzerDelegate {

  func videoAnalyzer(_ analyzer: VideoAnalyzer, didUpdateProgress progress: Float) {
    let phase: String
    if progress < 0.5 {
      phase = "Scanning for humans..."
    } else {
      phase = "Analyzing content..."
    }

    // We don't have full context here, just update progress
    updateProgress(
      progress: progress,
      phase: phase,
      nsfwCount: 0,  // Updated separately when NSFW found
      framesAnalyzed: 0
    )
  }

  func videoAnalyzer(_ analyzer: VideoAnalyzer, didFindNSFWAt timestamp: Double, confidence: Float) {
    // This is called when NSFW is detected
    // The activity will show updated count on next progress update
    NSLog("[VideoScanActivity] NSFW found at %.1fs (%.0f%% confidence)", timestamp, confidence * 100)
  }

  func videoAnalyzer(_ analyzer: VideoAnalyzer, didComplete result: VideoAnalysisResult) {
    completeActivity(
      nsfwCount: result.nsfwFrameCount,
      framesAnalyzed: result.totalFramesAnalyzed,
      isNSFW: result.isNSFW
    )
  }
}
