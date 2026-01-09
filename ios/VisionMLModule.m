#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(VisionML, NSObject)

// Object-based detector API
RCT_EXTERN_METHOD(createDetector:(NSString *)modelPath
                  classLabels:(NSArray *)classLabels
                  inputSize:(NSNumber *)inputSize
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(detect:(NSString *)detectorId
                  imageUri:(NSString *)imageUri
                  confidenceThreshold:(NSNumber *)confidenceThreshold
                  iouThreshold:(NSNumber *)iouThreshold
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(disposeDetector:(NSString *)detectorId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(disposeAllDetectors:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

// Live Activity Methods
RCT_EXTERN_METHOD(isLiveActivityAvailable:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(startVideoScanActivity:(NSString *)videoName
                  videoDuration:(NSNumber *)videoDuration
                  scanMode:(NSString *)scanMode
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(updateVideoScanActivity:(NSNumber *)progress
                  phase:(NSString *)phase
                  nsfwCount:(NSNumber *)nsfwCount
                  framesAnalyzed:(NSNumber *)framesAnalyzed
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(endVideoScanActivity:(NSNumber *)nsfwCount
                  framesAnalyzed:(NSNumber *)framesAnalyzed
                  isNSFW:(BOOL)isNSFW
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

// Video Analysis Methods
RCT_EXTERN_METHOD(analyzeVideo:(NSString *)detectorId
                  assetId:(NSString *)assetId
                  mode:(NSString *)mode
                  sampleInterval:(NSNumber *)sampleInterval
                  confidenceThreshold:(NSNumber *)confidenceThreshold
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(quickCheckVideo:(NSString *)detectorId
                  assetId:(NSString *)assetId
                  confidenceThreshold:(NSNumber *)confidenceThreshold
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

// Vision Framework Methods
RCT_EXTERN_METHOD(analyzeAnimals:(NSString *)assetId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(analyzeHumanPose:(NSString *)assetId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(analyzeComprehensive:(NSString *)assetId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

// Sensitive Content Analysis Methods (iOS 17+)
RCT_EXTERN_METHOD(getSensitiveContentAnalysisStatus:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(openSensitiveContentSettings:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(analyzeSensitiveContent:(NSString *)assetId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(batchAnalyzeSensitiveContent:(NSArray *)assetIds
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

// SCA Video Analysis Methods (iOS 17+)
RCT_EXTERN_METHOD(analyzeVideoSensitiveContent:(NSString *)assetId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(batchAnalyzeVideosSensitiveContent:(NSArray *)assetIds
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

+ (BOOL)requiresMainQueueSetup {
  return NO;
}

@end
