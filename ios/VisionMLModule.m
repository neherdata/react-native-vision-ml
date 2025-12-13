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

RCT_EXTERN_METHOD(analyzeAnimals:(NSString *)assetId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(analyzeHumanPose:(NSString *)assetId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(analyzeComprehensive:(NSString *)assetId
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)

+ (BOOL)requiresMainQueueSetup {
  return NO;
}

@end
