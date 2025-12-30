#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

/// Objective-C wrapper for ONNX Runtime to avoid Swift modular headers issue
@interface ONNXWrapper : NSObject

/// Initialize wrapper with model path
- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                     error:(NSError **)error;

/// Run inference on preprocessed input data
/// @param inputData Float array of preprocessed image data (NCHW format)
/// @param inputShape Shape of input tensor [1, 3, height, width]
/// @param error Error output if inference fails
/// @return Array of float arrays representing output tensor, or nil on error
- (nullable NSArray<NSNumber *> *)runInferenceWithInputData:(NSArray<NSNumber *> *)inputData
                                                 inputShape:(NSArray<NSNumber *> *)inputShape
                                                      error:(NSError **)error;

/// Check if model is loaded
@property (nonatomic, readonly) BOOL isModelLoaded;

/// Get input dimensions expected by the model
@property (nonatomic, readonly) NSInteger inputWidth;
@property (nonatomic, readonly) NSInteger inputHeight;

@end

NS_ASSUME_NONNULL_END
