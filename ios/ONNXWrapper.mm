#import "ONNXWrapper.h"
#import <onnxruntime_objc/ort_session.h>
#import <onnxruntime_objc/ort_env.h>
#import <onnxruntime_objc/ort_value.h>
#import <onnxruntime_objc/ort_coreml_execution_provider.h>

@interface ONNXWrapper ()
@property (nonatomic, strong) ORTEnv *environment;
@property (nonatomic, strong) ORTSession *session;
@property (nonatomic, assign) NSInteger inputWidth;
@property (nonatomic, assign) NSInteger inputHeight;
@end

@implementation ONNXWrapper

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
    self = [super init];
    if (self) {
        // Create ONNX Runtime environment
        NSError *envError = nil;
        _environment = [[ORTEnv alloc] initWithLoggingLevel:ORTLoggingLevelWarning error:&envError];
        if (envError) {
            if (error) *error = envError;
            return nil;
        }

        // Create session options with CoreML acceleration
        NSError *optionsError = nil;
        ORTSessionOptions *sessionOptions = [[ORTSessionOptions alloc] initWithError:&optionsError];
        if (optionsError) {
            if (error) *error = optionsError;
            return nil;
        }

        // Try to add CoreML execution provider for GPU acceleration
        NSError *coremlError = nil;
        ORTCoreMLExecutionProviderOptions *coremlOptions = [[ORTCoreMLExecutionProviderOptions alloc] init];
        coremlOptions.enableOnSubgraphs = YES;
        [sessionOptions appendCoreMLExecutionProviderWithOptions:coremlOptions error:&coremlError];
        // CoreML provider is optional - continue even if it fails

        // Create session
        NSError *sessionError = nil;
        _session = [[ORTSession alloc] initWithEnv:_environment
                                         modelPath:modelPath
                                    sessionOptions:sessionOptions
                                             error:&sessionError];
        if (sessionError) {
            if (error) *error = sessionError;
            return nil;
        }

        // Extract input dimensions from model metadata
        NSError *inputError = nil;
        NSArray<ORTValue *> *inputNames = [_session inputNamesWithError:&inputError];
        if (!inputError && inputNames.count > 0) {
            // Default dimensions for YOLO models
            _inputWidth = 320;
            _inputHeight = 320;
        }
    }
    return self;
}

- (BOOL)isModelLoaded {
    return _session != nil;
}

- (nullable NSArray<NSNumber *> *)runInferenceWithInputData:(NSArray<NSNumber *> *)inputData
                                                 inputShape:(NSArray<NSNumber *> *)inputShape
                                                      error:(NSError **)error {
    if (!_session) {
        if (error) {
            *error = [NSError errorWithDomain:@"ONNXWrapper"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey: @"Model not loaded"}];
        }
        return nil;
    }

    // Convert input shape to int64_t array
    NSUInteger shapeCount = inputShape.count;
    int64_t *shape = (int64_t *)malloc(shapeCount * sizeof(int64_t));
    for (NSUInteger i = 0; i < shapeCount; i++) {
        shape[i] = inputShape[i].longLongValue;
    }

    // Calculate total elements
    int64_t totalElements = 1;
    for (NSUInteger i = 0; i < shapeCount; i++) {
        totalElements *= shape[i];
    }

    // Convert input data to float array
    float *inputFloats = (float *)malloc(totalElements * sizeof(float));
    for (NSInteger i = 0; i < totalElements; i++) {
        inputFloats[i] = inputData[i].floatValue;
    }

    // Create input tensor
    NSError *tensorError = nil;
    NSMutableData *inputMutableData = [NSMutableData dataWithBytes:inputFloats length:totalElements * sizeof(float)];
    ORTValue *inputTensor = [[ORTValue alloc] initWithTensorData:inputMutableData
                                                     elementType:ORTTensorElementDataTypeFloat
                                                           shape:inputShape
                                                           error:&tensorError];
    free(inputFloats);
    free(shape);

    if (tensorError) {
        if (error) *error = tensorError;
        return nil;
    }

    // Get input name
    NSError *nameError = nil;
    NSArray<NSString *> *inputNames = [_session inputNamesWithError:&nameError];
    if (nameError || inputNames.count == 0) {
        if (error) *error = nameError ?: [NSError errorWithDomain:@"ONNXWrapper" code:2 userInfo:@{NSLocalizedDescriptionKey: @"No input names"}];
        return nil;
    }

    // Run inference
    NSError *runError = nil;
    NSDictionary<NSString *, ORTValue *> *inputs = @{inputNames[0]: inputTensor};
    NSArray<ORTValue *> *outputs = [_session runWithInputs:inputs
                                               outputNames:nil
                                             runOptions:nil
                                                  error:&runError];
    if (runError) {
        if (error) *error = runError;
        return nil;
    }

    if (outputs.count == 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"ONNXWrapper" code:3 userInfo:@{NSLocalizedDescriptionKey: @"No outputs"}];
        }
        return nil;
    }

    // Extract output tensor data
    NSError *outputError = nil;
    ORTValue *outputTensor = outputs[0];
    NSData *outputData = [outputTensor tensorDataWithError:&outputError];
    if (outputError) {
        if (error) *error = outputError;
        return nil;
    }

    // Convert to NSArray<NSNumber *>
    const float *outputFloats = (const float *)outputData.bytes;
    NSUInteger outputCount = outputData.length / sizeof(float);
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:outputCount];
    for (NSUInteger i = 0; i < outputCount; i++) {
        [result addObject:@(outputFloats[i])];
    }

    return result;
}

@end
