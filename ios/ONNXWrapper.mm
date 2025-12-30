#import "ONNXWrapper.h"
#include "onnxruntime/coreml_provider_factory.h"
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <vector>

@interface ONNXWrapper () {
    std::unique_ptr<Ort::Env> _env;
    std::unique_ptr<Ort::Session> _session;
    Ort::AllocatorWithDefaultOptions _allocator;
}
@property (nonatomic, assign) NSInteger inputWidth;
@property (nonatomic, assign) NSInteger inputHeight;
@end

@implementation ONNXWrapper

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
    self = [super init];
    if (self) {
        try {
            // Create ONNX Runtime environment
            _env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "VisionML");

            // Create session options with CoreML acceleration
            Ort::SessionOptions sessionOptions;
            sessionOptions.SetIntraOpNumThreads(1);
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // Try to add CoreML execution provider for GPU acceleration
            try {
                uint32_t coreml_flags = 0;
                coreml_flags |= COREML_FLAG_ENABLE_ON_SUBGRAPH;
                OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptions, coreml_flags);
            } catch (...) {
                // CoreML provider is optional - continue even if it fails
                NSLog(@"[ONNXWrapper] CoreML provider not available, using CPU");
            }

            // Create session
            std::string pathStr = [modelPath UTF8String];
            _session = std::make_unique<Ort::Session>(*_env, pathStr.c_str(), sessionOptions);

            // Default dimensions for YOLO models
            _inputWidth = 320;
            _inputHeight = 320;

            NSLog(@"[ONNXWrapper] Model loaded successfully");

        } catch (const Ort::Exception& e) {
            if (error) {
                *error = [NSError errorWithDomain:@"ONNXWrapper"
                                             code:1
                                         userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithUTF8String:e.what()]}];
            }
            return nil;
        } catch (const std::exception& e) {
            if (error) {
                *error = [NSError errorWithDomain:@"ONNXWrapper"
                                             code:2
                                         userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithUTF8String:e.what()]}];
            }
            return nil;
        }
    }
    return self;
}

- (BOOL)isModelLoaded {
    return _session != nullptr;
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

    try {
        // Convert input shape to int64_t vector
        std::vector<int64_t> shape;
        for (NSNumber *dim in inputShape) {
            shape.push_back([dim longLongValue]);
        }

        // Calculate total elements
        size_t totalElements = 1;
        for (int64_t dim : shape) {
            totalElements *= dim;
        }

        // Convert input data to float vector
        std::vector<float> inputFloats(totalElements);
        for (size_t i = 0; i < totalElements; i++) {
            inputFloats[i] = [inputData[i] floatValue];
        }

        // Create memory info
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Create input tensor
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputFloats.data(),
            totalElements,
            shape.data(),
            shape.size()
        );

        // Get input/output names
        Ort::AllocatedStringPtr inputNamePtr = _session->GetInputNameAllocated(0, _allocator);
        Ort::AllocatedStringPtr outputNamePtr = _session->GetOutputNameAllocated(0, _allocator);

        const char* inputNames[] = {inputNamePtr.get()};
        const char* outputNames[] = {outputNamePtr.get()};

        // Run inference
        std::vector<Ort::Value> outputs = _session->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1
        );

        if (outputs.empty()) {
            if (error) {
                *error = [NSError errorWithDomain:@"ONNXWrapper"
                                             code:3
                                         userInfo:@{NSLocalizedDescriptionKey: @"No outputs from inference"}];
            }
            return nil;
        }

        // Extract output data
        Ort::Value& outputTensor = outputs[0];
        auto typeInfo = outputTensor.GetTensorTypeAndShapeInfo();
        size_t outputCount = typeInfo.GetElementCount();

        float* outputData = outputTensor.GetTensorMutableData<float>();

        // Convert to NSArray<NSNumber *>
        NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:outputCount];
        for (size_t i = 0; i < outputCount; i++) {
            [result addObject:@(outputData[i])];
        }

        return result;

    } catch (const Ort::Exception& e) {
        if (error) {
            *error = [NSError errorWithDomain:@"ONNXWrapper"
                                         code:4
                                     userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithUTF8String:e.what()]}];
        }
        return nil;
    } catch (const std::exception& e) {
        if (error) {
            *error = [NSError errorWithDomain:@"ONNXWrapper"
                                         code:5
                                     userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithUTF8String:e.what()]}];
        }
        return nil;
    }
}

- (void)dealloc {
    _session.reset();
    _env.reset();
}

@end
