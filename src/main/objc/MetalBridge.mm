#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "MetalBridge.h"

@interface MetalContext : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLLibrary> library;
@property (nonatomic, strong) NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines;
@end

@implementation MetalContext
@end

JNIEXPORT jlong JNICALL Java_hs_ml_math_MetalBackend_nativeInit(JNIEnv *env, jobject obj) {
    @autoreleasepool {
        MetalContext* context = [[MetalContext alloc] init];
        
        // Get the default Metal device
        context.device = MTLCreateSystemDefaultDevice();
        if (!context.device) {
            return 0; // Metal not available
        }
        
        // Create command queue
        context.commandQueue = [context.device newCommandQueue];
        
        // Load Metal library
        NSError* error = nil;
        NSString* metalPath = @"TensorOps.metal";
        NSString* metalSource = [NSString stringWithContentsOfFile:metalPath 
                                                           encoding:NSUTF8StringEncoding 
                                                              error:&error];
        
        if (error) {
            // Try loading from bundle or default library
            context.library = [context.device newDefaultLibrary];
        } else {
            context.library = [context.device newLibraryWithSource:metalSource 
                                                           options:nil 
                                                             error:&error];
        }
        
        if (!context.library) {
            return 0;
        }
        
        // Create pipeline states
        context.pipelines = [NSMutableDictionary dictionary];
        NSArray* functionNames = @[@"matmul", @"add", @"subtract", @"hadamard", 
                                   @"scalar_mul", @"transpose", @"relu", @"relu_grad"];
        
        for (NSString* name in functionNames) {
            id<MTLFunction> function = [context.library newFunctionWithName:name];
            if (function) {
                id<MTLComputePipelineState> pipeline = 
                    [context.device newComputePipelineStateWithFunction:function error:&error];
                if (pipeline) {
                    context.pipelines[name] = pipeline;
                }
            }
        }
        
        return (jlong)CFBridgingRetain(context);
    }
}

JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeRelease(JNIEnv *env, jobject obj, jlong handle) {
    @autoreleasepool {
        if (handle != 0) {
            CFBridgingRelease((void*)handle);
        }
    }
}

JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeMatMul(
    JNIEnv *env, jobject obj, jlong handle,
    jfloatArray a, jfloatArray b, jfloatArray c,
    jint m, jint k, jint n)
{
    @autoreleasepool {
        MetalContext* context = (__bridge MetalContext*)(void*)handle;
        
        // Get array data
        jfloat* aData = env->GetFloatArrayElements(a, NULL);
        jfloat* bData = env->GetFloatArrayElements(b, NULL);
        jfloat* cData = env->GetFloatArrayElements(c, NULL);
        
        // Create Metal buffers
        id<MTLBuffer> bufferA = [context.device newBufferWithBytes:aData 
                                                            length:m * k * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [context.device newBufferWithBytes:bData 
                                                            length:k * n * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [context.device newBufferWithLength:m * n * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        
        // Execute kernel
        id<MTLCommandBuffer> commandBuffer = [context.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        id<MTLComputePipelineState> pipeline = context.pipelines[@"matmul"];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        [encoder setBytes:&m length:sizeof(uint) atIndex:3];
        [encoder setBytes:&k length:sizeof(uint) atIndex:4];
        [encoder setBytes:&n length:sizeof(uint) atIndex:5];
        
        MTLSize gridSize = MTLSizeMake(n, m, 1);
        NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger w = sqrt(threadGroupSize);
        MTLSize threadgroupSize = MTLSizeMake(w, w, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy result back
        memcpy(cData, [bufferC contents], m * n * sizeof(float));
        
        // Release arrays
        env->ReleaseFloatArrayElements(a, aData, JNI_ABORT);
        env->ReleaseFloatArrayElements(b, bData, JNI_ABORT);
        env->ReleaseFloatArrayElements(c, cData, 0);
    }
}

JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeAdd(
    JNIEnv *env, jobject obj, jlong handle,
    jfloatArray a, jfloatArray b, jfloatArray c, jint size)
{
    @autoreleasepool {
        MetalContext* context = (__bridge MetalContext*)(void*)handle;
        
        jfloat* aData = env->GetFloatArrayElements(a, NULL);
        jfloat* bData = env->GetFloatArrayElements(b, NULL);
        jfloat* cData = env->GetFloatArrayElements(c, NULL);
        
        id<MTLBuffer> bufferA = [context.device newBufferWithBytes:aData 
                                                            length:size * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [context.device newBufferWithBytes:bData 
                                                            length:size * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [context.device newBufferWithLength:size * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> commandBuffer = [context.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        id<MTLComputePipelineState> pipeline = context.pipelines[@"add"];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, size);
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        memcpy(cData, [bufferC contents], size * sizeof(float));
        
        env->ReleaseFloatArrayElements(a, aData, JNI_ABORT);
        env->ReleaseFloatArrayElements(b, bData, JNI_ABORT);
        env->ReleaseFloatArrayElements(c, cData, 0);
    }
}

JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeSubtract(
    JNIEnv *env, jobject obj, jlong handle,
    jfloatArray a, jfloatArray b, jfloatArray c, jint size)
{
    @autoreleasepool {
        MetalContext* context = (__bridge MetalContext*)(void*)handle;
        
        jfloat* aData = env->GetFloatArrayElements(a, NULL);
        jfloat* bData = env->GetFloatArrayElements(b, NULL);
        jfloat* cData = env->GetFloatArrayElements(c, NULL);
        
        id<MTLBuffer> bufferA = [context.device newBufferWithBytes:aData 
                                                            length:size * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [context.device newBufferWithBytes:bData 
                                                            length:size * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [context.device newBufferWithLength:size * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> commandBuffer = [context.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        id<MTLComputePipelineState> pipeline = context.pipelines[@"subtract"];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, size);
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        memcpy(cData, [bufferC contents], size * sizeof(float));
        
        env->ReleaseFloatArrayElements(a, aData, JNI_ABORT);
        env->ReleaseFloatArrayElements(b, bData, JNI_ABORT);
        env->ReleaseFloatArrayElements(c, cData, 0);
    }
}

JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeHadamard(
    JNIEnv *env, jobject obj, jlong handle,
    jfloatArray a, jfloatArray b, jfloatArray c, jint size)
{
    @autoreleasepool {
        MetalContext* context = (__bridge MetalContext*)(void*)handle;
        
        jfloat* aData = env->GetFloatArrayElements(a, NULL);
        jfloat* bData = env->GetFloatArrayElements(b, NULL);
        jfloat* cData = env->GetFloatArrayElements(c, NULL);
        
        id<MTLBuffer> bufferA = [context.device newBufferWithBytes:aData 
                                                            length:size * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [context.device newBufferWithBytes:bData 
                                                            length:size * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [context.device newBufferWithLength:size * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> commandBuffer = [context.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        id<MTLComputePipelineState> pipeline = context.pipelines[@"hadamard"];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, size);
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        memcpy(cData, [bufferC contents], size * sizeof(float));
        
        env->ReleaseFloatArrayElements(a, aData, JNI_ABORT);
        env->ReleaseFloatArrayElements(b, bData, JNI_ABORT);
        env->ReleaseFloatArrayElements(c, cData, 0);
    }
}

JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeScalarMul(
    JNIEnv *env, jobject obj, jlong handle,
    jfloatArray a, jfloatArray c, jfloat scalar, jint size)
{
    @autoreleasepool {
        MetalContext* context = (__bridge MetalContext*)(void*)handle;
        
        jfloat* aData = env->GetFloatArrayElements(a, NULL);
        jfloat* cData = env->GetFloatArrayElements(c, NULL);
        
        id<MTLBuffer> bufferA = [context.device newBufferWithBytes:aData 
                                                            length:size * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [context.device newBufferWithLength:size * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> commandBuffer = [context.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        id<MTLComputePipelineState> pipeline = context.pipelines[@"scalar_mul"];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferC offset:0 atIndex:1];
        [encoder setBytes:&scalar length:sizeof(float) atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, size);
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        memcpy(cData, [bufferC contents], size * sizeof(float));
        
        env->ReleaseFloatArrayElements(a, aData, JNI_ABORT);
        env->ReleaseFloatArrayElements(c, cData, 0);
    }
}

JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeTranspose(
    JNIEnv *env, jobject obj, jlong handle,
    jfloatArray a, jfloatArray b, jint rows, jint cols)
{
    @autoreleasepool {
        MetalContext* context = (__bridge MetalContext*)(void*)handle;
        
        jfloat* aData = env->GetFloatArrayElements(a, NULL);
        jfloat* bData = env->GetFloatArrayElements(b, NULL);
        
        id<MTLBuffer> bufferA = [context.device newBufferWithBytes:aData 
                                                            length:rows * cols * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [context.device newBufferWithLength:rows * cols * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> commandBuffer = [context.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        id<MTLComputePipelineState> pipeline = context.pipelines[@"transpose"];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBytes:&rows length:sizeof(uint) atIndex:2];
        [encoder setBytes:&cols length:sizeof(uint) atIndex:3];
        
        MTLSize gridSize = MTLSizeMake(cols, rows, 1);
        NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger w = sqrt(threadGroupSize);
        MTLSize threadgroupSize = MTLSizeMake(w, w, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        memcpy(bData, [bufferB contents], rows * cols * sizeof(float));
        
        env->ReleaseFloatArrayElements(a, aData, JNI_ABORT);
        env->ReleaseFloatArrayElements(b, bData, 0);
    }
}
