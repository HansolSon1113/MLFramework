#ifndef MetalBridge_h
#define MetalBridge_h

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Metal device and library
JNIEXPORT jlong JNICALL Java_hs_ml_math_MetalBackend_nativeInit(JNIEnv *env, jobject obj);

// Release Metal resources
JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeRelease(JNIEnv *env, jobject obj, jlong handle);

// Matrix multiplication
JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeMatMul(
    JNIEnv *env, jobject obj, jlong handle,
    jfloatArray a, jfloatArray b, jfloatArray c,
    jint m, jint k, jint n);

// Element-wise operations
JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeAdd(
    JNIEnv *env, jobject obj, jlong handle,
    jfloatArray a, jfloatArray b, jfloatArray c, jint size);

JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeSubtract(
    JNIEnv *env, jobject obj, jlong handle,
    jfloatArray a, jfloatArray b, jfloatArray c, jint size);

JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeHadamard(
    JNIEnv *env, jobject obj, jlong handle,
    jfloatArray a, jfloatArray b, jfloatArray c, jint size);

JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeScalarMul(
    JNIEnv *env, jobject obj, jlong handle,
    jfloatArray a, jfloatArray c, jfloat scalar, jint size);

JNIEXPORT void JNICALL Java_hs_ml_math_MetalBackend_nativeTranspose(
    JNIEnv *env, jobject obj, jlong handle,
    jfloatArray a, jfloatArray b, jint rows, jint cols);

#ifdef __cplusplus
}
#endif

#endif /* MetalBridge_h */
