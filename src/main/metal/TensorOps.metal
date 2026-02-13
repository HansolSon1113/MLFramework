#include <metal_stdlib>
using namespace metal;

// Matrix multiplication: C = A * B
kernel void matmul(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],  // rows of A
    constant uint& K [[buffer(4)]],  // cols of A / rows of B
    constant uint& N [[buffer(5)]],  // cols of B
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Element-wise addition: C = A + B
kernel void add(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    C[id] = A[id] + B[id];
}

// Element-wise subtraction: C = A - B
kernel void subtract(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    C[id] = A[id] - B[id];
}

// Element-wise multiplication (Hadamard product): C = A * B
kernel void hadamard(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    C[id] = A[id] * B[id];
}

// Scalar multiplication: C = A * scalar
kernel void scalar_mul(
    const device float* A [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    C[id] = A[id] * scalar;
}

// Transpose: B = A^T
kernel void transpose(
    const device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= rows || col >= cols) return;
    
    B[col * rows + row] = A[row * cols + col];
}

// ReLU activation
kernel void relu(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = max(0.0f, input[id]);
}

// ReLU gradient
kernel void relu_grad(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = input[id] > 0.0f ? 1.0f : 0.0f;
}
