#!/bin/bash

# Build script for Metal backend native library
# This script compiles Metal shaders and Objective-C JNI bridge

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
METAL_DIR="$PROJECT_ROOT/src/main/metal"
OBJC_DIR="$PROJECT_ROOT/src/main/objc"
BUILD_DIR="$PROJECT_ROOT/build/native"
LIB_DIR="$PROJECT_ROOT/src/main/resources/native"

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$LIB_DIR"

echo "Building Metal backend..."

# Find Java home for JNI headers
if [ -z "$JAVA_HOME" ]; then
    echo "Error: JAVA_HOME is not set"
    exit 1
fi

JNI_INCLUDE="$JAVA_HOME/include"
JNI_INCLUDE_DARWIN="$JAVA_HOME/include/darwin"

# Check if JNI headers exist
if [ ! -d "$JNI_INCLUDE" ]; then
    echo "Error: JNI headers not found at $JNI_INCLUDE"
    exit 1
fi

# Compile Metal shaders to default library
echo "Compiling Metal shaders..."
xcrun -sdk macosx metal -c "$METAL_DIR/TensorOps.metal" -o "$BUILD_DIR/TensorOps.air"
xcrun -sdk macosx metallib "$BUILD_DIR/TensorOps.air" -o "$BUILD_DIR/default.metallib"
cp "$BUILD_DIR/default.metallib" "$LIB_DIR/"

# Compile Objective-C JNI bridge
echo "Compiling JNI bridge..."
clang++ -c "$OBJC_DIR/MetalBridge.mm" \
    -o "$BUILD_DIR/MetalBridge.o" \
    -I"$JNI_INCLUDE" \
    -I"$JNI_INCLUDE_DARWIN" \
    -framework Foundation \
    -framework Metal \
    -std=c++11 \
    -fPIC

# Link dynamic library
echo "Linking dynamic library..."
clang++ -dynamiclib \
    "$BUILD_DIR/MetalBridge.o" \
    -o "$LIB_DIR/libmetalbridge.dylib" \
    -framework Foundation \
    -framework Metal \
    -framework CoreGraphics

echo "Build complete!"
echo "Library location: $LIB_DIR/libmetalbridge.dylib"
echo "Metal library location: $LIB_DIR/default.metallib"

# Set library path for running
echo ""
echo "To use the Metal backend, add this to your VM options:"
echo "-Djava.library.path=$LIB_DIR"
