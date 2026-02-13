package hs.ml.math.metal

import hs.ml.math.Tensor
import hs.ml.math.TensorFactory

class MetalBackend private constructor() {
    private var nativeHandle: Long = 0
    private var isInitialized = false

    companion object {
        @Volatile
        private var instance: MetalBackend? = null
        private var libraryLoaded = false

        init {
            try {
                System.loadLibrary("metalbridge")
                libraryLoaded = true
            } catch (e: UnsatisfiedLinkError) {
                System.err.println("Metal backend not available: ${e.message}")
                libraryLoaded = false
            }
        }

        fun getInstance(): MetalBackend? {
            if (!libraryLoaded) return null

            return instance ?: synchronized(this) {
                instance ?: MetalBackend().also { backend ->
                    backend.nativeHandle = backend.nativeInit()
                    if (backend.nativeHandle != 0L) {
                        backend.isInitialized = true
                        instance = backend
                    } else {
                        null
                    }
                }
            }
        }
    }

    private external fun nativeInit(): Long
    private external fun nativeRelease(handle: Long)
    private external fun nativeMatMul(
        handle: Long,
        a: FloatArray, b: FloatArray, c: FloatArray,
        m: Int, k: Int, n: Int
    )
    private external fun nativeAdd(
        handle: Long,
        a: FloatArray, b: FloatArray, c: FloatArray,
        size: Int
    )
    private external fun nativeSubtract(
        handle: Long,
        a: FloatArray, b: FloatArray, c: FloatArray,
        size: Int
    )
    private external fun nativeHadamard(
        handle: Long,
        a: FloatArray, b: FloatArray, c: FloatArray,
        size: Int
    )
    private external fun nativeScalarMul(
        handle: Long,
        a: FloatArray, c: FloatArray,
        scalar: Float, size: Int
    )
    private external fun nativeTranspose(
        handle: Long,
        a: FloatArray, b: FloatArray,
        rows: Int, cols: Int
    )

    private fun tensorToFloatArray(tensor: Tensor): FloatArray {
        val array = FloatArray(tensor.row * tensor.col)
        var idx = 0
        for (i in 0 until tensor.row) {
            for (j in 0 until tensor.col) {
                array[idx++] = tensor[i, j].toFloat()
            }
        }
        return array
    }

    private fun floatArrayToTensor(array: FloatArray, rows: Int, cols: Int): Tensor {
        return TensorFactory.create(rows, cols).apply {
            var idx = 0
            for (i in 0 until rows) {
                for (j in 0 until cols) {
                    this[i, j] = array[idx++].toDouble()
                }
            }
        }
    }

    fun matMul(a: Tensor, b: Tensor): Tensor {
        if (!isInitialized) throw IllegalStateException("Metal backend not initialized")
        if (a.col != b.row) {
            throw IllegalArgumentException("Matrix dimensions don't match for multiplication")
        }

        val aArray = tensorToFloatArray(a)
        val bArray = tensorToFloatArray(b)
        val cArray = FloatArray(a.row * b.col)

        nativeMatMul(nativeHandle, aArray, bArray, cArray, a.row, a.col, b.col)

        return floatArrayToTensor(cArray, a.row, b.col)
    }

    fun add(a: Tensor, b: Tensor): Tensor {
        if (!isInitialized) throw IllegalStateException("Metal backend not initialized")
        if (a.row != b.row || a.col != b.col) {
            throw IllegalArgumentException("Tensor dimensions must match for addition")
        }

        val size = a.row * a.col
        val aArray = tensorToFloatArray(a)
        val bArray = tensorToFloatArray(b)
        val cArray = FloatArray(size)

        nativeAdd(nativeHandle, aArray, bArray, cArray, size)

        return floatArrayToTensor(cArray, a.row, a.col)
    }

    fun subtract(a: Tensor, b: Tensor): Tensor {
        if (!isInitialized) throw IllegalStateException("Metal backend not initialized")
        if (a.row != b.row || a.col != b.col) {
            throw IllegalArgumentException("Tensor dimensions must match for subtraction")
        }

        val size = a.row * a.col
        val aArray = tensorToFloatArray(a)
        val bArray = tensorToFloatArray(b)
        val cArray = FloatArray(size)

        nativeSubtract(nativeHandle, aArray, bArray, cArray, size)

        return floatArrayToTensor(cArray, a.row, a.col)
    }

    fun hadamard(a: Tensor, b: Tensor): Tensor {
        if (!isInitialized) throw IllegalStateException("Metal backend not initialized")
        if (a.row != b.row || a.col != b.col) {
            throw IllegalArgumentException("Tensor dimensions must match for Hadamard product")
        }

        val size = a.row * a.col
        val aArray = tensorToFloatArray(a)
        val bArray = tensorToFloatArray(b)
        val cArray = FloatArray(size)

        nativeHadamard(nativeHandle, aArray, bArray, cArray, size)

        return floatArrayToTensor(cArray, a.row, a.col)
    }

    fun scalarMul(a: Tensor, scalar: Double): Tensor {
        if (!isInitialized) throw IllegalStateException("Metal backend not initialized")

        val size = a.row * a.col
        val aArray = tensorToFloatArray(a)
        val cArray = FloatArray(size)

        nativeScalarMul(nativeHandle, aArray, cArray, scalar.toFloat(), size)

        return floatArrayToTensor(cArray, a.row, a.col)
    }

    fun transpose(a: Tensor): Tensor {
        if (!isInitialized) throw IllegalStateException("Metal backend not initialized")

        val aArray = tensorToFloatArray(a)
        val bArray = FloatArray(a.row * a.col)

        nativeTranspose(nativeHandle, aArray, bArray, a.row, a.col)

        return floatArrayToTensor(bArray, a.col, a.row)
    }

    fun release() {
        if (isInitialized) {
            nativeRelease(nativeHandle)
            isInitialized = false
        }
    }
}