package hs.ml.math.metal

object MetalConfig {
    var enabled = true
    var minSizeForGPU = 100

    private val metalBackend: MetalBackend? by lazy {
        if (enabled) MetalBackend.getInstance() else null
    }

    fun isAvailable(): Boolean = enabled && metalBackend != null

    fun getBackend(): MetalBackend? = metalBackend
}

