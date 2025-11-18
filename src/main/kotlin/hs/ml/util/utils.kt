package hs.ml.util

import kotlin.math.roundToInt

fun formatBytes(bytes: Long): String {
    val units = listOf("B", "KB", "MB", "GB", "TB")
    var idx = 0
    var b = bytes.toDouble();
    while (b >= 1024) {
        b /= 1024
        idx += 1
    }

    return "${(b * 100).roundToInt().toDouble() / 100} ${units[idx]}"
}
