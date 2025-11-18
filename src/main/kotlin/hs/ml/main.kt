package hs.ml

import hs.ml.data.Tensor
import hs.ml.util.formatBytes
import java.io.File

fun main() {
    println("OOP Machine Learning Project")
    println("PWD : ${File(".").canonicalFile}")
    println("CPU : ${Runtime.getRuntime().availableProcessors()} cores")
    println("Mem : ${formatBytes(Runtime.getRuntime().maxMemory())}")
    println()

    val t1 = Tensor(2, 2, 1.0)
    val t2 = Tensor(2, 3, 2.0)
    println(t1 * t2)
}
