package hs.ml

import hs.ml.util.formatBytes
import java.io.File

fun main() {
    println("OOP Machine Learning Project")
    println("PWD: ${File(".").canonicalFile}")
    println("CPU : ${Runtime.getRuntime().availableProcessors()} cores")
    println("Mem : ${formatBytes(Runtime.getRuntime().maxMemory())}")
    println()
}
