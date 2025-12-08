package hs.ml.ui.view

import hs.ml.math.Tensor
import hs.ml.metric.F1
import hs.ml.metric.RootMeanSquaredError
import hs.ml.ui.view.MLView
import java.util.Scanner

class ConsoleView : MLView {
    private val scanner = Scanner(System.`in`)

    override fun showMessage(message: String) {
        clear()
        println(message)
    }
    override fun showError(message: String) = println("[ERROR] $message")

    override fun getInput(prompt: String): String {
        print("$prompt: ")
        return scanner.nextLine()
    }

    override fun showSingleSelectMenu(title: String, options: List<String>): Int {
        println("[$title]")

        for ((index, option) in options.withIndex()) {
            println("${index + 1}. $option")
        }

        var choice: Int
        do {
            print("> ")
            val input = scanner.nextLine()
            choice = input.toIntOrNull() ?: -1
            if (choice !in 1..options.size) {
                println("유효한 선택지가 아닙니다. 다시 시도해주세요.")
            } else {
                break
            }
        } while (true)
        return choice
    }

    override fun showMultiSelectMenu(title: String, options: List<String>): List<Int> {
        val selectedIndices = mutableSetOf<Int>()
        println("[$title]")

        while (true) {
            println("현재 선택됨: ${selectedIndices.joinToString(", ") { options[it] }}")
            options.forEachIndexed { index, option ->
                val marker = if (index in selectedIndices) "[v]" else "[ ]"
                println("$marker ${index + 1}. $option")
            }
            println("${options.size + 1}. 완료")

            print("> ")
            val choice = scanner.nextLine().toIntOrNull()

            when {
                choice == options.size + 1 -> break
                choice != null && choice in 1..options.size -> {
                    val idx = choice - 1
                    if (idx in selectedIndices) selectedIndices.remove(idx)
                    else selectedIndices.add(idx)
                }

                else -> println("유효한 선택지가 아닙니다. 다시 시도해주세요.")
            }
        }
        return selectedIndices.toList()
    }

    override fun showDataPreview(inputs: Tensor, limit: Int) {
        val rows = minOf(limit, inputs.row)
        for (i in 0..<rows) {
            val rowStr = (0..<inputs.col).joinToString(", ") { j ->
                String.format("%.4f", inputs[i, j])
            }
            println("[$i] $rowStr")
        }
    }

    override fun showTrainingLog(epoch: Int, log: String) {
        println("Epoch $epoch | $log")
    }

    override fun showEvaluationResult(index: Int, actual: Double, pred: Double) {
        val diff = pred - actual
        println(
            "[$index] 실제: ${String.format("%.4f", actual)} | 예측: ${
                String.format(
                    "%.4f", pred
                )
            } | 오차: ${String.format("%.4f", diff)}"
        )
    }

    fun clear() {
        repeat(50) { println() }
    }
}