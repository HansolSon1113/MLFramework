package hs.ml.ui.view

import hs.ml.math.Tensor

interface MLView {
    fun showMessage(message: String)
    fun showError(message: String)
    fun getInput(prompt: String): String
    fun showSingleSelectMenu(title: String, options: List<String>): Int
    fun showMultiSelectMenu(title: String, options: List<String>): List<Int>
    fun showDataPreview(inputs: Tensor, limit: Int = 5)
    fun showTrainingLog(epoch: Int, log: String)
    fun showEvaluationResult(index: Int, actual: Double, pred: Double)
}