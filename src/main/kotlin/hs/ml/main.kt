package hs.ml

import hs.ml.autograd.Node
import hs.ml.model.Model
import hs.ml.model.nn.Dense
import hs.ml.model.nn.activation.ReLU
import hs.ml.ui.MLController
import hs.ml.ui.MLModel
import hs.ml.ui.view.ConsoleView

//시연용 최적값(split 0.8, lr 0.0005, epoch 3000)
class HousingNeuralNet(inputSize: Int) : Model() {
    private val fc = arrayOf(
        Dense(inputSize, 32),
        Dense(32, 16),
    )
    private val linear = Dense(16, 1)
    private val relu = ReLU()

    override fun forward(x: Node): Node {
        var output = x
        for (layer in fc) {
            output = layer.forward(output)
            output = relu.forward(output)
        }
        output = linear.forward(output)

        return output
    }

    override fun params(): List<Node> {
        return fc.flatMap { it.params() } + linear.params()
    }

    override fun toString(): String {
        return "HousingNeuralNet(epoch=$epoch, trained=$isTrained)"
    }
}

fun main() {
    val view = ConsoleView()
    val model = MLModel()
    val controller = MLController(view, model)
    controller.start()
}
