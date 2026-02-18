import hs.ml.autograd.Node
import hs.ml.math.TensorFactory
import hs.ml.model.nn.sequential.recurrent.Recurrent
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.sqrt

class RNN(inputSize: Int, hiddenSize: Int, activation: (Node) -> Node) : Recurrent(inputSize, hiddenSize, activation) {
    override fun forward(input: Node): Node {
        require(state == null || state!!.data.shape.first == input.data.shape.first) { "Batch size between state and input does not match" }

        if (state == null) state = Node(TensorFactory.create(input.data.shape.first, hiddenSize, 0.0))

        val value = activation(input * weights + state!! * hiddens + bias)
        state = value

        return value
    }

    override fun params(): List<Node> = listOf(weights, hiddens, bias)
}