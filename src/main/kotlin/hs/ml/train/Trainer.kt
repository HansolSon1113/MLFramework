package hs.ml.train

import hs.ml.math.Tensor
import hs.ml.model.Model
import hs.ml.scaler.StandardScaler
import hs.ml.train.optimizer.Optimizer
import hs.ml.train.optimizer.SGD

class Trainer {
    val model: Model
    val grad: (Double) -> Double
    val optim: Optimizer

    constructor(model: Model, grad: (Double) -> Double, optim: Optimizer = SGD()) {
        this.model = model
        this.grad = grad
        this.optim = optim
    }

    companion object {
        inline fun <reified T : Model> create(noinline grad: (Double) -> Double): Trainer {
            val model = ModelFactory.create<T>().getModel()
            return Trainer(model, grad)
        }
    }

    fun descent(x: Tensor, y: Tensor, lr: Double) {
//        val yhat = model.predict(x)
//        val errors = Tensor(y.row, 1) { i, j -> grad(yhat[i, j] - y[i, j]) }
//        // 가중치와 편향 업데이트
//        for (j in 0 until x.col) {
//            var gradient = 0.0
//            for (i in 0 until x.row) {
//                gradient += errors[i, 0] * x[i, j] / x.row
//            }
//            model.weights[j, 0] -= lr * gradient
//        }
//
//        var biasGradient = 0.0
//        for (i in 0 until x.row) {
//            biasGradient += errors[i, 0] / x.row
//        }
//        model.bias -= lr * biasGradient
    }

    fun fit(xOrg: Tensor, y: Tensor, epochs: Int, lr: Double) {
//        model.scaler = StandardScaler()
//        model.scaler.fit(xOrg)
//        val x = model.scaler.predict(xOrg)
//
//        model.weights = Tensor(x.col, 1) { i, j -> 0.0 }
//        model.bias = 0.0
//
//        for (epoch in 0 until epochs) {
//            descent(x, y, lr)
//        }
//
//        model.epoch += epochs
    }
}