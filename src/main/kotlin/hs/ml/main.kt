package hs.ml

import hs.ml.autograd.Node
import hs.ml.data.DataPipeline
import hs.ml.importer.CsvImporter
import hs.ml.loss.MeanSquaredError
import hs.ml.metric.F1
import hs.ml.metric.RootMeanSquaredError
import hs.ml.model.Model
import hs.ml.model.nn.Dense
import hs.ml.model.nn.activation.ReLU
import hs.ml.preprocessing.DataPreprocessor
import hs.ml.preprocessing.policy.ReplaceToAvgPolicy
import hs.ml.scaler.StandardScaler
import hs.ml.train.Trainer
import hs.ml.train.optimizer.Adam
import hs.ml.util.formatBytes
import java.io.File

class HousingNeuralNet(inputSize: Int) : Model() {
    private val fc1 = Dense(inputSize, 32)
    private val relu = ReLU()
    private val fc2 = Dense(32, 1)

    override fun forward(x: Node): Node {
        val z1 = fc1.forward(x)
        val a1 = relu.forward(z1)
        val z2 = fc2.forward(a1)
        return z2
    }
    override fun params(): List<Node> {
        return fc1.params() + fc2.params()
    }

    override fun toString(): String {
        return "HousingNeuralNet(epoch=$epoch, trained=$isTrained)"
    }
}

fun main() {
    println("\n\n================================")
    println("OOP Machine Learning Project (Neural Net Test)")
    println("PWD : ${File(".").canonicalFile}")
    println("CPU : ${Runtime.getRuntime().availableProcessors()} cores")
    println("Mem : ${formatBytes(Runtime.getRuntime().maxMemory())}")
    println("================================\n\n")

    println("**데이터 불러오기 및 전처리 단계**\n")
    val importer = CsvImporter("data/housing.csv")
    val pipeline = DataPipeline(
        importer = importer,
        preprocessor = DataPreprocessor(
            missingPolicy = ReplaceToAvgPolicy(),
            scaler = StandardScaler()
        )
    )
    val batch = pipeline.run()

    println("데이터 shape -> x: ${batch.inputs.shape}, y: ${batch.labels.shape}")
    val inputFeatureSize = batch.inputs.col

    println("\n**모델 생성 단계 (Custom Neural Network)**\n")
    val model = HousingNeuralNet(inputFeatureSize)
    model.param.loss = MeanSquaredError()
    model.param.optimizer = Adam(lr = 0.01)
    model.param.metric.add(RootMeanSquaredError())
    println("모델 생성 완료! : $model")

    println("\n**학습 시작**\n")
    val trainer = Trainer(model)
    trainer.train(batch, epochs = 1000, verbose = true)
    println("\n**학습 완료**\n")

    println("\n예측 결과 예시 (상위 5개 데이터):")

    val inputVal = Node(batch.inputs)
    val outputVal = model.forward(inputVal)

    for (i in 0 until 5) {
        val prediction = outputVal.data[i, 0]
        val actual = batch.labels[i, 0]
        val diff = prediction - actual

        println("[$i] 실제값: ${String.format("%.4f", actual)} | 예측값: ${String.format("%.4f", prediction)} | 오차: ${String.format("%.4f", diff)}")
    }

    println("\n**평가**\n")
    val selectedResult = trainer.evaluate(batch, RootMeanSquaredError(), F1())
    println("평가: $selectedResult")
}