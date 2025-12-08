package hs.ml.ui

import hs.ml.HousingNeuralNet
import hs.ml.autograd.Node
import hs.ml.data.DataPipeline
import hs.ml.importer.CsvImporter
import hs.ml.importer.DataImporter
import hs.ml.loss.BinaryCrossEntropy
import hs.ml.loss.MeanSquaredError
import hs.ml.metric.F1
import hs.ml.metric.RootMeanSquaredError
import hs.ml.model.Model
import hs.ml.model.classifier.LogisticRegressor
import hs.ml.model.regressor.LinearRegressor
import hs.ml.preprocessing.DataPreprocessor
import hs.ml.preprocessing.policy.MissingPolicy
import hs.ml.preprocessing.policy.ReplaceToAvgPolicy
import hs.ml.preprocessing.policy.ReplaceToZeroPolicy
import hs.ml.scaler.MinMaxScaler
import hs.ml.scaler.Scaler
import hs.ml.scaler.StandardScaler
import hs.ml.train.Trainer
import hs.ml.train.optimizer.Adam
import hs.ml.train.optimizer.SGD
import hs.ml.ui.view.MLView
import hs.ml.util.trainTestSplit
import java.io.File
import kotlin.collections.plus

class MLController(private val view: MLView, private val model: MLModel) {
    fun getImporter(): DataImporter {
        var filePath: String
        do {
            filePath = view.getInput("CSV 파일 경로")
            val file = File(filePath)
            if (!file.exists() || file.extension.lowercase() != "csv") {
                println("유효한 CSV 파일 경로가 아닙니다. 다시 시도해주세요.")
                continue
            } else {
                break
            }
        } while (true)

        return CsvImporter(filePath)
    }

    fun getScaler(): Scaler? {
        val s = view.showSingleSelectMenu("스케일러 선택", listOf("StandardScaler", "MinMaxScaler", "None"))
        return when (s) {
            1 -> StandardScaler()
            2 -> MinMaxScaler()
            else -> null
        }
    }

    fun getMissingPolicy(): MissingPolicy {
        val p = view.showSingleSelectMenu("결측치 처리 정책 선택", listOf("평균값으로 대체", "0으로 대체"))

        return when (p) {
            1 -> ReplaceToAvgPolicy()
            else -> ReplaceToZeroPolicy()
        }
    }

    fun getModel(inputSize: Int): Model {
        val m = view.showSingleSelectMenu("모델 선택", listOf("Linear Regression", "Logistic Regression", "Neural Network"))

        val model = when (m) {
            1 -> LinearRegressor()
            2 -> LogisticRegressor()
            3 -> HousingNeuralNet(inputSize)
            else -> throw IllegalStateException("유효하지 않은 모델 선택")
        }

        val l = view.showSingleSelectMenu("loss 함수 선택", listOf("Mean Squared Error", "Binary Cross Entropy"))
        model.param.loss = when (l) {
            1 -> MeanSquaredError()
            else -> BinaryCrossEntropy()
        }

        val o = view.showSingleSelectMenu("optimizer 선택", listOf("SGD", "Adam"))
        val lr = view.getInput("학습률").toDoubleOrNull() ?: 0.01
        model.param.optimizer = when (o) {
            1 -> SGD(lr)
            else -> Adam(lr)
        }

        val mt = view.showMultiSelectMenu("metric 선택", listOf("Root Mean Squared Error", "F1"))
        mt.forEach { idx ->
            when (idx) {
                0 -> model.param.metric.add(RootMeanSquaredError())
                1 -> model.param.metric.add(F1())
            }
        }

        return model
    }

    fun start() {
        view.showMessage("OOP Machine Learning Project")
        view.getInput("INPUT ANY KEY TO CONTINUE")

        view.showMessage("\n**데이터 전처리 단계**")
        val importer = getImporter()
        val scaler = getScaler()
        val policy = getMissingPolicy()

        val pipeline = DataPipeline(
            importer = importer,
            preprocessor = DataPreprocessor(policy, scaler)
        )

        try {
            model.data = pipeline.run()
            view.showMessage("데이터 전처리 완료. 데이터 크기: ${model.data!!.inputs.shape}")

            view.showMessage("\n전처리된 데이터 미리보기 (첫 5개 행):")
            val previewRows = minOf(5, model.data!!.inputs.row)
            for (i in 0 until previewRows) {
                val rowValues = (0 until model.data!!.inputs.col).joinToString(", ") { j ->
                    String.format("%.4f", model.data!!.inputs[i, j])
                }

                view.showMessage("[$i] $rowValues")
            }

            val (train, test) = trainTestSplit(model.data!!, 0.8)

            view.showMessage("\n**모델 생성 단계**")
            val model = getModel(train.inputs.col)
            view.showMessage("모델 생성 완료: $model")

            view.showMessage("\n**학습 시작**")
            val trainer = Trainer(model)
            val epochs = view.getInput("epoch").toIntOrNull() ?: 1000
            trainer.train(train, epochs = epochs, verbose = view::showTrainingLog)

            view.showMessage("\n**학습 완료**")
            view.showMessage("Model = $model")

            view.showMessage("\n**테스트 데이터 평가**")
            val testInputs = Node(test.inputs)
            val predictions = model.forward(testInputs)

            val limit = minOf(5, test.labels.row)
            for (i in 0 until limit) {
                val predVal = predictions.data[i, 0]
                val actualVal = test.labels[i, 0]
                view.showEvaluationResult(i, actualVal, predVal)
            }

        } catch (e: Exception) {
            view.showError("${e.message}")
            e.printStackTrace()
        }
    }
}