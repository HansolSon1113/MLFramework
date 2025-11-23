package hs.ml

import hs.ml.importer.CsvImporter
import hs.ml.importer.DataImporter
import hs.ml.importer.LinearDataGenerator
import hs.ml.model.Evaluator
import hs.ml.model.LinearRegressor
import hs.ml.train.Derivatives.Companion.mse
import hs.ml.train.Trainer
import hs.ml.util.formatBytes
import java.io.File
import java.util.Scanner

fun main() {
    println("\n\n================================")
    println("OOP Machine Learning Project")
    println("PWD : ${File(".").canonicalFile}")
    println("CPU : ${Runtime.getRuntime().availableProcessors()} cores")
    println("Mem : ${formatBytes(Runtime.getRuntime().maxMemory())}")
    println()

    val scanner = Scanner(System.`in`)
    var input = -1

    println("**데이터 입력 단계**")
    println("1. CSV파일 불러오기")
    println("2. 랜덤 선형 데이터 생성하기")

    do {
        print(">> ")
        input = scanner.nextInt()
    } while (input != 1 && input != 2)

    var importer: DataImporter
    do {
        if (input == 1) {
            print("CSV 파일 경로 : ")
            scanner.nextLine() // 버퍼 비우기
            val path = scanner.nextLine()
            importer = CsvImporter(path)
        } else {
            print("데이터 개수 : ")
            val n = scanner.nextInt()
            print("기울기 : ")
            val slope = scanner.nextDouble()
            print("절편 : ")
            val bias = scanner.nextDouble()
            print("노이즈 : ")
            val noise = scanner.nextDouble()
            importer = LinearDataGenerator(n, slope, bias, noise)
        }
    } while (!importer.available())

    println("데이터 불러오는 중...")
    val (x, y) = importer.read()
    println("데이터 불러오기 완료!")
    println("x: ${x.shape}, y: ${y.shape}")

    println("\n================================\n\n")

    println("**모델 선택 단계**")
    // TODO : Model Selection

    val model = LinearRegressor()
    val trainer = Trainer(model, mse)
    println("모델 학습중...")
    trainer.fit(x, y, epochs = 1000, lr = 0.001)
    println("모델 학습 완료!")
    println(model)

    val rmse = model.evaluate(x, y, Evaluator::rmse)
    val r2 = model.evaluate(x, y, Evaluator::r2)
    println("모델 평가 점수 (RMSE) : $rmse")
    println("모델 평가 점수 (R2) : $r2")
    println("")
}
