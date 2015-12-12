import org.apache.spark._
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

/**
 * @author chengqj
 */
object Linear extends App {
  
val sparkConf = new SparkConf().setAppName("graph")
val sc = new SparkContext(sparkConf)

// Load training data in LIBSVM format.
val data = MLUtils.loadLibSVMFile(sc, "/data/chengqj/mllib/sample_libsvm_data.txt")

// Split data into training (60%) and test (40%).
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

// Run training algorithm to build the model
val numIterations = 100
val svmAlg = new SVMWithSGD()
//svmAlg.optimizer.s
val model = SVMWithSGD.train(training, numIterations)

// Clear the default threshold.
model.clearThreshold()

// Compute raw scores on the test set.
val scoreAndLabels = test.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}

// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()
metrics.areaUnderPR()
//metrics.
println("Area under ROC = " + auROC)





// Save and load model
//model.save(sc, "myModelPath")
//val sameModel = SVMModel.load(sc, "myModelPath")
}