import java.io.{FileWriter, BufferedWriter, PrintWriter, File}

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark._
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors, SparseVector, Vector}
import org.apache.spark.storage.StorageLevel

object app{
    val sparkConf = new SparkConf().setAppName("pred_source_all")
    val sc = new SparkContext(sparkConf)
    def main(args: Array[String]) = {
       var filename = "/data/mllib/svm_test_dyh"
       var trainer = "kernel"
       var kernelName = "polynomial"   //polynomial, gaussian
       var miniBatch = 0.01
       var stepSize = 1.0
       var regParam = 0.01
       var output = "result.txt"
       var numIterations = 200  
       
       //load data
       val data = sc.textFile(filename).map{ line =>
       val tokens = line.split(" ")
       val label = tokens(0)
       val features = tokens.slice(1, tokens.size)
       LabeledPoint(label.toDouble, Vectors.dense(features.map(f => f.toDouble)))
       }
       
       var training = data
       
        val biased = false

        //val kernelName = "gaussian"
        //val kernelName = "polynomial"
        val m = KernelSVMWithPegasos.train(training, numIterations, regParam, biased, kernelName)
        //val model = SVMWithSVM.train(training, numIterations)

        // Clear the default threshold.
        m.clearThreshold()
        var model = m
        model
        
        var test = training.cache()
    //  Compute raw scores on the test set.
        val scoreAndLabels = test.map { point =>
              val score = model.predict(point.features)
              (score, point.label)
        }
       
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
       val auROC = metrics.areaUnderROC()

       println("Area under ROC = " + auROC+"**********************************************")
    }
}