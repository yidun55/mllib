import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

/**
 * @author chengqj
 */
object LogisticRegression extends App {
  
val sparkConf = new SparkConf().setAppName("graph")
val sc = new SparkContext(sparkConf)
  // Load training data in LIBSVM format.
//val data = MLUtils.loadLibSVMFile(sc, "file:///root/data/mllib/sample_libsvm_data.txt")
  
  var  data :RDD[LabeledPoint]  = sc.textFile("/data/mllib/pred_source_all_com",50).map { line =>
       val fields = line.split("\001")
       var vfields =new  Array[Double](fields.length-1) 
       
       for(i<- 0 to fields.length-2){
         try{
         vfields(i)=fields(i+1).toDouble
         }catch{                
              case  e:Exception => vfields(i)=0L
         }          
       }  
       LabeledPoint(fields(0).toDouble, Vectors.dense( vfields))
    }
 data.count()
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

// Run training algorithm to build the model
val model = new LogisticRegressionWithLBFGS().run(training)

// Compute raw scores on the test set.
val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

// Get evaluation metrics.
val metrics = new MulticlassMetrics(predictionAndLabels)
val precision = metrics.precision
println("Precision = " + precision)

// Save and load model
//model.save(sc, "myModelPath")
//val sameModel = LogisticRegressionModel.load(sc, "myModelPath")
}