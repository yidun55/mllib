import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import  org.apache.spark.mllib.linalg.Vectors
import  org.apache.spark.mllib.linalg.Vector

/**
 * @author chengqj
 */
object newSvm extends App {
  
val sparkConf = new SparkConf().setAppName("graph")
val sc = new SparkContext(sparkConf)


  
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

/*
 var  data1 :RDD[LabeledPoint]  = sc.textFile("/data/mllib/test_source_all_clean",50).map { line =>
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
 data1.count()
 * */

// Load training data in LIBSVM format.
//val data = MLUtils.loadLibSVMFile(sc, "file:///root/data/mllib/sample_libsvm_data.txt")

// Split data into training (60%) and test (40%).
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)

val training = splits(0)
val test = splits(1)

// Run training algorithm to build the model
val numIterations = 8
val svmAlg = new SVMWithSGD()
//svmAlg.optimizer.s
val model = SVMWithSGD.train(training, numIterations)

// Clear the default threshold.
//model.clearThreshold()

// Compute raw scores on the test set.
val scoreAndLabels = test.map { point =>
  val score = model.predict(point.features)
 
  (score, point.label)
}

// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()
metrics.areaUnderPR()

val testErr = scoreAndLabels.filter(r => r._1 != r._2).count.toDouble / test.count()
println("Test Error = " + testErr)
//metrics.
println("Area under ROC = " + auROC)
var precision = scoreAndLabels.filter(r=>r._1==r._2&&r._2==1).count.toDouble/scoreAndLabels.filter(r=>r._2==1).count
println("TP/(TP+FP)"+precision+"################################################")


 var recall = scoreAndLabels.filter(r=>r._1==r._2&&r._2==1).count.toDouble/(scoreAndLabels.filter(r=>r._1==r._2&&r._2==1).count.toDouble+scoreAndLabels.filter(r=>r._1==0&&r._2==1).count.toDouble)
    println("TP/(TP+FN)"+recall+"################################################")
    



// Save and load model
//model.save(sc, "myModelPath")
//val sameModel = SVMModel.load(sc, "myModelPath")
}