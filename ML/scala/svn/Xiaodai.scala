import org.apache.spark._
import org.apache.spark.rdd.RDD
import  scala.collection.mutable.Set
import  org.apache.spark.mllib.tree.RandomForest
import  org.apache.spark.mllib.regression.LabeledPoint
import  org.apache.spark.mllib.linalg.Vectors
import  org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
/**
 * @author CHENGQJ
 */
object Xiaodai extends App {
    val sparkConf = new SparkConf().setAppName("MLIB")
    val sc = new SparkContext(sparkConf)
    
   var  data :RDD[LabeledPoint]  = sc.textFile("/data/mllib/pred_source_all_com",50).map { line =>
  //var  data :RDD[LabeledPoint]  = sc.textFile("/data/mllib/test_source_all_clean",50).map { line =>
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
 var  data1 :RDD[LabeledPoint]  = sc.textFile("/data/mllib/skb_test_0901",50).map { line =>
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
*/
// Load and parse the data file.
//val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
// Split the data into training and test sets (30% held out for testing)
var splits = data.randomSplit(Array(0.7, 0.3))
var (trainingData, testData) = (splits(0), splits(1))
//var (trainingData, testData) = (data, data1)

// Train a RandomForest model.
//  Empty categoricalFeaturesInfo indicates all features are continuous.
var numClasses = 2
var categoricalFeaturesInfo = Map[Int, Int]()
var numTrees = 100// Use more in practice.
var featureSubsetStrategy = "auto" // Let the algorithm choose.
var impurity = "gini"
var maxDepth = 6
var maxBins = 32

var model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

// Evaluate model on test instances and compute test error
var labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
var testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
println("Test Error = " + testErr)
 var precision = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/labelAndPreds.filter(r=>r._2==1).count
    println("TP/(TP+FP)"+precision+"################################################")

 var recall = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/(labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble+labelAndPreds.filter(r=>r._1==0&&r._2==1).count.toDouble)
    println("TP/(TP+FN)"+recall+"################################################")


println("Learned classification forest model:\n" + model.toDebugString)
}