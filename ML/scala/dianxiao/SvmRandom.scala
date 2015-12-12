
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
 * @author Administrator
 */
object SvmRandom {
  
  val sparkConf = new SparkConf().setAppName("MLIB")
  val sc = new SparkContext(sparkConf)
    
   /*
  var  data :RDD[LabeledPoint]  = sc.textFile("/data/mllib/test_Titanic.txt",50).map { line =>
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
*/
// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "/data/mllib/test_Titanic.txt")
// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

// Train a RandomForest model.
//  Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 50// Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "gini"
val maxDepth = 4
val maxBins = 32

val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
println("Test Error = " + testErr)
println("Learned classification forest model:\n" + model.toDebugString)
}