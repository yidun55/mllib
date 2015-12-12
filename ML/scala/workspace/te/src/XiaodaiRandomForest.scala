
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
object XiaodaiRandomForest {
  val sparkConf = new SparkConf().setAppName("xiaodai")
  val sc = new SparkContext(sparkConf)
  
 var  city = sc.textFile("/data/chengqj/city.csv",1).collect()
 var  hash=Map[String,Double]()
 city.foreach{x  =>  hash +=(x.split(",")(1)  ->  x.split(",")(0).toDouble)}
 var  data:RDD[LabeledPoint]  = sc.textFile("/data/chengqj/xiaodai1.csv",10).mapPartitions{
    iter  => 
    var res = List[LabeledPoint]() 
    while (iter.hasNext){
       var  line =iter.next()
       println(line)
       val fields = line.split(",")
       var vfields =new  Array[Double](56) 
       vfields(0)=fields(0).toDouble
       for(i<- 2 to 55){
         try{
           if(i==19 ||i==20 || i==21 |i==22)
             if(hash(fields(i+1))==null)
               vfields(i)=0L
             else
               vfields(i)=hash(fields(i+1))
           else
             vfields(i)=fields(i+1).toDouble
         
         }catch{                
              case  e:Exception => vfields(i)=0L
         }          
       }  
       res .::=(LabeledPoint(fields(1).toDouble, Vectors.dense( vfields)) )
    }
    res.iterator

    }
 data.count()
 data.foreach { println(_) }


var splits = data.randomSplit(Array(0.7, 0.3))
var (trainingData, testData) = (splits(0), splits(1))
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
var precision = labelAndPreds.filter(r=>r._1==r._2&&r._2==0).count.toDouble/labelAndPreds.filter(r=>r._2==0).count
    println("TP/(TP+FP)"+precision+"################################################")

var recall = labelAndPreds.filter(r=>r._1==r._2&&r._2==0).count.toDouble/(labelAndPreds.filter(r=>r._1==r._2&&r._2==0).count.toDouble+labelAndPreds.filter(r=>r._1==0&&r._2==0).count.toDouble)
    println("TP/(TP+FN)"+recall+"################################################")


println("Learned classification forest model:\n" + model.toDebugString)
}