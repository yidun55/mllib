
import org.apache.spark._
import org.apache.spark.rdd.RDD
import  scala.collection.mutable.Set
import  org.apache.spark.mllib.tree.RandomForest
import  org.apache.spark.mllib.regression.LabeledPoint
import  org.apache.spark.mllib.linalg.Vectors
import  org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
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
import Array._

/**
 * @author Administrator
 */
object XiaodaiSvm {
  
 def main(args: Array[String]) = {
     val sparkConf = new SparkConf().setAppName("xiaodai")
     val sc = new SparkContext(sparkConf)
        
     var  city = sc.textFile("/data/chengqj/city.csv",1).collect()
     var  hash=Map[String,Double]()
     city.foreach{x  =>  hash +=(x.split(",")(1)  ->  x.split(",")(0).toDouble)}
     var  data:RDD[LabeledPoint]  = sc.textFile("/data/chengqj/myxiaodai.csv",10).map{
        line =>
           //line=line.trim()
           //println(line)
           var fields = line.split(",",-1)
           var length = fields.length;
           var vfields = ArrayBuffer[Double]() 
           //vfields(0)=fields(0).toDouble
           //vfields(0)=fields(0).substring(0, 3).toDouble
           var rm_arr = Array(0, 4).toBuffer ++ range(19,24)
           for(i<- 2 until length){
             try{
               //if(rm_arr.exists(el => el==i)){
                if(i==19 ||i==20 || i==21 |i==22){
                 vfields(i-2) = 0L
                }
               else{
                 vfields(i-2)=fields(i).toDouble
               }
             
             }catch{                
                  case  e:Exception => vfields(i)=0L
             }          
           }  
           LabeledPoint(fields(1).toDouble, Vectors.dense(vfields))

    }
      data.count()
      data.foreach { println(_) }
      
      
      var splits = data.randomSplit(Array(0.7, 0.3))
      var (trainingData, testData) = (splits(0), splits(1))
      
      
       //var trainer = "kernel"
       var kernelName = "gaussian"   //polynomial, gaussian
       var miniBatch = 0.01
       var stepSize = 0.1
       var regParam = 0.01
       var numIterations = 200  
      
       
        val biased = false
       // val kernelName = "gaussian"
        //val kernelName = "polynomial"
       
        val m = KernelSVMWithPegasos.train(trainingData, numIterations, regParam, biased, kernelName)
        //val model = SVMWithSVM.train(training, numIterations)
      
        // Clear the default threshold.
        m.clearThreshold()
        var model = m
        var test = testData
      //  Compute raw scores on the test set.
        var labelAndPreds = testData.map { point =>
        val prediction = model.predict(point.features)
          (prediction,point.label)
        }
      
      
       //val metrics = new BinaryClassificationMetrics(labelAndPreds)
      // val auROC = metrics.areaUnderROC()
      
      
      // Evaluate model on test instances and compute test error
      
      var testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count() 
      var precision = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/(labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble+labelAndPreds.filter(r=>r._1==1&&r._2==0).count.toDouble)

      var recall = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/(labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble+labelAndPreds.filter(r=>r._1==0&&r._2==1).count.toDouble)
      var metrics = new BinaryClassificationMetrics(labelAndPreds)
      var auROC = metrics.areaUnderROC()
      println("AUC =" + auROC)
      println("Test Error = " + testErr)
      println("TP/(TP+FP)"+precision+"################################################")
      println("TP/(TP+FN)"+recall+"################################################")
   }

}