
import org.apache.spark._
import org.apache.spark.rdd.RDD
import  scala.collection.mutable.Set
import  org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Vectors, SparseVector, Vector}
import Array._
import scala.collection.mutable.ArrayBuffer
import scala.math


//去除所有的离散变量
 var  data:RDD[LabeledPoint]  = sc.textFile("/data/chengqj/myxiaodai.csv",10).mapPartitions{
    iter  => 
    var res = List[LabeledPoint]() 
    while (iter.hasNext){
       var  line =iter.next()
       line=line.trim()
       println(line)
       var fields = line.split(",",-1)
       var length = fields.length;
       var vfields =new  ArrayBuffer[Double]() 
       //vfields += fields(0).substring(0, 3).toDouble
       var rm_col = range(19,24).toBuffer
       rm_col += (0,1,2,4,25,26)  //去除所有离散变量
       rm_col ++= range(28,36) 
       //rm_col += (0,1,4,25,26)  //去离散变量，但保留伯努利变量
       // rm_col ++= range(6, 17)  //去除所有分布不均衡的变量
       // rm_col += (24,26,36,37,38,44,46,48,51,52,54,56) //去除所有分布不均衡的变量

       /*var rm_col = range(1, 12).toBuffer  //只保留(12,13,15,16,25,27)这几个变量
       rm_col += (14,26)
       rm_col ++= range(17,25)
       rm_col ++= range(28, 57)*/

       for(i<- 0 to length-1){
         try{
           if(rm_col.exists(el=>el==i))
               println("")
           else
             vfields+=fields(i).toDouble
         
         }catch{                
              case  e:Exception => vfields+=0L
         }          
       }  
       res .::=(LabeledPoint(fields(1).toDouble, Vectors.dense( vfields.toArray)) )
    }
    res.iterator

}

val resample_time = data.filter(p=>p.label==0).count.toDouble/data.filter(p=>p.label==1).count
data = data.filter(p=>p.label==0).union(data.filter(p=>p.label==1).sample(true, resample_time))


val normal_data = data


var splits = normal_data.randomSplit(Array(0.7, 0.3), seed=1)
var (trainingData, testData) = (splits(0), splits(1))
trainingData.cache 

val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
boostingStrategy.treeStrategy.numClasses = 2
boostingStrategy.treeStrategy.maxDepth = 5
//  Empty categoricalFeaturesInfo indicates all features are continuous.
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

var labelAndPreds = testData.map { point =>
val prediction = model.predict(point.features)
  (prediction,point.label)
}
var testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count() 
var precision = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/(labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble+labelAndPreds.filter(r=>r._1==1&&r._2==0).count.toDouble)

var recall = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/(labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble+labelAndPreds.filter(r=>r._1==0&&r._2==1).count.toDouble)

var p_n = labelAndPreds.filter(r=>r._2==1).count.toDouble/labelAndPreds.filter(r=>r._2==0).count
println("Test Error = " + testErr)
println("TP/(TP+FP)"+precision+"################################################")
println("TP/(TP+FN)"+recall+"################################################")

