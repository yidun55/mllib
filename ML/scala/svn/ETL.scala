import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
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
object ETL {
   val sparkConf = new SparkConf().setAppName("MLIB")
   val sc = new SparkContext(sparkConf)
    
   var  data:RDD[Tuple2[String,Int]]  = sc.textFile("/data/chengqj/xiaodai1.csv",50).flatMap[Tuple2[String,Int]] { line =>
       val fields = line.split(",")
       var vfields =new  Array[Tuple2[String,Int]](4) 
       for(i<- 0 to 3){
         try{
           vfields(i)=Tuple2(fields(i+19),1)
         }catch{                
              case  e:Exception => 
         }          
       }         
       vfields
    }
 var  result=data.reduceByKey(_+_)
 var  ok =result.map{case (word,num) => word}

 //var ranks = data.reduceByKey(_ + _).mapValues(0.15 + 0.85 * _)
 result.count()
}