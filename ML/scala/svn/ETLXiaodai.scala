import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import  scala.collection.mutable.Map
import  org.apache.spark.mllib.tree.RandomForest
import  org.apache.spark.mllib.regression.LabeledPoint
import  org.apache.spark.mllib.linalg.Vectors
import  org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils

/**
 * @author Administrator
 */
object ETLXiaodai {
   val sparkConf = new SparkConf().setAppName("MLIB")
   val sc = new SparkContext(sparkConf)
  
 var  city = sc.textFile("/data/chengqj/city.csv",1).collect()
 var  hash=Map[String,Double]()
 city.foreach{x  =>  hash +=(x.split(",")(1)  ->  x.split(",")(0).toDouble)}
 var  data1:RDD[LabeledPoint]  = sc.textFile("/data/chengqj/xiaodai1.csv",10).mapPartitions{
    iter  => 
    //var res =new Array[LabeledPoint](iter.length)
    var res = List[LabeledPoint]() 
    //var j=0
 
    while (iter.hasNext){
       var  line =iter.next()
       println(line)
       val fields = line.split(",")
       var vfields =new  Array[Double](fields.length-1) 
       vfields(0)=fields(0).toDouble
       for(i<- 2 to fields.length-2){
         try{
           if(i==19 ||i==20 || i==21 |i==22)
              vfields(i)=hash(fields(i+1))
           else
             vfields(i)=fields(i+1).toDouble
         
         }catch{                
              case  e:Exception => vfields(i)=0L
         }          
       }  
       res .::=(LabeledPoint(fields(1).toDouble, Vectors.dense( vfields)) )
       //res(j)=LabeledPoint(fields(1).toDouble, Vectors.dense( vfields)) 
       //j +=1
    }
    res.iterator

    }
 data1.count()
 data1.collect()
}