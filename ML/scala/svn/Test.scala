
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import  scala.collection.mutable.ArrayBuffer
import  org.apache.spark.mllib.tree.RandomForest
import  org.apache.spark.mllib.regression.LabeledPoint
import  org.apache.spark.mllib.linalg.Vectors
import  org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.DenseVector
/**
 * @author Administrator
 */
object Test {
   val sparkConf = new SparkConf().setAppName("MLIB")
   val sc = new SparkContext(sparkConf)
   
   def myfunc[String](iter: Iterator[String]) : Iterator[(String, String)] = {
    var res = List[(String,String)]() 
    var pre = iter.next
    while (iter.hasNext) {
        val cur = iter.next
        res .::= (pre, cur)
        pre = cur;
    } 
    res.iterator
  }
   var  data1  = sc.textFile("/data/chengqj/xiaodai1.csv",20)
   var  data3= data1.mapPartitions(myfunc).collect()
   var  data2= data1.mapPartitions{
    iter  => 
    var res = List[String]() 
    var  j=0
    while (iter.hasNext){
       var  line =iter.next().mkString
       res  .::=(line)
    }
    res.iterator
    }
 data1.count()
 data2.collect
 
}
