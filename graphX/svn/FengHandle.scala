import org.apache.spark._
import org.apache.spark.graphx._
// To make some of the examples work we will also need RDD
import org.apache.spark.rdd.RDD
import  scala.collection.mutable.Set

/**
 * @author Administrator
 */
object FengHandle extends App {
        val sparkConf = new SparkConf().setAppName("graph")
        val sc = new SparkContext(sparkConf)
        
        var data: RDD[ String]  = sc.textFile("/user/root/fengkongn4",100).map { line =>
          line.replaceAll("\\(", "").replaceAll("\\)", "")     
        }
        data.saveAsTextFile("/user/root/fengkongresult");
}