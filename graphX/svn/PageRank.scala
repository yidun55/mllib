import org.apache.spark._
import org.apache.spark.graphx._
// To make some of the examples work we will also need RDD
import org.apache.spark.rdd.RDD

/**
 * @author chengqj
 */
object PageRank {
   def  main(args: Array[String]) {
      
        val sparkConf = new SparkConf().setAppName("graph")
        val sc = new SparkContext(sparkConf)
        
        val graph = GraphLoader.edgeListFile(sc, "file:///root/graphx/data/followers.txt")
        // Run PageRank
        val ranks = graph.pageRank(0.001).vertices
  
        
        println(ranks.collect().mkString("\n"))
     
   }
}