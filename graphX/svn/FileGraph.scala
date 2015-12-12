
import org.apache.spark._
import org.apache.spark.graphx._
// To make some of the examples work we will also need RDD
import org.apache.spark.rdd.RDD
/**
 * @author chengqj
 */
object FileGraph {
      def  main(args: Array[String]) {
      
        val sparkConf = new SparkConf().setAppName("graph")
        val sc = new SparkContext(sparkConf)
        val users: RDD[(VertexId, (String, String))]  = sc.textFile("file:///root/graphx/data/users.txt").map { line =>
        val fields = line.split(",")
        (fields(0).toLong, (fields(1),fields(2)))
        }
        
        val relationships = sc.textFile("file:///root/graphx/data/relation.txt").map { line =>
          val fields = line.split(",")
          (Edge(fields(0).toLong, fields(1).toLong,fields(2)))
        }
        
        // Define a default user in case there are relationship with missing user
        val defaultUser = ("John Doe", "Missing")
        
        
        // Build the initial Graph
        val graph = Graph(users, relationships, defaultUser)
        
        val facts: RDD[String] =
        graph.triplets.map(triplet =>
        triplet.srcAttr._1 + " is the " + triplet.attr + " of " + triplet.dstAttr._1)
        facts.collect.foreach(println(_))
      }
}