import org.apache.spark._
import org.apache.spark.graphx._
// To make some of the examples work we will also need RDD
import org.apache.spark.rdd.RDD
import  scala.collection.mutable.Set
/**
 * @author chengqj
 *  
 */
object UserNeighbor {
      def  main(args: Array[String]) {
    
        val sparkConf = new SparkConf().setAppName("graph")
        val sc = new SparkContext(sparkConf)
        
        var users: RDD[(VertexId, String)]  = sc.textFile("/root/users.txt").map { line =>
          val fields = line.split(",")
          (fields(0).toLong, fields(1))
        }
        
        var bebavior = sc.textFile("/root/behavior.txt").map { line =>
          val fields = line.split(",")
          (Edge(fields(0).toLong, fields(1).toLong,fields(2)))
        }
        
        // Define a default user
        var defaultUser = ("189114222")
        
        // Build the initial Graph
        var graph = Graph(users, bebavior, defaultUser)
        graph=graph.reverse
        
        def msgFun(triplet: EdgeContext[String, String, String]) {
          triplet.sendToDst(triplet.srcAttr)
        }
        
        def mergeStr(a:String,b:String) :String={
            var aSet= Set[String]()
            a.split(" ").foreach { x :String => aSet.add(x) }
            b.split(" ").foreach { x :String => aSet.add(x) }
            aSet.mkString(" ")
        }
        
        def reduceFun(a: String, b: String): String ={mergeStr(a,b)} 
        
        
        var result = graph.aggregateMessages[String](msgFun, reduceFun)
        result.collect.foreach(println(_))
        
        for (i<- 1 to 4){
           graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeStr(oldStr,newStr) )
           graph.vertices.collect.foreach(println(_))
           result = graph.aggregateMessages[String](msgFun, reduceFun)
           result.collect.foreach(println(_))
        }
        graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeStr(oldStr,newStr) )
        graph.vertices.collect.foreach(println(_))
        graph.vertices.saveAsTextFile("/root/result1");
    }
}