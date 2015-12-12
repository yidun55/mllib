
import org.apache.spark._
import org.apache.spark.graphx._
// To make some of the examples work we will also need RDD
import org.apache.spark.rdd.RDD
import  scala.collection.mutable.Set
import  org.apache.spark.graphx.VertexId
/**
 * @author Administrator
 */
object Trans extends App {
    val sparkConf = new SparkConf().setAppName("graph")
    val sc = new SparkContext(sparkConf)
    
   
  var vertice: RDD[(VertexId, String)]  = sc.textFile("/data/hive/vmerge/*",50).map { line =>
          val fields = line.split(",")
          try{  
                (fields(0).toLong, fields(0)+"#"+fields(1)+"#"+fields(2)+"#"+fields(3))
                // (fields(0).toLong, fields(0)+":"+fields(1))
           }catch{
              case  e:Exception =>{ (0L,"")}
           }    
        }
    
        
    var edges = sc.textFile("/data/hive/etrans/*",20).map { line =>
          val fields = line.split(",")
          try{
            (Edge(fields(0).toLong, fields(1).toLong,"trans"))
          }catch{
              case  e:Exception =>{ Edge(0L,0L,"")}
           }
        }
      // Define a default user
        var defaultUser = ("")
        
        // Build the initial Graph
      
       var graph = Graph(vertice, edges,defaultUser)
       graph=graph.reverse
       graph.vertices
       def msgFun(triplet: EdgeContext[String, String, String]) {
          triplet.sendToDst(triplet.srcAttr)
        }
        
        def mergeStr(a:String,b:String) :String={
            var aSet= Set[String]()
            a.split(" ").foreach { x :String => aSet.add(x) }
            b.split(" ").foreach { x :String => aSet.add(x) }
            aSet.mkString(" ")
        }
        var result = graph.aggregateMessages[String](msgFun, mergeStr)
        //result.collect.foreach(println(_))
        
        for (i<- 1 to 4){
           graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeStr(oldStr,newStr) )
           //graph.vertices.collect.foreach(println(_))
           result = graph.aggregateMessages[String](msgFun, mergeStr)
           //result.collect.foreach(println(_))
        }
        graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeStr(oldStr,newStr) )
        //graph.vertices.collect.foreach(println(_))
        graph.vertices.saveAsTextFile("/root/result1");
    
}