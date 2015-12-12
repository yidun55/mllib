import org.apache.spark._ 
import org.apache.spark.graphx._ 
import org.apache.spark.rdd.RDD
import scala.collection.mutable.Map



//实验
val users: RDD[(VertexId, Map[String, Int])] =sc.parallelize(
    Array((3L, Map("3"->1)), (7L, Map("7"->1)),
    (5L, Map("5"->1)), (2L, Map("2"->1)),
    (4L, Map("4"->1))))

// Create an RDD for edges
val relationships: RDD[Edge[Map[String, Int]]] =
  sc.parallelize(Array(Edge(3L, 7L, Map("7"->1)),    
    Edge(5L, 3L, Map("3"->1)),
                       Edge(2L, 5L, Map("5"->1)), 
                       Edge(5L, 7L, Map("7"->1)),
                       Edge(4L, 0L, Map("default"->1)), 
                         Edge(5L, 0L, Map("default"->1))))


// Define a default user in case there are relationship with missing user
val defaultUser = Map("default"->1)
// Build the initial Graph
var graph = Graph(users, relationships, defaultUser)
graph=graph.cache  

                          
def msgFun(triplet: EdgeContext[Map[String, Int], Map[String, Int], Map[String, Int]]) {
    var a = triplet.srcAttr
    var b = triplet.dstAttr
    val a_iter = a.keys.iterator
    while(a_iter.hasNext){
        val a_key = a_iter.next
        if(!b.contains(a_key)){
            b += (a_key -> (a(a_key)+1))
        }
    }
    triplet.sendToDst(b) 

}


def mergeStr(a:Map[String, Int],b:Map[String, Int]) :Map[String, Int]={
    val a_iter = a.keys.iterator
    while(a_iter.hasNext){
        val a_key = a_iter.next
        if(!b.contains(a_key)){
            b += (a_key -> a(a_key))
        }else{
            if(b(a_key)>=a(a_key)){
                b(a_key) = a(a_key)  //取小的
            }
        }
    }
    b
}


def reduceFun(a: Map[String, Int], b: Map[String, Int]): Map[String, Int] ={mergeStr(a,b)} 


var result = graph.aggregateMessages[Map[String, Int]](msgFun, reduceFun)
result.cache()
//result.count
//result.collect.foreach(println(_))

graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeStr(oldStr,newStr) )
graph.cache()

//graph.vertices.collect.foreach(println(_))
result = graph.aggregateMessages[Map[String, Int]](msgFun, reduceFun)
result.cache()   

result.collect.foreach(println(_))

