import org.apache.spark._
import org.apache.spark.graphx._
// To make some of the examples work we will also need RDD
import org.apache.spark.rdd.RDD
import  scala.collection.mutable.Set
/**
 * @author chengqj
 *  
 */
object FengKong {
      def  main(args: Array[String]) {
    
        val sparkConf = new SparkConf().setAppName("graph")
        val sc = new SparkContext(sparkConf)
        
        var users: RDD[(VertexId, String)]  = sc.textFile("/user/hive/warehouse/zhengxin.db/fengkong",1000).map { line =>
          //val fields = line.split(",")
          (line.toLong, line)
        }
        
        var behavior = sc.textFile("/user/hive/warehouse/zhengxin.db/fengkong_rela_new_1y_full_g",1000).map { line =>
          val fields = line.split("\001")
       
          try{
             (Edge(fields(0).toLong, fields(1).toLong,""))
         
         }catch{                
              case  e:Exception =>(Edge("189000000".toLong, "189000000".toLong,""))
         } 
        }
        users.cache()
        behavior.cache()
        
        // Define a default user
        var defaultUser = ("189000000")
        
        // Build the initial Graph
        var graph = Graph(users, behavior, defaultUser)
        graph=graph.reverse
        
        
        def msgFun(triplet: EdgeContext[String, String, String]) {
          triplet.sendToDst(triplet.srcAttr)
        }
        
        def mergeStr(a:String,b:String) :String={
            var aSet= Set[String]()
            a.split(" ").foreach { x :String => aSet.add(x) }
            if(aSet.size < 30){
              b.split(" ").foreach { x :String => aSet.add(x) }
            }
            aSet.mkString(" ")
        }
        
        def reduceFun(a: String, b: String): String ={mergeStr(a,b)} 
        
        
        var result = graph.aggregateMessages[String](msgFun, reduceFun)
        result.cache()
        //result.count
        //result.collect.foreach(println(_))
        
        for (i<- 1 to 3){
           graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeStr(oldStr,newStr) )
           
           graph.vertices.saveAsTextFile("/user/root/fengkongn"+i);
           //graph.vertices.collect.foreach(println(_))
           result = graph.aggregateMessages[String](msgFun, reduceFun)
           graph.cache()
           result.cache()   
           //result.count
           //result.collect.foreach(println(_))
        }
        graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeStr(oldStr,newStr) )
        //graph.vertices.collect.foreach(println(_))
        graph.vertices.saveAsTextFile("/user/root/fengkongn4");
    }
}