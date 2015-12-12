import org.apache.spark._
import org.apache.spark.graphx._
// To make some of the examples work we will also need RDD
import org.apache.spark.rdd.RDD
import  scala.collection.mutable.Set
/**
 * @author chengqj
 *  
 */
object FengKongFull {
      def  main(args: Array[String]) {
    
        val sparkConf = new SparkConf().setAppName("graph")
        val sc = new SparkContext(sparkConf)
        
        var users: RDD[(VertexId, String)]  = sc.textFile("/user/hive/warehouse/zhengxin.db/fengkong",1000).map { line =>
          //val fields = line.split(",")
          (line.toLong, "")
        }
        
        var behavior = sc.textFile("/user/hive/warehouse/zhengxin.db/fengkong_rela_full_1y_r",1000).map { line =>
          val fields = line.split("\001")
       
          try{
             var  str=fields(0)+":"+fields(1)+":"+fields(2)+":"+fields(3)+":"+fields(4)+":"+(fields(5).replaceAll(" ", "^"))
             (Edge(fields(0).toLong, fields(1).toLong,str))
         
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
          if(triplet.srcAttr != "")
             triplet.sendToDst(triplet.srcAttr+" "+triplet.attr)
          else
             triplet.sendToDst(triplet.attr)
        }
        
        def mergeStr(a:String,b:String) :String={
            var aSet= Set[String]()
            a.split(" ").foreach { x :String => aSet.add(x) }
            if(aSet.size < 100){ 
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
           
           graph.vertices.saveAsTextFile("/user/root/fengkongba"+i);
           //graph.vertices.collect.foreach(println(_))
           result = graph.aggregateMessages[String](msgFun, reduceFun)
           graph.cache()
           result.cache()   
           //result.count
           //result.collect.foreach(println(_))
        }
        graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeStr(oldStr,newStr) )
        //graph.vertices.collect.foreach(println(_))
        graph.vertices.saveAsTextFile("/user/root/fengkongba4");
    }
}