import org.apache.spark._
import org.apache.spark.graphx._
// To make some of the examples work we will also need RDD
import org.apache.spark.rdd.RDD
import  scala.collection.mutable.Set
import scala.collection.mutable.Map
/**
 * @author dengyh
 *  
 */

 object FengKongOverDue {
      def  main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("FengKongOverDue")
        val sc = new SparkContext(sparkConf)
        var card: RDD[(VertexId, String)]  = sc.textFile("/user/hive/warehouse/graphx.db/bianming_vertex",1000).map { line =>
          
          //val fields = line.split(",")
          try{
            (line.toLong, "")
           }catch{                
              case  e:Exception =>( ("189000000".toLong, ""))
         }
        }
        card = card.filter{case (id, (attr))=> id!=189000000L}

        var card_rela: RDD[Edge[String]] = sc.textFile("/user/hive/warehouse/graphx.db/bianming_relationship",1000).map { line =>
          val fields = line.split("\001")
       
          try{
             var  str=fields(0)+":"+fields(1)+":"+fields(2)+":"+fields(3)+":"+fields(4)+":"+(fields(5).replaceAll(" ", "^"))
             (Edge(fields(0).toLong, fields(1).toLong,str))
         
         }catch{                
              case  e:Exception =>(Edge("189000000".toLong, "189000000".toLong,""))
         } 
        }
        card_rela = card_rela.filter{e => e.srcId!=189000000L && e.dstId!=189000000L}

        card.cache()
        card_rela.cache()
        
        // Define a default user
        var defaultUser = ("3000000000000000")
        
        // Build the initial Graph
        var graph = Graph(card, card_rela, defaultUser)


        graph=graph.reverse.cache                            
        def msgFun(triplet: EdgeContext[String, String, String]) {
              if(triplet.srcAttr == "")
                 triplet.sendToDst(Map(triplet.srcAttr -> triplet.srcAttr.toString)
              else
                 triplet.sendToDst(triplet.srcAttr)
        }
        
        def mergeStr(a:String,b:String) :String={
            val flag = b.getOrElse()
            if()
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

           graph.vertices.saveAsTextFile("/data/mllib/graphx/fengkongcn"+i);
           //graph.vertices.collect.foreach(println(_))
           result = graph.aggregateMessages[String](msgFun, reduceFun)
           graph.cache()  
           result.cache()   
           //result.count
           //result.collect.foreach(println(_))
        }
        graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeStr(oldStr,newStr) )
        //graph.vertices.collect.foreach(println(_))
        graph.vertices.saveAsTextFile("/data/mllib/graphx/fengkongcn4");
    }
}





