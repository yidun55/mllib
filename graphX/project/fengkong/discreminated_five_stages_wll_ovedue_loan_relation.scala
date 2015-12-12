import org.apache.spark._ 
import org.apache.spark.graphx._ 
import org.apache.spark.rdd.RDD
import scala.collection.mutable.Map
/**
 * @author dengyh
 *  
 */

object FengKongOverDue {
      def  main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("FengKongOverDue")
        val sc = new SparkContext(sparkConf)
        var card: RDD[(VertexId, Map[String, Int])]  = sc.textFile("/user/hive/warehouse/graphx.db/bianming_vertex/000000_0",1000).map { line =>
          
          //val fields = line.split(",")
          try{
            (line.toLong, Map(""->1))
           }catch{                
              case  e:Exception =>( ("189000000".toLong, Map(""->1)))
         }
        }
        card = card.filter{case (id, (attr))=> id!=189000000L}

        var card_rela: RDD[Edge[Map[String, Int]]] = sc.textFile("/user/hive/warehouse/graphx.db/bianming_relationship/000434_0",1000).map { line =>
          val fields = line.split("\001")
       
          try{
             var  str=fields(0)+":"+fields(1)+":"+fields(2)+":"+fields(3)+":"+fields(4)+":"+(fields(5).replaceAll(" ", "^"))
             (Edge(fields(0).toLong, fields(1).toLong,Map(str->1)))
         
         }catch{                
              case  e:Exception =>(Edge("189000000".toLong, "189000000".toLong,Map(""->1)))
         } 
        }
        card_rela = card_rela.filter{e => e.srcId!=189000000L && e.dstId!=189000000L}

        card.cache()
        card_rela.cache()
        
        // Define a default user
        var defaultUser = (Map("3000000000000000"->1))
        
        // Build the initial Graph
        var graph = Graph(card, card_rela, defaultUser)


        graph=graph.reverse.cache                            
        def msgFun(triplet: EdgeContext[Map[String, Int], Map[String, Int], Map[String, Int]]) {
            var a = triplet.srcAttr
            var b = triplet.dstAttr
            var c = triplet.attr
            val a_iter = a.keys.iterator
            while(a_iter.hasNext){
                val a_key = a_iter.next
                if(!b.contains(a_key)){
                    b += (a_key -> (a(a_key)+1))
                }
            }
            val c_iter = c.keys.iterator    //把边上的信息加到顶点中
            while(c_iter.hasNext){
                val c_key = c_iter.next
                if(!b.contains(c_key)){
                    b += (c_key -> (c(c_key)+1))
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

        for (i<- 1 to 3){
           graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeStr(oldStr,newStr) )

           graph.vertices.saveAsTextFile("/data/mllib/graphx/te_my"+i);
           //graph.vertices.collect.foreach(println(_))
           result = graph.aggregateMessages[Map[String, Int]](msgFun, reduceFun)
           graph.cache()  
           result.cache()   
           //result.count
           //result.collect.foreach(println(_))
        }
        graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeStr(oldStr,newStr) )
        graph.vertices.collect.foreach(println(_))
        graph.vertices.saveAsTextFile("/data/mllib/graphx/te_my4");                
      }
}