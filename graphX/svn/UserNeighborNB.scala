import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import  scala.collection.mutable.Map

/**
 * @author chengqj
 */
object UserNeighborNB {
      def  main(args: Array[String]) {
        
        val sparkConf = new SparkConf().setAppName("graph")
        val sc = new SparkContext(sparkConf)
        
        var users: RDD[(VertexId, String)]  = sc.textFile("/root/users.txt").map { line =>
        val fields = line.split(",")
         (fields(0).toLong, fields(1)+":1")
        }
        
        var bebavior = sc.textFile("/root/behavior.txt").map { line =>
          val fields = line.split(",")
          (Edge(fields(0).toLong, fields(1).toLong,fields(2)))
        }
        
        // Define a default user
        var defaultUser = ("189114222")
        
        // Build the initial Graph
        var graph = Graph(users, bebavior, defaultUser)
        
        var rgraph=graph.reverse
        
        graph.edges.collect.foreach(println(_))
             
        var  times=2
        def makeSendStr(a:String) :String={
            var nmap = Map[String,String]()   
            var resultStr :String=""     
            println(times)
            a.split(" ").foreach { x :String =>  if (!nmap.contains(x.split(":")(0)) )  nmap += (x.split(":")(0)->(""+times))  }
            for(e <- nmap){
              if(resultStr.length()>0)
                resultStr += " "+e._1+":"+e._2
              else
                resultStr = e._1+":"+e._2               
            }         
            resultStr        
        }
        
        
        def msgFun(triplet: EdgeContext[String, String, String]) {
          triplet.sendToDst(makeSendStr(triplet.srcAttr))
        }  

        def mergeStr(a:String,b:String) :String={
            var nmap = Map[String,String]()   
            var resultStr :String=""     
            println(times)
            a.split(" ").foreach { x :String =>  if (!nmap.contains(x.split(":")(0)) )  nmap += (x.split(":")(0)->(""+times))  }
            b.split(" ").foreach { x :String =>  if (!nmap.contains(x.split(":")(0)) )  nmap += (x.split(":")(0)->(""+times))  }
            for(e <- nmap){
              if(resultStr.length()>0)
                resultStr += " "+e._1+":"+e._2
              else
                resultStr = e._1+":"+e._2               
            }           
            resultStr        
        }
        
        def mergeVertice(a:String,b:String) :String={
            var nmap = Map[String,String]()   
            var resultV :String=""  
            a.split(" ").foreach { x :String => if (!nmap.contains(x.split(":")(0)) )  nmap += (x.split(":")(0)->x.split(":")(1))  }
            b.split(" ").foreach { x :String => if (!nmap.contains(x.split(":")(0)) )  nmap += (x.split(":")(0)->x.split(":")(1))  }
            for(e <- nmap){
              if(resultV.length()>0)
                resultV += " "+e._1+":"+e._2
              else
                resultV = e._1+":"+e._2               
            }
            resultV        
        }
        
        
        var result = graph.aggregateMessages[String](msgFun, mergeStr)
        result.collect.foreach(println(_))
        
        for (i<- 1 to 4){
           times=times+1
           graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeVertice(oldStr,newStr) )
           graph.edges.collect.foreach(println(_))
           graph.vertices.collect.foreach(println(_))
           result = graph.aggregateMessages[String](msgFun, mergeStr)
           result.collect.foreach(println(_))
        }
        graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeVertice(oldStr,newStr) )
        graph.edges.collect.foreach(println(_))
        graph.vertices.collect.foreach(println(_))
        
        times=2
        result = rgraph.aggregateMessages[String](msgFun, mergeStr)
        result.collect.foreach(println(_))
        
        for (i<- 1 to 4){
           times=times+1
           rgraph = rgraph.joinVertices(result)((id, oldStr, newStr) => mergeVertice(oldStr,newStr) )
           rgraph.edges.collect.foreach(println(_))
           rgraph.vertices.collect.foreach(println(_))
           result = rgraph.aggregateMessages[String](msgFun, mergeStr)
           result.collect.foreach(println(_))
        }
        rgraph = rgraph.joinVertices(result)((id, oldStr, newStr) => mergeVertice(oldStr,newStr) )
        rgraph.edges.collect.foreach(println(_))
        rgraph.vertices.collect.foreach(println(_))
        //var merge=graph.vertices.union(rgraph.vertices)
        var merge=graph.vertices.innerJoin[String,String](rgraph.vertices)((id, oldStr, newStr) => mergeVertice(oldStr,newStr) )  
        merge.collect.foreach(println(_))
        merge.saveAsTextFile("/root/result3")
     }
}