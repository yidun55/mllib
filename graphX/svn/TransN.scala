
import org.apache.spark._
import org.apache.spark.graphx._
// To make some of the examples work we will also need RDD
import org.apache.spark.rdd.RDD
import  scala.collection.mutable.Map
import  org.apache.spark.graphx.VertexId
/**
 * @author Administrator
 */
object TransN extends App {
    val sparkConf = new SparkConf().setAppName("graph")
    val sc = new SparkContext(sparkConf)
    
    
  var vertice: RDD[(VertexId, String)]  = sc.textFile("/data/hive/vmerge/*",100).map { line =>
          val fields = line.split(",")
          try{  
                //(fields(0).toLong, fields(0)+"*"+fields(1)+":1")
                (fields(0).toLong, fields(0)+"#"+fields(1)+"#"+fields(2)+"#"+fields(3)+":1")
           }catch{
              case  e:Exception =>{ (0L,""+":1")}
           }    
        }
        
    var edges = sc.textFile("/data/hive/etrans/*",30).map { line =>
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
       
       
       
       var times =2 
       
       //var times = sc.accumulator(2, "My Accumulator")
       def makeSendStr(a:String) :String={
            var nmap = Map[String,String]()   
            var resultStr :String=""     
           //println("***************"+times)
            //val  times=this.times;
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
            //val  times=this.times;
            //println("***************"+times)
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
            a.split(" ").foreach { x :String =>
                if(x.split(":").length>1 ){
                  val attr=x.split(":")(0)
                  val level = x.split(":")(1)
                  if (!nmap.contains(attr)  )  
                    nmap +=(attr->level)
                  else if  (nmap.contains(attr) &&  nmap(attr).toLong >level.toLong )
                    nmap +=(attr->level)
                }
            }
            b.split(" ").foreach { x :String =>
               if(x.split(":").length>1 ){
                  val attr=x.split(":")(0)
                  val level = x.split(":")(1)
                  if (!nmap.contains(attr)  )  
                    nmap +=(attr->level)
                  else if  (nmap.contains(attr) &&  nmap(attr).toLong >level.toLong )
                    nmap +=(attr->level)
               }
            }        
            for(e <- nmap){
              if(resultV.length()>0)
                resultV += " "+e._1+":"+e._2
              else
                resultV = e._1+":"+e._2               
            }
            resultV        
        }

     
        var result = graph.aggregateMessages[String](msgFun, mergeStr)
        result.count();
        //result.collect.foreach(println(_))
        //graph.vertices.collect.foreach(println(_))
        result.cache()
        for (i<- 1 to 3){
           times +=1
           graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeVertice(oldStr,newStr) )
           graph.edges.count()
           graph.vertices.count()
           graph.cache()
           //graph.edges.collect.foreach(println(_))
           //graph.vertices.collect.foreach(println(_))
           result = graph.aggregateMessages[String](msgFun, mergeStr)
           result.count()
           //result.collect.foreach(println(_))
        }
        graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeVertice(oldStr,newStr) )
        println("***************"+times)
        graph.vertices.saveAsTextFile("/root/result65")
        //graph.edges.collect.foreach(println(_))
        //graph.vertices.collect.foreach(println(_))
    
}