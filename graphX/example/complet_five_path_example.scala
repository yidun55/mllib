import org.apache.spark._ 
import org.apache.spark.graphx._ 
import org.apache.spark.rdd.RDD
import scala.collection.mutable.Map



//实验
val users: RDD[(VertexId, Map[String, Map[String, String]])] =sc.parallelize(
    Array((3L, Map("3"->Map("3"->"3"))), (7L, Map("7"->Map("7"->"7"))),
    (5L, Map("5"->Map("5"->"5"))), (2L, Map("2"->Map("2"->"2"))),
    (4L, Map("4"->Map("4"->"4")))))

// Create an RDD for edges
val relationships: RDD[Edge[Map[String, Map[String, String]]]] =
  sc.parallelize(Array(Edge(3L, 7L, Map("7"->Map("7"->"7"))),    
    Edge(5L, 3L, Map("3"->Map("3"->"3"))),
                       Edge(2L, 5L, Map("5"->Map("5"->"5"))), 
                       Edge(5L, 7L, Map("7"->Map("7"->"7"))),
                       Edge(4L, 0L, Map("default"->Map("default"->"default"))), 
                         Edge(5L, 0L, Map("default"->Map("default"->"default")))))


// Define a default user in case there are relationship with missing user
val defaultUser = Map("default"->Map("default"->"default"))
// Build the initial Graph
var graph = Graph(users, relationships, defaultUser)
graph=graph.cache  

def getPrimKey(input:Map[String, Map[String, String]]):String ={
    var result = ""
    val keys_iter = input.keys.iterator
    while(keys_iter.hasNext){
        val key_val = keys_iter.next
        val keys_keys_iter = input(key_val).keys.iterator
        while(keys_keys_iter.hasNext){
            val key_key_val = keys_keys_iter.next
            if(key_key_val==input(key_val)(key_key_val) && (input(key_val).toArray.length==1)){
                result = key_val
            }
        }
    }
    if(result==""){
       "100000000"
    }else{
        result
    }
}

                          
def msgFun(triplet: EdgeContext[Map[String, Map[String, String]], Map[String, Map[String, String]], Map[String, Map[String, String]]]) {
    var a = triplet.srcAttr
    var b = triplet.dstAttr
    var flag = true   //是否是第一次传送
    // if(b.size>1){
    //   flag = false
    // }
    val tmp_iter_a = a.keys.iterator
    while(tmp_iter_a.hasNext){
       val a_el = tmp_iter_a.next()
       val tmp_iter_b = b.keys.iterator
       while(tmp_iter_b.hasNext){
          val b_el = tmp_iter_b.next()
          if(a_el==b_el){
             flag = false
          }
       }
    }
    if(a.size==1){   //如果a是graph中的一个端点
        flag = true
    }
    if(flag==true){
        val b_id = getPrimKey(b)
        // val a_key = a.keys.toBuffer(0)
        val a_key = getPrimKey(a)
        // b ++= a
        b ++= Map(a_key->Map(a_key->a_key))
        b(a_key) += (b_id -> (b_id+"|"+a(a_key)(a_key)), a_key->a_key)
        val send_el = b
        triplet.sendToDst(send_el)       
    }
    else{
        val a_id = getPrimKey(a)
        // a-=(a_id)  //去除a中含有顶点的信息
        val tmp_iter_a = a.keys.iterator  //
        // triplet.sendToDst(a)
        while(tmp_iter_a.hasNext){
            val a_key = tmp_iter_a.next    //迭代a中的key
            val tmp_iter_a_a = a(a_key).keys.iterator   //a_key的key
            while(tmp_iter_a_a.hasNext){
                val a_key_key = tmp_iter_a_a.next
                if(a_id!=a_key){
                  if(!b(a_id).contains(a_key_key)){
                      // val tmp_iter_b = b(a_key_key).keys.iterator  //b中a_id对应的内容
                      // while(tmp_iter_b.hasNext){
                      //     val b_key_key = tmp_iter_b.next
                      //     a(a_key) -= (b_key_key)         //去除a.keys/b.keys
                      // }
                      val b_id = getPrimKey(b)
                      val a_new_id = a_key_key
                      // val a_new_id = getPrimKey(Map(a_key->a(a_key)))
                      if(!b(a_id).contains(a_key_key)){  //解除环
                          b(a_id) += (a_new_id->a_new_id, b_id->(b(a_id)(b_id)+"|"+a_new_id))
                          val send_el = b
                          triplet.sendToDst(send_el) 
                      }else{
                          val send_el = b
                          triplet.sendToDst(send_el)
                      } 
                  }
              }
            }
        }
    }
 
}


def mergeStr(a:Map[String, Map[String, String]],b:Map[String, Map[String, String]]) :Map[String, Map[String, String]]={
    b ++= a
    b
}


def reduceFun(a: Map[String, Map[String, String]], b: Map[String, Map[String, String]]): Map[String, Map[String, String]] ={mergeStr(a,b)} 


var result = graph.aggregateMessages[Map[String, Map[String, String]]](msgFun, reduceFun)
result.cache()
//result.count
//result.collect.foreach(println(_))

graph = graph.joinVertices(result)((id, oldStr, newStr) => mergeStr(oldStr,newStr) )
graph.cache()

//graph.vertices.collect.foreach(println(_))
result = graph.aggregateMessages[Map[String, Map[String, String]]](msgFun, reduceFun)
result.cache()   

result.collect.foreach(println(_))

