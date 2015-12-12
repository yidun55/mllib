import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import Array._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map


//去除所有的离散变量
 var  data:RDD[(String, Array[String])]  = sc.textFile("/data/chengqj/myxiaodai.csv",10).mapPartitions{
    iter  => 
    var res = List[(String, Array[String])]() 
    while (iter.hasNext){
       var  line =iter.next()
       line=line.trim()
       println(line)
       var fields = line.split(",",-1)
       var length = fields.length;
       var vfields =new  ArrayBuffer[String]() 
       //vfields += fields(0).substring(0, 3).toDouble
       var rm_col = Array(0,1,2,4,23,25,26).toBuffer
       rm_col ++= Array(12,13,15,18,24,27,40,42,49,52,53)  //连续变量

       for(i<- 0 to length-1){
         try{
           if(rm_col.exists(el=>el==i))
               println("")
           else
             vfields+=fields(i)
         
         }catch{                
              case  e:Exception => vfields+=""
         }          
       }  
       res .::= ((fields(1), vfields.toArray))
    }
    res.iterator

}

val resample_time = data.filter(p=>p._1=="0").count.toDouble/data.filter(p=>p._1=="1").count
data = data.filter(p=>p._1=="0").union(data.filter(p=>p._1=="1").sample(true, resample_time))

//计算各个非离散变量的最大值和最小值，并以字典的形式保存
//其形式如下 var a = Map(0->(12,1), 1->(13, 2)......)

var f_length = data.first._2.length   //labeledPoint中features的维度
var tmp_container = new ArrayBuffer[(Int,Map[String, Double])]()
for(i<- 0 until f_length){
  //求维度中各个特征出现的次数
  val features_times = data.map(p=>(p._2(i),1)).groupByKey().map{e=>(e._1, e._2.sum)}
  val total_times = features_times.map(p=>p._2).sum
  val features_prob = features_times.map(e=>(e._1, e._2.toDouble/total_times))  //求每个特征值的概率
  val fea_arr = features_prob.toArray.iterator
  var tmp_map = Map[String, Double]()
  while (fea_arr.hasNext){
    val tmp = fea_arr.next()
    tmp_map += (tmp._1 -> tmp._2)
  }
  tmp_container += ((i, tmp_map))
}
val normal_map =  tmp_container.toMap



val normal_data = data.mapPartitions{
    iter =>
    var res = List[LabeledPoint]()
    while (iter.hasNext){
        var normal_features = new ArrayBuffer[Double]()
        val lp = iter.next()
        for(i <- 0 until f_length){
            val total_l = normal_map(i)._1 - normal_map(i)._2
            val trans_val = ((lp.features(i)-normal_map(i)._2)/(total_l/1000))
            val val_n = f"$trans_val%1.2f".toDouble
            normal_features += (val_n)
        }
        res .::=(LabeledPoint(lp.label, Vectors.dense(normal_features.toArray)))
    }
    res.iterator

}


val splits = normal_data.randomSplit(Array(0.6, 0.4), seed=0)
val trainData = splits(0)
trainData.cache
val testData = splits(1)

val model = NaiveBayes.train(trainData)   //, lambda = 1.0, modelType = "multinomial"


//评估模型函数
def evaluated_model(testdata:RDD[LabeledPoint], model:NaiveBayesModel):Map[String, Double]={
    val PAL = testdata.map{
        lp =>
        val pred = model.predict(lp.features)
        (pred, lp.label)
    }
    var testErr = PAL.filter(r => r._1 != r._2).count.toDouble / PAL.count()

    var precision = PAL.filter(r=>r._1==r._2&&r._2==1).count.toDouble/PAL.filter(r=>r._1==1).count.toDouble
    var recall = PAL.filter(r=>r._1==r._2&&r._2==1).count.toDouble/PAL.filter(r=>r._2==1).count.toDouble
    val total = Map("testErr"->testErr, "precision"->precision, "recall"->recall)
    total      
}

val result = evaluated_model(testData, model)
println(result)


