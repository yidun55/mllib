
import org.apache.spark._
import org.apache.spark.rdd.RDD
import  scala.collection.mutable.Set
import  org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import  org.apache.spark.mllib.regression.LabeledPoint
import  org.apache.spark.mllib.linalg.Vectors
import  org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import java.io.{FileWriter, BufferedWriter, PrintWriter, File}

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark._
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors, SparseVector, Vector}
import org.apache.spark.storage.StorageLevel
import Array._
import scala.collection.mutable.ArrayBuffer
import scala.math
import org.apache.spark.mllib.optimization.L1Updater

/**
 * @author Administrator
 */
object Xiaodaihandle extends App {
  
    val sparkConf = new SparkConf().setAppName("xiaodai")
    val sc = new SparkContext(sparkConf)
   //去除所有的离散变量
     var  data:RDD[LabeledPoint]  = sc.textFile("/data/chengqj/myxiaodai.csv",10).mapPartitions{
        iter  => 
        var res = List[LabeledPoint]() 
        while (iter.hasNext){
           var  line =iter.next()
           line=line.trim()
           println(line)
           var fields = line.split(",",-1)
           var length = fields.length;
           var vfields =new  ArrayBuffer[Double]() 
           //vfields += fields(0).substring(0, 3).toDouble
           var rm_col = range(19,24).toBuffer
           rm_col += (0,1,2,4,25,26)  //去除所有离散变量
           rm_col ++= range(28,36) 
           //rm_col += (0,1,4,25,26)  //去离散变量，但保留伯努利变量
           // rm_col ++= range(6, 17)  //去除所有分布不均衡的变量
           // rm_col += (24,26,36,37,38,44,46,48,51,52,54,56) //去除所有分布不均衡的变量

           /*var rm_col = range(1, 12).toBuffer  //只保留(12,13,15,16,25,27)这几个变量
           rm_col += (14,26)
           rm_col ++= range(17,25)
           rm_col ++= range(28, 57)*/

           for(i<- 0 to length-1){
             try{
               if(rm_col.exists(el=>el==i))
                   println("")
               else
                 vfields+=fields(i).toDouble
             
             }catch{                
                  case  e:Exception => vfields+=0L
             }          
           }  
           res .::=(LabeledPoint(fields(1).toDouble, Vectors.dense( vfields.toArray)) )
        }
        res.iterator

    }


//计算各个非离散变量的最大值和最小值，并以字典的形式保存
//其形式如下 var a = Map(0->(12,1), 1->(13, 2)......)

    var f_length = data.first.features.toArray.length   //labeledPoint中features的维度
    var tmp_container = new ArrayBuffer[(Int,(Double, Double))]()
    for(i<- 0 until f_length){
      //求维度中最大值和最小值
      var max_v = data.map(p=>p.features(i)).max
      var min_v = data.map(p=>p.features(i)).min
      tmp_container += ((i, (max_v, min_v)))
    }
    val normal_map =  tmp_container.toMap

   for(i<- 0 until f_length){
      var mean_v = data.map(p=>p.features(i)).mean
      var var_v = math.sqrt(data.map(p=>p.features(i)).variance)
      tmp_container += ((i, (mean_v, var_v)))
   }
   val std_map = tmp_container.toMap

//直接归一化
    var normal_data:RDD[LabeledPoint] = data.map{
      LP=>
         var normal_features = new ArrayBuffer[Double]()
         val log10_trans_col = ArrayBuffer(12,13,15,16,25,27)
         for(i<- 0 until f_length){
            //normal_features += (LP.features(i)-normal_map(i)._2)/(normal_map(i)._1-normal_map(i)._2)
            //归一化处理的公式 new_x = (x-min)/(max-min)
            //normal_features += math.log10(LP.features(i)+1)
            //归一化处理的公式 new_x = math.log10(x+1)
            //normal_features += math.log10((LP.features(i)-normal_map(i)._2)/(normal_map(i)._1-normal_map(i)._2)+1)
            //归一化处理的公式 new_x = log10((x-min)/(max-min)+1)
            //normal_features += (math.log10(LP.features(i)+1)-math.log10(normal_map(i)._2+1))/(math.log10(normal_map(i)._1+1)-math.log10(normal_map(i)._2+1))
            //归一化处理的公式 new_x = log10((x-min)/(max-min)+1) 
            // if(log10_trans_col.exists(el=>el==i)){// (12,13,15,16,25,27)进行log10(x)/log10(max)转换 
            //     normal_features += math.log10(LP.features(i)+1)/math.log10(normal_map(i)._1)
            // }else{
            //     normal_features += LP.features(i)
            // }
            normal_features += (LP.features(i)-std_map(i)._1)/std_map(i)._2  //进行标准化处理(x-mean)/var  
            //normal_features += LP.features(i)-std_map(i)._1    //进行x-mean处理 
            // normal_features += math.atan(LP.features(i))*2/math.Pi   //反正切转换atan(x)*2/pi,对regParam依赖很强     
         }          
         LabeledPoint(LP.label, Vectors.dense(normal_features.toArray))
      }
      normal_data.count




      var splits = normal_data.randomSplit(Array(0.7, 0.3), seed=1)
      var (trainingData, testData) = (splits(0), splits(1))
      trainingData.cache 
       val numIterations = 200  
       val regParam = 0
       val svm = new SVMWithSGD().setIntercept(false)
       svm.optimizer.setStepSize(1.0).setRegParam(regParam).setNumIterations(numIterations)
       //svm.setUpdater(new L1Updater)   //换成L1范式
        val m = svm.run(trainingData)
        // val m = SVMWithSGD.train(trainingData, numIterations)
        //m.clearThreshold()
        var model = m
        var test = testData
      //  Compute raw scores on the test set.
        var labelAndPreds = testData.map { point =>
        val prediction = model.predict(point.features)
          (prediction,point.label)
        }
      var testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count() 
      var precision = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/(labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble+labelAndPreds.filter(r=>r._1==1&&r._2==0).count.toDouble)

      var recall = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/(labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble+labelAndPreds.filter(r=>r._1==0&&r._2==1).count.toDouble)
      
      var p_n = labelAndPreds.filter(r=>r._2==1).count.toDouble/labelAndPreds.filter(r=>r._2==0).count
      println("Test Error = " + testErr)
      println("TP/(TP+FP)"+precision+"################################################")
      println("TP/(TP+FN)"+recall+"################################################")

}