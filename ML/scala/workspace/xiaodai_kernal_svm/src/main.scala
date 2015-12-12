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
import org.apache.spark.rdd.RDD
import Array._
import scala.collection.mutable.ArrayBuffer
import scala.math

object app{
    val sparkConf = new SparkConf().setAppName("xiaodai")
    val sc = new SparkContext(sparkConf)
    def main(args: Array[String]) = {
       var filename = "/data/chengqj/myxiaodai.csv"
//       var kernelName = "polynomial"   //polynomial, gaussian
//       var miniBatch = 0.01
//       var stepSize = 1.0
//       var regParam = 0.01
//       var numIterations = 200  
       
       //load data
       //去除所有的离散变量
        var  data:RDD[LabeledPoint]  = sc.textFile(filename,10).mapPartitions{
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
//            rm_col += (0,1,2,4,25,26)  //去除所有离散变量
//            rm_col ++= range(28,36) 
           rm_col += (0,1,4,25,26)  //去离散变量，但保留伯努利变量
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

      val resample_time = data.filter(p=>p.label==0).count.toDouble/data.filter(p=>p.label==1).count
      data = data.filter(p=>p.label==0).union(data.filter(p=>p.label==1).sample(true, resample_time))


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

    //  直接归一化
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
              //     normal_features += math.log10(LP.features(i)+1)/math.log10(normal_map(i)._2)
              // }else{
              //     normal_features += LP.features(i)
              // }
              normal_features += (LP.features(i)-std_map(i)._1)/std_map(i)._2  //进行标准化处理(x-mean)/var  
              //normal_features += LP.features(i)-std_map(i)._1    //进行x-mean处理 
              //normal_features += math.atan(LP.features(i))*2/math.Pi   //反正切转换atan(x)*2/pi,对regParam依赖很强     
         }          
         LabeledPoint(LP.label, Vectors.dense(normal_features.toArray))
      }
       
       
       var splits = normal_data.randomSplit(Array(0.7, 0.3), seed=1)
       var (trainingData, testData) = (splits(0), splits(1))
       trainingData.cache
       
       val biased = true
       var kernelName = "gaussian"   //polynomial, gaussian, linear
       var miniBatch = 0.01
       var stepSize = 1.0
       var regParam = 10000
       var numIterations = 200  
        val m = KernelSVMWithPegasos.train(trainingData, numIterations, regParam, biased, kernelName)
        //val model = SVMWithSVM.train(training, numIterations)

        // Clear the default threshold.
        m.clearThreshold()
        var model = m
        
    //  Compute raw scores on the test set.
        val scoreAndLabels = testData.map { point =>
              val score = model.predict(point.features)
              (score, point.label)
        }
       
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
       val auROC = metrics.areaUnderROC()
       println("Area under ROC = " + auROC+"**********************************************")
       
        var model = m
    //  Compute raw scores on the test set.
        val LAP = testData.map { point =>
              val score = model.predict(point.features)
              (score, point.label)
        }
       
       val testErr = LAP.filter(p=>p._1!=p._2).count.toDouble/LAP.count
       val precision = LAP.filter(p=>p._1==p._2 && p._1==1).count.toDouble/LAP.filter(p=>p._1==1).count
       val recall = LAP.filter(p=>p._1==p._2 && p._1==1).count.toDouble/LAP.filter(p=>p._2==1).count
       println("Test Error = " + testErr)
       println("TP/(TP+FP)"+precision+"################################################")
       println("TP/(TP+FN)"+recall+"################################################")
       
    }
}