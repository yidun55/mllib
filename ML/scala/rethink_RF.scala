import org.apache.spark._
import org.apache.spark.rdd.RDD 
import scala.collection.mutable.Set
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.regression.LabeledPoint 
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector 
import org.apache.spark.mllib.util.MLUtils
import Array._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import scala.math

/*
dengyouhui
*/
object DRandomForest extends App{
    val sparkConf = new SparkConf().setAppName("pred_source_all")
    val sc = new SparkContext(sparkConf)
//===========================================================
//load Data set for training model

    var del_col = range(80, 83).toBuffer
    del_col += (84)
    del_col ++= range(87, 92)
    del_col += (97)  //the columns to be removed

    //load data which from users buy products
    var data = sc.textFile("/data/mllib/test_source_all_clean").map{
        line => 
        val fields = line.split("\001")
        val all_col = range(1, fields.length).toBuffer
        val exist_col = for(ele<-all_col if(!del_col.exists(s=>s==ele))) yield ele
        var vfields = new ArrayBuffer[Double]()
        for(col <- exist_col){
            try{
                vfields += fields(col).toDouble
            }catch{
                case e:Exception=> vfields(col)=0L
            }
        }
        //if(fields(0).toDouble==1){
            var label = fields(0).toDouble
            // var indices = range(0, vfields.length)
            var value = vfields.toArray
            (label, value)
        //}
    }

    // val d = data.map{case (label, indices, value) =>
    //     indices.lastOption.getOrElse(0)
    // }.reduce(math.max) + 1

    var data_LP = data.map{case (label, value) =>
        LabeledPoint(label, Vectors.dense(value))
    }

    var data_p = data_LP.filter(point=>point.label==1)
    var data_n = data_LP.filter(point=>point.label==0)
    var data_union = data_n.union(data_p.sample(true, 1.0/93)).cache()

//=============================================================================================
//只加载(1,2,6,10,35,37,50,68,69,71,83)
    //var in_col = ArrayBuffer(1,2,6,10,35,37,50,68,69,71,83)
    var in_col = range(20,40)

    //load data which from users buy products
    var data = sc.textFile("/data/mllib/test_source_all_clean").map{
        line => 
        val fields = line.split("\001")
        var vfields = new ArrayBuffer[Double]()
        for(col <- in_col){
            try{
                vfields += fields(col).toDouble
            }catch{
                case e:Exception=> vfields(col)=0L
            }
        }
        //if(fields(0).toDouble==1){
            var label = fields(0).toDouble
            // var indices = range(0, vfields.length)
            var value = vfields.toArray
            (label, value)
        //}
    }

    // val d = data.map{case (label, indices, value) =>
    //     indices.lastOption.getOrElse(0)
    // }.reduce(math.max) + 1

    var data_LP = data.map{case (label, value) =>
        LabeledPoint(label, Vectors.dense(value))
    }

    var data_p = data_LP.filter(point=>point.label==1)
    var data_n = data_LP.filter(point=>point.label==0)
    var data_union = data_p.union(data_n.sample(true, 1.0/93)).cache()
//=============================================================================================

//计算各个非离散变量的最大值和最小值，并以字典的形式保存
//其形式如下 var a = Map(0->(12,1), 1->(13, 2)......)
    var f_length = data_union.first.features.toArray.length   //labeledPoint中features的维度
    var tmp_container = new ArrayBuffer[(Int,(Double, Double))]()
   for(i<- 0 until f_length){
      var mean_v = data_union.map(p=>p.features(i)).mean
      var var_v = math.sqrt(data_union.map(p=>p.features(i)).variance)
      tmp_container += ((i, (mean_v, var_v)))
   }
   val std_map = tmp_container.toMap



    var normal_data:RDD[LabeledPoint] = data_union.map{
          LP=>
             var normal_features = new ArrayBuffer[Double]()
             //val log10_trans_col = ArrayBuffer(12,13,15,16,25,27)
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




    //load true unbalance data set
    var data_0827 = sc.textFile("/data/mllib/skb_test_0827").map{
        line =>
        val fields = line.split("\001")
        val all_col = range(1, fields.length-2).toBuffer  //rm last to col
        val exist_col = for(ele<-all_col if(!del_col.exists(s=>s==ele))) yield ele
        var vfields = new ArrayBuffer[Double]()
        for(col <- exist_col){
            try{
                vfields += fields(col).toDouble
            }catch{
                case e:Exception=> vfields(col)=0L
            }
        }
        var label = fields(fields.length-1).toDouble
        //var indices = range(0, vfields.length)
        var value = vfields.toArray
        LabeledPoint(label, Vectors.dense(value))
        //The last col is response variable
    }


    //计算各个非离散变量的最大值和最小值，并以字典的形式保存
    //其形式如下 var a = Map(0->(12,1), 1->(13, 2)......)
        var f_length_0827 = data_0827.first.features.toArray.length   //labeledPoint中features的维度
        var tmp_container_0827 = new ArrayBuffer[(Int,(Double, Double))]()
       for(i<- 0 until f_length_0827){
          var mean_v = data_0827.map(p=>p.features(i)).mean
          var var_v = math.sqrt(data_0827.map(p=>p.features(i)).variance)
          tmp_container_0827 += ((i, (mean_v, var_v)))
       }
       val std_map_0827 = tmp_container_0827.toMap



//直接归一化
    var normal_data_0827:RDD[LabeledPoint] = data_0827.map{
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
            normal_features += (LP.features(i)-std_map_0827(i)._1)/std_map_0827(i)._2  //进行标准化处理(x-mean)/var  
            //normal_features += LP.features(i)-std_map(i)._1    //进行x-mean处理 
            //normal_features += math.atan(LP.features(i))*2/math.Pi   //反正切转换atan(x)*2/pi,对regParam依赖很强     
         }          
         LabeledPoint(LP.label, Vectors.dense(normal_features.toArray))
      }

//===============================================================================================
    //load true unbalance data set
    var data_0827 = sc.textFile("/data/mllib/skb_test_0827").map{
        line =>
        val fields = line.split("\001")
        var vfields = new ArrayBuffer[Double]()
        for(col <- in_col){
            try{
                vfields += fields(col).toDouble
            }catch{
                case e:Exception=> vfields(col)=0L
            }
        }
        var label = fields(fields.length-1).toDouble
        //var indices = range(0, vfields.length)
        var value = vfields.toArray
        LabeledPoint(label, Vectors.dense(value))
        //The last col is response variable
    }


    //计算各个非离散变量的最大值和最小值，并以字典的形式保存
    //其形式如下 var a = Map(0->(12,1), 1->(13, 2)......)
        var f_length_0827 = data_0827.first.features.toArray.length   //labeledPoint中features的维度
        var tmp_container_0827 = new ArrayBuffer[(Int,(Double, Double))]()
       for(i<- 0 until f_length_0827){
          var mean_v = data_0827.map(p=>p.features(i)).mean
          var var_v = math.sqrt(data_0827.map(p=>p.features(i)).variance)
          tmp_container_0827 += ((i, (mean_v, var_v)))
       }
       val std_map_0827 = tmp_container_0827.toMap



//直接归一化
    var normal_data_0827:RDD[LabeledPoint] = data_0827.map{
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

//=============================================================================================== 

//===================================================================

//===================SVM==============================================================
      var splits = normal_data.randomSplit(Array(0.7, 0.3), seed=1)   //data_union
      var (trainingData, testData) = (splits(0), splits(1))
      trainingData.cache 
       val numIterations = 200  
       val regParam = 0
       val svm = new SVMWithSGD().setIntercept(false)
       svm.optimizer.setStepSize(1.0).setRegParam(regParam).setNumIterations(numIterations)
       //svm.setUpdater(new L1Updater)   //换成L1范式
        val m = svm.run(trainingData)
        // val m = SVMWithSGD.train(trainingData, numIterations)
        // m.clearThreshold()
        m.setThreshold(0)
        var model = m
        var test = testData
      //  Compute raw scores on the test set.
        var labelAndPreds = normal_data_0827.map { point =>        //testData
        val prediction = model.predict(point.features)
          (prediction,point.label)
        }
      var testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / normal_data_0827.count() 
      var precision = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/(labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble+labelAndPreds.filter(r=>r._1==1&&r._2==0).count.toDouble)

      var recall = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/(labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble+labelAndPreds.filter(r=>r._1==0&&r._2==1).count.toDouble)
      
      // var p_n = labelAndPreds.filter(r=>r._2==1).count.toDouble/labelAndPreds.filter(r=>r._2==0).count
      var p_n = labelAndPreds.filter(r=>r._2==1).count.toDouble/labelAndPreds.count() 
      println("Test Error = " + testErr)
      println("TP/(TP+FP)"+precision+"################################################")
      println("TP/(TP+FN)"+recall+"################################################")
//===================SVM==============================================================


    //split data set
    var splits = data_union.randomSplit(Array(0.7, 0.3))
    var (trainData, testData) = (splits(0), splits(1))

    //train model
    var numClasses = 2
    var categoricalFeaturesInfo = Map[Int, Int]()
    var numTrees = 50
    var featureSubsetStrategy = "auto"
    var impurity = "gini"
    var maxDepth = 15
    var maxBins = 32

    var model = RandomForest.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, 
        numTrees, featureSubsetStrategy,impurity, maxDepth, maxBins)

    //Evaluate model
    var labelAndPreds = data_0827_LP.map{
        point =>
        val pred = model.predict(point.features)
        (point.label, pred)
    }
    var testErr = labelAndPreds.filter(r =>r._2==r._1).count.toDouble/labelAndPreds.count()
    println("pred positive rate = " + testErr+"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")
    var true_positive = labelAndPreds.filter(r=>r._1==r._2 &&r._2==1).count.toDouble/labelAndPreds.filter(r=>r._1==1).count
    println("true_positive = " + true_positive+"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    var precision = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/labelAndPreds.filter(r=>r._2==1).count
    println("TP/(TP+FP)"+precision+"################################################")
    var p_n = labelAndPreds.filter(r=>r._1==1).count.toDouble/labelAndPreds.filter(r=>r._1==0).count
    println("1/0 = "+p_n +"**************************************************************")

    val metrics = new BinaryClassificationMetrics(labelAndPreds)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)


}