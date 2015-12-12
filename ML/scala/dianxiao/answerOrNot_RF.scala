import org.apache.spark._
import org.apache.spark.rdd.RDD 
import scala.collection.mutable.Set
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.classification.{SVMModel,SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint 
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector 
import org.apache.spark.mllib.util.MLUtils
import Array._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel


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

    var del_col_te = for(ele<-del_col) yield ele + 3
    var del_col1 = del_col_te.toBuffer
    del_col1 ++= range(0, 4)
    
    var data = sc.textFile("/data/mllib/test_source_all",50).map{
        line =>
        val fields = line.split("\001")
        val all_col = range(1, fields.length).toBuffer

        val feature_col = for(ele<-all_col if(!del_col1.exists(s=>s==ele))) yield ele
        var vfields = new ArrayBuffer[Double]()
        for(col <- feature_col){
            try{
                vfields += fields(col).toDouble
            }catch{
                case e:Exception=> vfields(col)=0L
            }
        }
        //if(fields(0).toDouble==1){
            var label = fields(3).toDouble
            var indices = range(0, vfields.length)
            var value = vfields.toArray
            (label, indices, value)
        //}
    }

    val d = data.map{case (label, indices, value) =>
        indices.lastOption.getOrElse(0)
    }.reduce(math.max) + 1

    var data_LP = data.map{case (label, indices, value) =>
        LabeledPoint(label, Vectors.sparse(d, indices, value))
    }

    var data_p = data_LP.filter(point=>point.label==1)
    var data_n = data_LP.filter(point=>point.label==0)
    var data_bal = data_n.union(data_p.sample(true, data_n.count/data_p.count.toDouble)).cache()


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
        var label = fields(fields.length-2).toDouble
        var indices = range(0, vfields.length)
        var value = vfields.toArray
        (label, indices, value)
        //The last col is response variable
    }

    val d = data_0827.map{case (label, indices, value) =>
        indices.lastOption.getOrElse(0)
    }.reduce(math.max) + 1

    var data_0827_LP = data_0827.map{case (label, indices, value) =>
        LabeledPoint(label, Vectors.sparse(d, indices, value))
    }

//===================================================================


    //split data set
    var splits = data_bal.randomSplit(Array(0.7, 0.3))
    var (trainData, testData) = (splits(0), splits(1))

/*    //train model
    var numClasses = 2
    var categoricalFeaturesInfo = Map[Int, Int]()
    var numTrees = 5
    var featureSubsetStrategy = "auto"
    var impurity = "gini"
    var maxDepth = 10
    var maxBins = 32

    var model = RandomForest.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, 
        numTrees, featureSubsetStrategy,impurity, maxDepth, maxBins)

*/

    //Train a GradientBoostedTrees model
    //The defaultParams for Classification use LogLoss by default
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 30 
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 10
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model = GradientBoostedTrees.train(trainData, boostingStrategy)

    //Evaluate model
    var labelAndPreds = testData.map{
        point =>
        val pred = model.predict(point.features)
        (point.label, pred)
    }
    var testErr = labelAndPreds.filter(r =>r._2!=r._1).count.toDouble/labelAndPreds.count()
    println("testErr = " + testErr+"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")
    var true_positive = labelAndPreds.filter(r=>r._1==r._2 &&r._2==1).count.toDouble/labelAndPreds.filter(r=>r._1==1).count
    println("true_positive = " + true_positive+"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    var precision = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/labelAndPreds.filter(r=>r._2==1).count
    println("TP/(TP+FP)"+precision+"################################################")
    var p_n = labelAndPreds.filter(r=>r._1==1).count.toDouble/labelAndPreds.filter(r=>r._1==0).count
    println("1/0 = "+p_n +"**************************************************************")



}