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
    var data_p = sc.textFile("/data/mllib/test_source_all_clean").map{
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
            var indices = range(0, vfields.length)
            var value = vfields.toArray
            (label, indices, value)
        //}
    }
    var data_p_fi = data_p.filter(r=>r._1==1)

    var data_0827 = sc.textFile("/data/mllib/skb_test_0827").filter{
        line =>
        val fields = line.split("\001")
        fields(fields.length-2).toDouble == 1
    }.map{
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
        var indices = range(0, vfields.length)
        var value = vfields.toArray
        (label, indices, value)
        //The last col is response variable
    }
    var data_union = data_0827.union(data_p_fi.sample(true, 1.5))
    
    data_union.persist()   //StorageLevel.MEMORY_ONLY
    val d = data_union.map{case (label, indices, value) =>
        indices.lastOption.getOrElse(0)
    }.reduce(math.max) + 1

    var data_union_LP = data_union.map{case (label, indices, value) =>
        LabeledPoint(label, Vectors.sparse(d, indices, value))
    }
    //balance data set

    //load true unbalance data set
    var data_0901 = sc.textFile("/data/mllib/skb_test_0901").filter{
        line =>
        val fields = line.split("\001")
        fields(fields.length-2).toDouble == 1   //filter out users who didn't answer the call
    }.map{
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
        var indices = range(0, vfields.length)
        var value = vfields.toArray
        (label, indices, value)
        //The last col is response variable
    }
    
    data_0901.persist()  //StorageLevel.MEMORY_ONLY
    val d = data_0901.map{case (label, indices, value) =>
        indices.lastOption.getOrElse(0)
    }.reduce(math.max) + 1

    var data_0901_LP = data_0901.map{case (label, indices, value) =>
        LabeledPoint(label, Vectors.sparse(d, indices, value))
    }

//===================================================================


    //split data set
    var splits = data_union_LP.randomSplit(Array(0.7, 0.3))
    var (trainData, testData) = (splits(0), splits(1))

    //train model
    var numClasses = 2
    var categoricalFeaturesInfo = Map[Int, Int]()
    var numTrees = 100
    var featureSubsetStrategy = "auto"
    var impurity = "gini"
    var maxDepth = 15
    var maxBins = 32

    var model = RandomForest.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, 
        numTrees, featureSubsetStrategy,impurity, maxDepth, maxBins)

    //Evaluate model
    var labelAndPreds = data_0901_LP.map{
        point =>
        val pred = model.predict(point.features)
        (point.label, pred)
    }
    var testErr = labelAndPreds.filter(r =>r._2==1).count.toDouble/data_0901_LP.count()
    println("pred positive rate = " + testErr+"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")
    var true_positive = labelAndPreds.filter(r=>r._1==r._2 &&r._2==1).count.toDouble/labelAndPreds.filter(r=>r._1==1).count
    println("true_positive = " + true_positive+"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    var precision = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/labelAndPreds.filter(r=>r._2==1).count
    println("TP/(TP+FP)"+precision+"################################################")
    var p_n = labelAndPreds.filter(r=>r._1==1).count.toDouble/labelAndPreds.filter(r=>r._1==0).count
    println("1/0 = "+p_n +"**************************************************************")

//===================================================================
// load data set for prediction
var to_predict = sc.textFile("/data/mllib/pred_source_all_20150905").map{
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
        //var label = fields(0).toDouble
        var label = fields(0)
        var indices = range(0, vfields.length)
        var value = vfields.toArray
        (label, indices, value)
}

to_predict.persist() //StorageLevel.MEMORY_ONLY
val d = to_predict.map{case (label, indices, value) =>
    indices.lastOption.getOrElse(0)
}.reduce(math.max) + 1

var to_predict_LP = to_predict.map{case (label, indices, value) =>
    //LabeledPoint(label, Vectors.sparse(d, indices, value))
    (label, Vectors.sparse(d, indices, value))
}

//==========================================================

//predict and save the positive data
    var mobileAndPreds = to_predict_LP.map{
        point =>
        val pred = model.predict(point._2)
        //val pred = model.predict(point.features)
        (point._1, pred)   //point.label = mobile num
    }

    var saved_data = mobileAndPreds.filter(r=>r._2==1).repartition(1)
    saved_data.saveAsTextFile("/tmp/dyh_output_mobile")

}