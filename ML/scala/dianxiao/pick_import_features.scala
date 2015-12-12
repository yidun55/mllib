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
//function for data transform
    def numerical(in_path:String, col_num:Int)={
        /*
        get the mean of the column containing na
        */
        val in = sc.textFile(in_path)
        val fi_data = in.filter{line=>
                val f = line.split("\001")
                f(col_num) != "NA"
            }

        val me_value = fi_data.map{line=>
            val fields = line.split("\001")
            fields(col_num).toDouble
        }.reduce(_+_)/fi_data.count

        val na_map = Map("NA"->me_value)
        na_map

    }

    def classi(in_path:String,col_num:Int)={
        /*
        tranform features from char to numerical
        */
        val in = sc.textFile(in_path)
        val data_pairs = in.map{line=>
            val fields = line.split("\001")
            (fields(col_num), 0)
        }.reduceByKey(_+_)
        
        var uni = data_pairs.collect()
        var labels = ArrayBuffer[String]()
        uni.foreach{ele=>
            labels += ele._1
        }
        var zi = labels.zipWithIndex
        var zi_map = zi.toMap
        zi_map   
    }    

//======================================================
// load data and transform data
    val path = "/data/mllib/test_source_all_clean"
    var class_f = range(80, 83).toBuffer
    class_f.remove(1)    //remove 81th feature
    class_f += (84)
    class_f ++= Array(88)
    class_f ++= Array(90)
    var num_f = Array(91)
    //all_map = Map(87->Map("长沙"->1), ......)
    var all_map = ArrayBuffer[Map[String, Any]]()
    for(col<-class_f){
        all_map += classi(path,col)
    }
    for(col<-num_f){
        all_map += numerical(path, col)
    }
    var col_all = class_f.toBuffer
    col_all ++= num_f
    var all_map_zip = (col_all zip all_map).toMap //Map(87->Map("长沙"->1), ......)
    all_map_zip ++= Map(84 -> Map("3" -> 0, "1" -> 1, "2"->"3", "NA"->4))  //因为data_0827中有NA


    var data = sc.textFile("/data/mllib/test_source_all_clean").map{
        line => 
        val fields = line.split("\001")
        //val all_col = range(1, fields.length).toBuffer
        val tmp_col = Array(3,4,8,10,14,15,17,21,23,25,26,27,30,36,38,45,49,50,59,72,
            73,79,82,83,85,86,87,88,91,93,96,99,103).toBuffer //挑选Lily模型中的重要变量
        val all_col = tmp_col.map(i=>i-3)
        // all_col.remove(all_col.indexOf(81))   //remove 81th column
        var vfields = new ArrayBuffer[Double]()
        for(col <- all_col){
            if(class_f.exists(el=>el==col)){
                try{
                    vfields += all_map_zip(col)(fields(col)).toString.toDouble
                }catch{
                    case e:Exception=> vfields(col)=0L
                }
            }else if(num_f.exists(el=>el==col)){
                try{
                    if(fields(col)=="NA"){
                        vfields += all_map_zip(col)("NA").toString.toDouble
                    }else{
                        vfields += fields(col).toDouble
                    }
                }catch{
                        case e:Exception=>println("error")
                    }
                }
            else{
                //try{
                    vfields += fields(col).toDouble
                //}catch{
                //    case e:Exception=> vfields(col)=0L
                //}                
            }
        }
        //if(fields(0).toDouble==1){
            var label = fields(0).toDouble
            var indices = range(0, vfields.length)
            var value = vfields.toArray
            (label, indices, value)
        //}
    }
    val d = data.first._2.length
    var data_LP = data.map{case (label, indices, value) =>
        LabeledPoint(label, Vectors.sparse(d, indices, value))
    }

    var data_p = data_LP.filter(point=>point.label==1)
    var data_n = data_LP.filter(point=>point.label==0)
    var data_union = data_n.union(data_p.sample(true, 93)).cache()  //随机过采
    //var data_union = data_p.union(data_n.sample(false, 0.0107)).cache()  //随机欠采


    var data_0827 = sc.textFile("/data/mllib/skb_test_0827").map{
        line =>
        val fields = line.split("\001")
        //val all_col = range(1, fields.length-2).toBuffer  //rm last to col
        val tmp_col = Array(3,4,8,10,14,15,17,21,23,25,26,27,30,36,38,45,49,50,59,72,
            73,79,82,83,85,86,87,88,91,93,96,99,103).toBuffer //挑选Lily模型中的重要变量
        val all_col = tmp_col.map(i=>i-3)
        // all_col.remove(all_col.indexOf(81))   //remove 81th column
        var vfields = new ArrayBuffer[Double]()
        for(col <- all_col){
            if(class_f.exists(el=>el==col)){
                try{
                    vfields += all_map_zip(col)(fields(col)).toString.toDouble
                }catch{
                    case e:Exception=> vfields(col)=0L
                }
            }else if(num_f.exists(el=>el==col)){
                try{
                    if(fields(col)=="NA"){
                        vfields += all_map_zip(col)("NA").toString.toDouble
                    }else{
                        vfields += fields(col).toDouble
                    }
                }catch{
                        case e:Exception=>println("error")
                    }
                }
            else{
                //try{
                    vfields += fields(col).toDouble
                //}catch{
                //    case e:Exception=> vfields(col)=0L
                //}                
            }
        }
        //if(fields(0).toDouble==1){
            var label = fields(fields.length-1).toDouble
            var indices = range(0, vfields.length)
            var value = vfields.toArray
            (label, indices, value)
        //}
    }

    val d = data_0827.first._2.length
    var data_0827_LP = data_0827.map{case (label, indices, value) =>
        LabeledPoint(label, Vectors.sparse(d, indices, value))
    }.cache()

//===================================================================


    //split data set
    var splits = data_union.randomSplit(Array(0.7, 0.3))
    var (trainData, testData) = (splits(0), splits(1))

    //train model
    var numClasses = 2
    var categoricalFeaturesInfo = Map[Int, Int]()
    var numTrees = 20
    var featureSubsetStrategy = "auto"
    var impurity = "gini"
    var maxDepth = 15
    var maxBins = 32

    val tmp_col = Array(3,4,8,10,14,15,17,21,23,25,26,27,30,36,38,45,49,50,59,72,
        73,79,82,83,85,86,87,88,91,93,96,99,103).toBuffer //挑选Lily模型中的重要变量
    val all_col = tmp_col.map(i=>i-3)
    val trans_map = all_col.zipWithIndex.toMap   //转换后数据的位置发生改变    
    for(col<-col_all){
        if(col!=91){
            categoricalFeaturesInfo += (trans_map(col)->all_map_zip(col).toArray.length)
        }
    }
    val keys = Array(32) //挑选Lily模型中的重要变量
    val valu = Array(2) //挑选Lily模型中的重要变量
    val tmp_map = (keys zip valu).toMap
    categoricalFeaturesInfo ++= tmp_map


    //var trainData = data_union //随机欠采
    var model = RandomForest.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, 
        numTrees, featureSubsetStrategy,impurity, maxDepth, maxBins)


    var labelAndPreds = testData.map{
        point =>
        val pred = model.predict(point.features)
        (point.label, pred)
    }
    var testErr = labelAndPreds.filter(r =>r._2!=r._1).count.toDouble/labelAndPreds.count()
    println("testErr = " + testErr+"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")
    var true_positive = labelAndPreds.filter(r=>r._1==r._2 &&r._2==1).count.toDouble/labelAndPreds.filter(r=>r._1==1).count
    println("recall = " + true_positive+"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    var precision = labelAndPreds.filter(r=>r._1==r._2&&r._2==1).count.toDouble/labelAndPreds.filter(r=>r._2==1).count
    println("TP/(TP+FP)"+precision+"################################################")
    var p_n = labelAndPreds.filter(r=>r._1==1).count.toDouble/labelAndPreds.filter(r=>r._1==0).count
    println("1/0 = "+p_n +"**************************************************************")

}
