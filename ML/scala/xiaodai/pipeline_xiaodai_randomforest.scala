import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils

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

//去除所有的离散变量
     var  raw_data:RDD[LabeledPoint]  = sc.textFile("/data/chengqj/myxiaodai.csv",10).mapPartitions{
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


// Load and parse the data file, converting it to a DataFrame.
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt").toDF()

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a RandomForest model.
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Chain indexers and forest in a Pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Train model.  This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("precision")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println("Learned classification forest model:\n" + rfModel.toDebugString)


