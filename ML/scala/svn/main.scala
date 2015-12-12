import java.io.{FileWriter, BufferedWriter, PrintWriter, File}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark._
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors, SparseVector, Vector}
import org.apache.spark.storage.StorageLevel

object app{
    val sparkConf = new SparkConf().setAppName("pred_source_all")
    val sc = new SparkContext(sparkConf)
    def main(args: Array[String]) = {
       var filename = "/data/mllib/svm_test_dyh"
       var trainer = "kernel"
       var kernelName = "polynomial"   //polynomial, gaussian
       var miniBatch = 0.01
       var stepSize = 1.0
       var regParam = 0.01
       var output = "result.txt"
       var numIterations = 200  
       
       //load data
     
       val data = sc.textFile(filename).map{ line =>
       val tokens = line.split(" ")
       val label = tokens(0)
       val features = tokens.slice(1, tokens.size)
       LabeledPoint(label.toDouble, Vectors.dense(features.map(f => f.toDouble)))
       }
       
       var  splits= data.randomSplit(Array(0.7,0.3))
       var (training,test) =(splits(0),splits(1))
       
       
       /*
       var  city = sc.textFile("/data/chengqj/city.csv",1).collect()
 var  hash=Map[String,Double]()
 city.foreach{x  =>  hash +=(x.split(",")(1)  ->  x.split(",")(0).toDouble)}
 var  data:RDD[LabeledPoint]  = sc.textFile("/data/chengqj/myxiaodai.csv",10).mapPartitions{
    iter  => 
    var res = List[LabeledPoint]() 
    while (iter.hasNext){
       var  line =iter.next()
       println(line)
       val fields = line.split(",",-1)
       var vfields =new  Array[Double](56) 
       vfields(0)=fields(0).toDouble
       for(i<- 2 to 55){
         try{
           if(i==19 ||i==20 || i==21 |i==22)
             if(hash(fields(i+1))==null)
               vfields(i)=0L
             else
               vfields(i)=hash(fields(i+1))
           else
             vfields(i)=fields(i+1).toDouble
         
         }catch{                
              case  e:Exception => vfields(i)=0L
         }          
       }  
       res .::=(LabeledPoint(fields(1).toDouble, Vectors.dense( vfields)) )
    }
    res.iterator

    }
 data.count()
 data.foreach { println(_) }


var splits = data.randomSplit(Array(0.7, 0.3))
var (training,test) =(splits(0),splits(1))

*/
        val biased = false

        //val kernelName = "gaussian"
        //val kernelName = "polynomial"
        val m = KernelSVMWithPegasos.train(training, numIterations, regParam, biased, kernelName)
        //val model = SVMWithSVM.train(training, numIterations)

        // Clear the default threshold.
        m.clearThreshold()
        var model = m
        model
        
        //var test = training.cache()
    //  Compute raw scores on the test set.
        val scoreAndLabels = test.map { point =>
              val score = model.predict(point.features)
              (score, point.label)
        }
       
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
       val auROC = metrics.areaUnderROC()

       println("Area under ROC = " + auROC+"**********************************************")
    }
}