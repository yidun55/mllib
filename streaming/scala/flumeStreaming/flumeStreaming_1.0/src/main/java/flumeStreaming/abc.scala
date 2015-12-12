package flumeStreaming

import org.apache.spark.streaming.flume._
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.SparkContext
import org.apache.spark.streaming.Seconds
import org.apache.spark._

/**
 * @author yn
 */
object abc {
    def main(args:Array[String]){
        val sc = new SparkConf().setMaster("local[2]").setAppName("flume_streaming")
        val ssc = new StreamingContext(sc, Seconds(10))
        val flumeStream = FlumeUtils.createStream(ssc, "10.1.80.65", 33333)
        flumeStream.map(el=> "ok" + el +"yes").print()
        
        ssc.start()
        ssc.awaitTermination()
        ssc.stop()
    }
}