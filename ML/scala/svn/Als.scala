
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark._
import org.apache.spark.SparkContext
import scala.beans.BeanInfo
/**
 * @author chengqj
 */
@BeanInfo
case class keys(user: Int, product: Int)
object Als extends App {

  val sparkConf = new SparkConf().setAppName("graph")
val sc = new SparkContext(sparkConf)
// Load and parse the data
val data = sc.textFile("file:///root/data/mllib/als/test.data")
val ratings = data.map(_.split(',') match { case Array(user, item, rate) =>
    Rating(user.toInt, item.toInt, rate.toDouble)
  })

// Build the recommendation model using ALS
val rank = 10
val numIterations = 20
val model = ALS.train(ratings, rank, numIterations, 0.01)

// Evaluate the model on rating data
val usersProducts = ratings.map { case Rating(user, product, rate) =>
  (user, product)
}
val predictions = 
  model.predict(usersProducts).map { case Rating(user, product, rate) => 
   (keys(user, product), rate)
  }

val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>  (keys(user, product), rate)}.join(predictions)
val MSE = ratesAndPreds.map { case (keys, (r1, r2)) => 
  val err = (r1 - r2)
  err * err
}.mean()
println("Mean Squared Error = " + MSE)
}