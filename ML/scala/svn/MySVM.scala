import java.io.Serializable

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import breeze.linalg._
import breeze.numerics.{sqrt, cos}
import breeze.stats.distributions.{Uniform, Gaussian}
import org.apache.spark.annotation.Experimental
//import org.apache.spark.mllib.classification.{SVMModelMy, ClassificationModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.rdd.RDDFunctions._
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.util.DataValidators

import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._

class KernelSVMWithPegasos private(
                                    private var numIterations: Int,
                                    private var regParam: Double,
                                    private var biased: Boolean,
                                    private var kernel: (Vector, Vector) => Double)
  extends Serializable {

  protected def createModel(supporters: Array[(LabeledPoint, Double)],
                            kernel: (Vector, Vector) => Double,
                            biased: Boolean,
                            regParam: Double): KernelSVMModel = {
    new KernelSVMModel(supporters, kernel, biased, regParam)
  }

  def run(input: RDD[LabeledPoint]): KernelSVMModel = {
    val sc = input.context
    val data = input.map { point =>
      val y = point.label * 2 - 1
      val x = if (biased) {
        appendBias(point.features)
      } else {
        point.features
      }
      LabeledPoint(y, x)
    }.zipWithIndex().cache()
    val count = data.count()
    val alpha = BSV.zeros[Double](count.toInt)

    for (i <- 1 to numIterations) {
      val stepSize = 1 / (regParam * i)
      val sample = data.takeSample(false, 1, 42 + i)(0)

      val bcSample = sc.broadcast(sample)
      val bcAlpha = sc.broadcast(alpha)

      val res = data.treeAggregate(0.0)(
        seqOp = (c, v) => {
          val y = v._1.label
          val features = v._1.features
          val index = v._2

          if (index != bcSample.value._2) {
            val a = bcAlpha.value(index.toInt)
            val res = y * a * kernel(features, bcSample.value._1.features)
            c + res
          } else {
            c
          }
        },
        combOp = (c1, c2) => {
          c1 + c2
        }
      ) * sample._1.label * stepSize

      if (res < 1) {
        val a = alpha(sample._2.toInt)
        alpha(sample._2.toInt) = a + 1
      }

    }
    val supporters = data.filter { v =>
      val index = v._2
      if (alpha(index.toInt) > 0) {
        true
      } else {
        false
      }
    }.map { v =>
      //(lablePoint, alpha)
      (v._1, alpha(v._2.toInt))
    }.collect()

    createModel(supporters, kernel, biased, regParam * numIterations)
  }
}


/**
 * Top-level methods for calling SVM. NOTE: Labels used in SVM should be {0, 1}.
 */
object KernelSVMWithPegasos {
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             regParam: Double,
             biased: Boolean,
             kernelName: String): KernelSVMModel = {
    val kernel = Kernel.fromName(kernelName)
    new KernelSVMWithPegasos(numIterations, regParam, biased, kernel).run(input)
  }
}

/*
supportVecors: Array[(label, point, alpha)]
 */
class KernelSVMModel(val supportVectors: Array[(LabeledPoint, Double)], val kernel: (Vector, Vector) => Double, biased: Boolean, regParam: Double) extends ClassificationModel with Serializable {

  var intercept: Double = 0
  var threshold: Option[Double] = Some(0.0)

  @Experimental
  def setThreshold(threshold: Double): this.type = {
    this.threshold = Some(threshold)
    this
  }

  @Experimental
  def clearThreshold(): this.type = {
    threshold = None
    this
  }

  protected def predictPoint(dataMatrix: Vector): Double = {
    val margin = supportVectors.map { v =>
      val y = v._1.label
      val features = v._1.features
      val alpha = v._2

      val x = if (biased) {
        appendBias(dataMatrix)
      } else {
        dataMatrix
      }
      alpha * y * kernel(features, x)
    }.sum / regParam + intercept

    threshold match {
      case Some(t) => if (margin > t) 1.0 else 0.0
      case None => margin
    }
  }

  override def predict(testData: RDD[Vector]): RDD[Double] = {
    testData.mapPartitions { iter =>
      iter.map(v => predictPoint(v))
    }
  }

  override def predict(testData: Vector): Double = {
    predictPoint(testData)
  }
}
