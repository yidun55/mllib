/**
 * Created by LU Tianming on 15-4-9.
 */
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, norm}

object Kernel {
  def linear(v1: Vector, v2: Vector): Double = {
    BDV(v1.toArray).dot(BDV(v2.toArray))
  }
  def gaussian(v1: Vector, v2: Vector): Double = {
    val bv1 = BDV(v1.toArray)
    val bv2 = BDV(v2.toArray)
    val n = norm(bv1 - bv2)
    math.exp(math.pow(n, 2) * -0.5)
  }
  def polynomial(v1: Vector, v2: Vector): Double = {
    val bv1 = BDV(v1.toArray)
    val bv2 = BDV(v2.toArray)
    val k = bv1.dot(bv2)
    math.pow(k + 1, 2)
  }

  def fromName(name: String): (Vector, Vector) => Double = {
    name match {
      case "linear" => linear
      case "gaussian" => gaussian
      case "polynomial" => polynomial
      case _ => linear
    }
  }
}
