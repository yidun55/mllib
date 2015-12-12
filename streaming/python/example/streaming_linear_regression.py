from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD


#create a local streamingcontext with two 
#working thread and batch interval of 1 second
sc = SparkContext("local[2]", "streaming_lr")
ssc = StreamingContext(sc, 1)

def parse(lp):
	label = float(lp[lp.find("(") + 1: lp.find(",")])
	vec = Vectors.dense(lp[lp.find("[") + 1: lp.find("]")].split(","))
	return LabeledPoint(label, vec)

trainingData = ssc.textFileStream("/training/data/dir").map(parse).cache()
testData = ssc.textFileStream("/testing/data/dir").map(parse)

numFeatures = 3
model = StreamingLinearRegressionWithSGD()
model.setInitialWeigths([0.0,0.0,0.0])

model.trainOn(trainingData)
print(model.predictOnValues(testData.map(lambda lp: (lp.label, lp.features))))

ssc.start()
ssc.awaitTermination()

