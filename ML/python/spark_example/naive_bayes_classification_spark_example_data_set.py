#!usr/bin/env python
#coding: utf-8

from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext

class naive(object):
    """
    spark mllib官网上的例子
    """
    @classmethod
    def data_process(cls, data):
        data = data.map(naive.parse)
        training, test = data.randomSplit([0.6, 0.4], seed=0)
        return training, test

    @staticmethod
    def parse(line):
        parts = line.split(",")
        label = float(parts[0])
        features = Vectors.dense([float(x) for x in parts[1].split(" ")])
        return LabeledPoint(label, features)

    @classmethod
    def train(cls, data, s_lambda=1.0):
        """
        @data, LabeledPoint组成RDD
        @s_lambda, 平均指数,默认拉普拉斯平滑(s_lambda=1.0)
        """
        first = data.first()
        assert isinstance(first, LabeledPoint), "data, LabeledPoint组成RDD"
        return NaiveBayes.train(data, s_lambda)

    @classmethod
    def evaluat_model(cls, model, testData):
        """
        评估模型
        testErr:错误率
        precison:精度(TP/(TP+NP))
        recall:召回率(TP/(TP+NF))
        """
        pred = model.predict(testData.map(lambda x:x.features))
        LAP = testData.map(lambda p:p.label).zip(pred)
        testErr = LAP.filter(lambda (v,p):v!=p).count()/float(LAP.count())
        precison = LAP.filter(lambda (v,p):v==p==1).count()/\
                    float(LAP.filter(lambda (v,p):p==1).count())
        recall = LAP.filter(lambda (v,p):v==p==1).count()/\
                    float(LAP.filter(lambda (v,p):v==1).count())

        return {"testErr":testErr, "precison":precison, \
               "recall":recall}

    @classmethod
    def save_model(cls, model, path="$SPARK_HOME/NaiveBayes"):
        """
        """
        model.save(sc, path)

    @classmethod
    def load_model(cls, path="$SPARK_HOME/NaiveBayes"):
        """
        """
        return NaiveBayesModel.load(sc, path)    


if __name__ == '__main__':
    sc = SparkContext(appName='naive')
    #加载数据
    data = sc.textFile("/data/mllib/spark_example_data_set/sample_naive_bayes_data.txt")

    trainData, testData = naive.data_process(data)
    trainData.cache()

    model = naive.train(trainData)

    print naive.evaluat_model(model, testData)
