#!usr/bin/env python
#coding: utf-8

from pyspark import SparkContext
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors, SparseVector
import numpy as np


class GBT_lkl(object):
    """
    用spark自带的数据集sample_libsvm_data测试spark mllib的
    梯度提升算法
    """

    def __init__(self, data_path):
        """
        做数据的加载
        """
        self.data = MLUtils.loadLibSVMFile(sc, data_path)

    def data_process(self):
        """
        做前期的各种数据处理
        """
        trainData, testData = self.data.randomSplit([0.7, 0.3])
        return (trainData, testData)

    def train_model(self, trianData, cateFeaInfo={}, iterTimes=3):
        """
        训练模型
        """
        model = GradientBoostedTrees.trainClassifier(trianData, \
            categoricalFeaturesInfo=cateFeaInfo, numIterations=iterTimes)
        return model

    def evaluat_model(self, model, testData):
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

    def save_model(self, model, path="$SPARK_HOME/gbts_example"):
        """
        保存模型
        """
        model.save(sc, path)

    def load_model(self, modelclass, path="$SPARK_HOME/gbts_example"):
        """
        重新载入模型
        """
        return modelclass.load(sc, path)

sc = SparkContext(appName='spark_example')

data_path = "/data/mllib/spark_example_data_set/sample_libsvm_data.txt"

try:
    gbts = GBT_lkl(data_path)  #创建对象

    trainData, testData = gbts.data_process()  #切分训练集和测试集

    #model = gbts.train_model(trainData)   #训练模型
    
    #gbts.save_model(model)    #保存模型

    model_reloaded = gbts.load_model(GradientBoostedTreesModel, "$SPARK_HOME/gbts_example")

    evaluate_result = gbts.evaluat_model(model_reloaded, testData)  #评估模型

    print evaluate_result, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

    new_indices = [127,128,129,130,131,154,155,156,157,158,159,181,182,183,184,185,186,187,188,189,207,208,209,210,211,212,213,214,215,216,217,235,236,237,238,239,240,241,242,243,244,245,262,263,264,265,266,267,268,269,270,271,272,273,289,290,291,292,293,294,295,296,297,300,301,302,316,317,318,319,320,321,328,329,330,343,344,345,346,347,348,349,356,357,358,371,372,373,374,384,385,386,399,400,401,412,413,414,426,427,428,429,440,441,442,454,455,456,457,466,467,468,469,470,482,483,484,493,494,495,496,497,510,511,512,520,521,522,523,538,539,540,547,548,549,550,566,567,568,569,570,571,572,573,574,575,576,577,578,594,595,596,597,598,599,600,601,602,603,604,622,623,624,625,626,627,628,629,630,651,652,653,654,655,656,657]
    new_values = [51.0,159.0,253.0,159.0,50.0,48.0,238.0,252.0,252.0,252.0,237.0,54.0,227.0,253.0,252.0,239.0,233.0,252.0,57.0,6.0,10.0,60.0,224.0,252.0,253.0,252.0,202.0,84.0,252.0,253.0,122.0,163.0,252.0,252.0,252.0,253.0,252.0,252.0,96.0,189.0,253.0,167.0,51.0,238.0,253.0,253.0,190.0,114.0,253.0,228.0,47.0,79.0,255.0,168.0,48.0,238.0,252.0,252.0,179.0,12.0,75.0,121.0,21.0,253.0,243.0,50.0,38.0,165.0,253.0,233.0,208.0,84.0,253.0,252.0,165.0,7.0,178.0,252.0,240.0,71.0,19.0,28.0,253.0,252.0,195.0,57.0,252.0,252.0,63.0,253.0,252.0,195.0,198.0,253.0,190.0,255.0,253.0,196.0,76.0,246.0,252.0,112.0,253.0,252.0,148.0,85.0,252.0,230.0,25.0,7.0,135.0,253.0,186.0,12.0,85.0,252.0,223.0,7.0,131.0,252.0,225.0,71.0,85.0,252.0,145.0,48.0,165.0,252.0,173.0,86.0,253.0,225.0,114.0,238.0,253.0,162.0,85.0,252.0,249.0,146.0,48.0,29.0,85.0,178.0,225.0,253.0,223.0,167.0,56.0,85.0,252.0,252.0,252.0,229.0,215.0,252.0,252.0,252.0,196.0,130.0,28.0,199.0,252.0,252.0,253.0,252.0,252.0,233.0,145.0,25.0,128.0,252.0,253.0,252.0,141.0,37.0]
    
    features_te = Vectors.sparse(692, new_indices, new_values)

    print model_reloaded.predict(features_te), "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

except Exception,e:
    print "error info^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",e







