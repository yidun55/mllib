#!usr/bin/env python
#coding: utf-8

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel

class GBT_lkl(object):
    """
    用spark自带的数据集sample_libsvm_data测试spark mllib的
    梯度提升算法
    """

    def __init__(self, data_path="/data/mllib/spark_example_data_set/sample_libsvm_data.txt"):
        """
        做数据的加载
        """
        pass
        #self.data = MLUtils.loadLibSVMFile(sc, data_path)

    def data_process(self):
        """
        做前期的各种数据处理
        """
        trainData, testData = self.data.randomSplit([0.7, 0.3])
        return (trainData, testData)
    
    @classmethod
    def train_model(cls, trianData, cateFeaInfo={}, iterTimes=3):
        """
        训练模型
        """
        model = GradientBoostedTrees.trainClassifier(trianData, \
            categoricalFeaturesInfo=cateFeaInfo, numIterations=iterTimes)
        return model
    
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
    def save_model(cls, model, path="$SPARK_HOME/gbts_example"):
        """
        保存模型
        """
        model.save(sc, path)
    
    @classmethod
    def load_model(cls, modelclass, path="$SPARK_HOME/gbts_example"):
        """
        重新载入模型
        """
        return modelclass.load(sc, path)


class parse_xiaodai_streaming(GBT_lkl):
    """
    继承GBT_lkl用于GBTs模型的加载，
    并加入spark streaming 数据处理的方法
    """
    @classmethod
    def data_process(cls, line, col_rm):
        """
        @col_rm：要去除列号组成的列表，比如[0,3,5],就会去除这几列
        @model,做预测的模型
        """
        assert isinstance(col_rm, list), "col_rm must be list"
        tmp = line.strip().split(",")
        features = [tmp[i] for i in xrange(len(tmp)) if i not in col_rm]
        features = ["0" if ele=="" else ele for ele in features]
        return features



def classify(x):
    sample = x.take(x.count())
    for i in sample:
        answer = model.predict(sample)
        print "==============================="
        print answer


if __name__ == '__main__':
    sc = SparkContext("local[2]", "streaming_gbts")

    #加载GBTs模型
    model = parse_xiaodai_streaming.load_model(GradientBoostedTreesModel, path="$SPARK_HOME/gbts_xiaodai_1")

    #create a local streamingcontext with two 
    # working thread and batch interval of 1 second
    ssc = StreamingContext(sc, 1)

    data_process = parse_xiaodai_streaming.data_process
    file_stream = ssc.textFileStream("/data/mllib/streaming/")
    col_rm = [0,1,2,4,25,26] + range(19, 24)
    testData = file_stream.map(lambda line:data_process(line, col_rm))

    testData.pprint()
    testData.foreachRDD(classify)
    # print parse_xiaodai_streaming.evaluat_model(model, testData)

    ssc.start()
    ssc.awaitTermination(100)

