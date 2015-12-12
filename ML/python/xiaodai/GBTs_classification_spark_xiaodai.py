#!usr/bin/env python
#coding: utf-8

from pyspark import SparkContext
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors, SparseVector
from pyspark.mllib.regression import LabeledPoint


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


class GBT_lkl_xiaodai(GBT_lkl):
    """
    用小贷数据测SPARK mllib GBTs
    """
    @classmethod
    def data_process_1(cls, data, col_rm):
        """
        做前期的各种数据处理
        @col_rm:要删除数据的列数
        """
        assert isinstance(col_rm, list), "col_rm must be list"
        pdata = data.map(lambda line: _load_rm(line, col_rm))
        resample_time = pdata.filter(lambda p:p.label==0).count()/\
           float(pdata.filter(lambda p:p.label==1).count())
        pdata = pdata.filter(lambda p:p.label==0).union(pdata.filter(lambda p:p.label==1).sample(True, resample_time))
        trainData, testData = pdata.randomSplit([0.7, 0.3])
        return (trainData, testData)
    
def _load_rm(line, col_rm):
    """
    col_rm:要去除的列(从零开始计数)
    """
    tmp = line.strip().split(",")
    features = [tmp[i] for i in xrange(len(tmp)) if i not in col_rm]
    features = ["0" if ele=="" else ele for ele in features]
    return LabeledPoint(tmp[1], features)



if __name__ == '__main__':
    sc = SparkContext(appName='GBTs_xiaodai')

    data_path = "/data/chengqj/myxiaodai.csv"

    col_rm = [0,1,2,4,25,26] + range(19, 24)

    try:
        gbts = GBT_lkl_xiaodai()  #创建对象

        data = sc.textFile(data_path)  #加载数据

        trainData, testData = GBT_lkl_xiaodai.data_process_1(data, col_rm)  #数据处理并切分
        trainData.cache()
        print trainData.first()    

        model = gbts.train_model(trainData, iterTimes=15) #训练模型
        
        #gbts.save_model(model)    #保存模型

        # model_reloaded = gbts.load_model(GradientBoostedTreesModel, "$SPARK_HOME/gbts_example")

        evaluate_result = gbts.evaluat_model(model, testData)  #评估模型

        print evaluate_result, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

        #feature = ['27', '1', '5', '5', '0', '3', '3', '0', '3000', '3000', '4', '3000', '1', '3', '3000', '3000', 0, '1', '0', '0', '0', '0', '0', '0', '0', '4', 0, '23', '6', '283000', '3', '94333.33333', 0, 0, 0, '2', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
       
        # features_te = Vectors.dense(feature)

        # print model_reloaded.predict(features_te), "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

    except Exception,e:
        print "error info^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",e







