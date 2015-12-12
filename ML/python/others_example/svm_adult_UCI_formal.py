#!usr/bin/env python
#encoding: utf-8

from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector, Vectors
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.util import MLUtils
import numpy as np
import re

class SVM(object):
    """
    用小贷数据测spark mllib支持向量机
    """

    def __init__(self, data_path="/data/mllib/spark_example_data_set/sample_libsvm_data.txt"):
        """
        做数据的加载
        """
        pass
        #self.data = MLUtils.loadLibSVMFile(sc, data_path)
    @classmethod
    def data_process(self):
        """
        做前期的各种数据处理
        """
        trainData, testData = self.data.randomSplit([0.7, 0.3])
        return (trainData, testData)
    
    @classmethod
    def train_model(cls, trianData, iters=100):
        """
        训练模型
        """
        model = SVMWithSGD.train(trainData, iterations=iters)
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
    def save_model(cls, model, path="/data/mllib/model/svm_adult"):
        """
        保存模型
        """
        model.save(sc, path)
    
    @classmethod
    def load_model(cls, modelclass, path="/data/mllib/model/svm_adult"):
        """
        重新载入模型
        """
        return modelclass.load(sc, path)

class SVM_xiaodai(SVM):
    """
    用小贷数据测SPARK mllib SVM
    """
    # @classmethod
    def data_process(self, data, col_list):
        """
        做前期的各种数据处理
        @col_list: the col number of categroy feature
        [1,2,3] contains the second, thirth fourth feature
        """
        assert isinstance(col_list, list), "col_list must be list"
        data = data.filter(self.fi)     #去除有缺失值的行
        data = data.map(lambda line:line.strip().split(", "))
        uni_feature = []
        for col in col_list:
            uni_feature.append(np.unique(data.map(lambda x:x[col]).collect()))
        pdata = data.map(lambda x:self.to_dumpvars(x, col_list, uni_feature))  #加入哑变量
        pdata.cache()
        numFeatures = pdata.map(lambda x:-1 if x[1].size==0 else x[1][-1]).reduce(max)+1
        labeled_data = pdata.map(lambda x: LabeledPoint(x[0], Vectors.sparse(numFeatures, x[1],x[2])))  #转换成labeledPoint数据类型

        resample_time = labeled_data.filter(lambda p:p.label==0).count()/\
           float(labeled_data.filter(lambda p:p.label==1).count())
        labeled_data = labeled_data.filter(lambda p:p.label==0).union(labeled_data.filter(lambda p:p.label==1).sample(True, resample_time))
        trainData, testData = labeled_data.randomSplit([0.7, 0.3])
        return (trainData, testData)

    def fi(self, x):
        """
        filter out ?
        """
        pat = r"\?"
        if len(re.findall(pat, x))==0:
            return True
        else:
            return False
   
    def to_dumpvars(self, x, class_col, uni_f):
        """
        分类变量转换成哑变量
        @class_col：分类变量的列号
        @uni_f:[["a":2,"b":3], ["ok":5]] 其中"a"和"b"是第一个特征可能的取值以这些
        取值出现的次数，"ok"是第二个特征可能的取值
        """
        a = x
        li_list = []
        te = zip(class_col, uni_f)
        for el in te:
            for element in el[1]:
                if a[el[0]] == element:
                    li_list.append(1)
                else:
                    li_list.append(0)
        li_list.extend([a[i] for i in [0,2,4,10,11,12]])        
        if a[14]=='<=50K':
            a[14] = 0
        else:
            a[14] = 1
        indice = [i for i in xrange(len(li_list)) if li_list[i]!=0]
        value = [li_list[i] for i in xrange(len(li_list)) if li_list[i] != 0]
        indice_te = range(len(li_list))
        
        return float(a[14]), np.array(indice_te, dtype=np.int32), np.array(li_list,dtype=np.float64)


        

if __name__ == '__main__':
    sc = SparkContext(appName='svm_adult')

    data_path = "/data/mllib/adult_small.txt"

    col_list = [1,3,5,6,7,8,9,13]

    try:
        svm = SVM_xiaodai()  #创建对象

        data = sc.textFile(data_path)  #加载数据

        trainData, testData = svm.data_process(data, col_list)  #数据处理并切分
        trainData.cache()
        print trainData.first()    

        model = svm.train_model(trainData, iters=30) #训练模型
        
        #svm.save_model(model)    #保存模型

        # model_reloaded = svm.load_model(SVMModel, "/data/mllib/model/svm_xiaodai")

        evaluate_result = svm.evaluat_model(model, testData)  #评估模型

        print evaluate_result, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

    except Exception, e:
        print "error info^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",e    




