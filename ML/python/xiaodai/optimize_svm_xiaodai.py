#coding: utf-8
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector, Vectors
from pyspark.mllib.classification import SVMWithSGD, SVMModel
import math


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
    def save_model(cls, model, path="/data/mllib/model/svm_xiaodai"):
        """
        保存模型
        """
        model.save(sc, path)
    
    @classmethod
    def load_model(cls, modelclass, path="/data/mllib/model/svm_xiaodai"):
        """
        重新载入模型
        """
        return modelclass.load(sc, path)


class SVM_xiaodai(SVM):
    """
    用小贷数据测SPARK mllib SVM
    """
    # @classmethod
    def data_process(self, data, col_rm):
        data = data.map(lambda line: self.load_rm(line, col_rm))
        pdata = self.rebalace_data(data)
        std_map = self.std_map(pdata)
        columns = len(list(pdata.first().features))  #求出数据的列数
        normal_data = pdata.map(lambda p: self.data_normalization(p, columns, std_map))
        trainData, testData = normal_data.randomSplit([0.7, 0.3])
        return (trainData, testData)


    #加载数据并去除某些行
    def load_rm(self,line, col_rm):
        """
        col_rm:要去除的列(从零开始计数)
        """
        tmp = line.strip().split(",")
        features = [tmp[i] for i in xrange(len(tmp)) if i not in col_rm]
        features = ["0" if ele=="" else ele for ele in features]
        return LabeledPoint(tmp[1], features)


    def rebalace_data(self, data):
        """
        用过采的方法对数据进行平衡
        """
        pdata = data
        resample_time = pdata.filter(lambda p:p.label==0).count()/float(pdata.filter(lambda p:p.label==1).count())
        pdata = pdata.filter(lambda p:p.label==0).union(pdata.filter(lambda p:p.label==1).sample(True, resample_time))        
        return pdata

    def std_map(self, pdata):
        """
        #计算各个非离散变量的最大值和最小值，并以字典的形式保存
        #其形式如下
        """
        f_length = len(list(pdata.first().features))
        # normal_map = {}
        # for i in xrange(f_length):
        #     max_v = pdata.map(lambda p: p.features[i]).max()
        #     min_v = pdata.map(lambda p: p.features[i]).min()
        #     normal_map[i] = (max_v, min_v)

        std_map = {}
        for i in xrange(f_length):
            mean_v = pdata.map(lambda p: p.features[i]).mean()
            std_v = math.sqrt(pdata.map(lambda p: p.features[i]).variance())
            std_map[i] = (mean_v, std_v)   

        return std_map   


    def data_normalization(self, p, columns, std_map):
        """
        对数据进行归一化，如反正切，标准化等处理
        """
        log10_trans_col = [12, 13, 15, 16, 25, 27]
        normal_features = []
        for i in xrange(columns):
            # tmp = normal_map[i]
            # normal_features.append((p.features[i]-tmp[1])/(tmp[0]-tmp[1]))
            #归一化处理的公式new_x = (x-min)/(max-min)
            # normal_features.append(math.log10(p.features[i]+1))
            #归一化的公式 new_x = log10(x+1)
            # normal_features.append(math.log10((p.features[i]-tmp[1])/(tmp[0]-tmp[1])))
            #归一化的公式new_x = log10((x-min)/(max-min))
            # if i in log10_trans_col: #对12,13,15,16,25,27进行log10(x)/log10(max)转换
            #     normal_features.append((math.log10(p.features[i]+1))/math.log10(tmp[1]))
            # else:
            #     normal_features.append(p.features[i])
            tmp1 = std_map[i]
            normal_features.append((p.features[i]-tmp1[0])/tmp1[1])
            #进行标准化处理new_x = (x-mean)/std
            # normal_features.append(p.features[i]-tmp1[0])
            # 进行new_x = x - mean
            # normal_features.append(math.atan(p.features[i])*2/math.pi)
            #反正切转换new_x=atan(x)*w/pi
        return LabeledPoint(p.label, normal_features)

if __name__ == '__main__':
    sc = SparkContext(appName='svm_xiaodai')

    data_path = "/data/chengqj/myxiaodai.csv"
    #col_remove = [0,1,2,4,25,26] + range(19,24) + range(28, 36)  #去除所有离散变量
    col_rm = [0,1,2,4,25,26] + range(19,24) #

    try:
        svm = SVM_xiaodai()  #创建对象

        data = sc.textFile(data_path)  #加载数据

        trainData, testData = svm.data_process(data, col_rm)  #数据处理并切分
        trainData.cache()

        model = svm.train_model(trainData, 15) #训练模型   
        
        #svm.save_model(model)    #保存模型

        # model_reloaded = svm.load_model(GradientBoostedTreesModel, "/data/mllib/model/svm_xiaodai")

        evaluate_result = svm.evaluat_model(model, testData)  #评估模型

        print evaluate_result, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

    except Exception,e:
        print "error info^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",e     

