from pyspark import  SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector, Vectors
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
import numpy as np
import re

sc = SparkContext(appName='skb')

def trans_fun(data, col_list):
    """
    data:dataset
    col_list: the col number of categroy feature
    [1,2,3] contains the second, thirth fourth feature
    """
    data = data.map(lambda line:line.strip().split("\001"))
    uni_feature = []
    for col in col_list:
        uni_feature.append(list(np.unique(data.map(lambda x:x[col]).collect())))
    return uni_feature

def feature_char_to_num(x):
    """
    change categorical features from char to numerical by its index in uni_feature
    list eg. ["deng", "shi"] -> [0, 1]
    """
    a = x.strip().split("\001")
    li_list = []
    te = zip(class_col.value, uni_f.value)
    for el in te:
        a[el[0]] = el[1].index(a[el[0]])  #change categorical feature to its index in list
                                          #e.g. ["deng", "shi"] -> [0, 1]
                      
    del a[84]   #rm 85th element
    value = a[1:-2] 
    indice_te = range(len(value))

    return float(a[-1]), np.array(indice_te, dtype=np.int32), np.array(value,dtype=np.float64)
    #return LabeledPoint(a[14], Vectors.sparse(len(li_list), indice, value))
    #return LabeledPoint(a[14], (len(li_list), li_list))

def fi(x):
    """
    filter out the user who didn't answer the call
    """
    a = x.strip().split("\001")
    if float(a[-2])==1:
        return True
    else:
        return False


def filter_positive(x):
    """
    find out user who buy products
    """
    a = x.strip().split("\001")
    if int(a[0])==1:
        return True
    else:
        return False


def data_p_std(x):
    """
    make data from test_source_all_clean to format of 
    data from skb_test_0827
    """
    tmp = x.strip().split("\001")
    tmp.extend([tmp[0]]*2)
    return "\001".join(tmp)+"\n"

def filter_positive_data(x):
    """
    find out user who buy products from skb_test_0827
    """
    a = x.strip().split("\001")
    if float(a[-1]) ==1:
        return True
    else:
        return False

#load data and prepare it for the ML models
data_0827 = sc.textFile("/data/mllib/skb_test_0827")
data1 = sc.textFile("/data/mllib/skb_test_0901")

#filter out user who didn't answer the call
data_ans = data1.filter(fi)
data_ans_0827 = data_0827.filter(fi)

data_p = sc.textFile("/data/mllib/test_source_all_clean")
data_p_fi = data_p.filter(filter_positive)
data_p_std = data_p_fi.map(data_p_std)

#use pred_source_all_com_small training tree
data_pred_source_all = sc.textFile("/data/mllib/pred_source_all_com_small").map(data_p_std)

#data_p_std = data.filter(filter_positive_data).sample(True, 50) #resampling positive sample

#union data and data_p
data_trans_feature = data_ans.union(data_p_std).union(data_ans_0827)
#data_union = data_ans.sample(False, 0.5).union(data_p_std)  #balance data set by reduce negative data
data_union = data_ans.union(data_p_std.sample(True, 1.5))  #balance data set by reduce negative data

#get the unique features, broadcast the value
#col_na = range(80, 83) + [84] + range(87, 92) + [97]
col_na = range(80, 83) + range(87, 92) + [97]
fe = trans_fun(data_trans_feature, col_na)
class_col = sc.broadcast(col_na)
uni_f = sc.broadcast(fe)    #broadcast uni_feature list


#transform raw data to labeledPoint
parsed_data = data_union.map(feature_char_to_num)
#parsed_data = data_pred_source_all.map(feature_char_to_num)



numFeatures = -1
if numFeatures <= 0:
    parsed_data.cache()
    numFeatures = parsed_data.map(lambda x:-1 if x[1].size==0 else x[1][-1]).reduce(max)+1
labeled_data = parsed_data.map(lambda x: LabeledPoint(x[0], Vectors.sparse(numFeatures, x[1],x[2])))

unbalance_test = data_ans_0827.map(feature_char_to_num).cache()
l_unbal_te = unbalance_test.map(lambda x: LabeledPoint(x[0], Vectors.sparse(numFeatures, x[1], x[2])))


#splite data to trainData and testData
(trianData, testData) = labeled_data.randomSplit([0.9, 0.1])

len_list = [len(i) for i in fe]
col_na_l = [i-1 for i in col_na]  #because slice out the first data in vector [1:-2]
col_na_l = [i-1 for i in col_na_l if i >= 83]   #for drop out the 85th col
features_dict = dict(zip(col_na_l, len_list))  #feature dict eg. {1:3, 5:8}
model = RandomForest.trainClassifier(trianData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=50, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=5, maxBins=32)


# Evaluate model on test instances and compute test error
predictions = model.predict(l_unbal_te.map(lambda x: x.features))
labelsAndPredictions = l_unbal_te.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(l_unbal_te.count())
print('Test Error = ' + str(testErr) +"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
true_positive = labelsAndPredictions.filter(lambda (v,p):v==p and p==1).count()/float(labelsAndPredictions.filter(lambda (v,p):v==1).count())
print "true_positive", true_positive, "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
f_true = labelsAndPredictions.filter(lambda (v,p):v==p and v==1).count()/float(labelsAndPredictions.filter(lambda (v,p):p==1).count())
print "precision=TP/(TP+Fp)", f_true, "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
print "1/0",labelsAndPredictions.filter(lambda (v,p):v==1).count()/float(labelsAndPredictions.filter(lambda (v,p):v==0).count()), "##############################################################################################"
#print "False", labeled_data.filter(lambda p:p.label==0).count(), "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2"
#print "Positive",  labeled_data.filter(lambda p:p.label==1).count(),"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

