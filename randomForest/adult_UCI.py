from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector, Vectors
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
import numpy as np
import re

sc = SparkContext(appName='adult')

def trans_fun(data, col_list):
    """
    data:dataset
    col_list: the col number of categroy feature
    [1,2,3] contains the second, thirth fourth feature
    """
    data = data.map(lambda line:line.strip().split(", "))
    uni_feature = []
    for col in col_list:
        uni_feature.append(np.unique(data.map(lambda x:x[col]).collect()))
    return uni_feature

def binary_vector(x):
    """
    transform to binary_matrix
    """
    a = x.strip().split(", ")
    li_list = []
    te = zip(class_col.value, uni_f.value)
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
    #return LabeledPoint(a[14], Vectors.sparse(len(li_list), indice, value))
    #return LabeledPoint(a[14], (len(li_list), li_list))

def fi(x):
    """
    filter out ?
    """
    pat = r"\?"
    if len(re.findall(pat, x))==0:
        return True
    else:
        return False
    

#load data and prepare it for the ML models
data = sc.textFile("/data/mllib/adult_small.txt")

#filter out observation with "?"
data_fi = data.filter(fi)  

#get the unique features, broadcast the value
fe = trans_fun(data_fi, [1,3,5,6,7,8,9,13])
class_col = sc.broadcast([1,3,5,6,7,8,9,13])
uni_f = sc.broadcast(fe)    #broadcast uni_feature list

#transform raw data to labeledPoint
parsed_data = data_fi.map(binary_vector)
numFeatures = -1
if numFeatures <= 0:
    parsed_data.cache()
    numFeatures = parsed_data.map(lambda x:-1 if x[1].size==0 else x[1][-1]).reduce(max)+1
labeled_data = parsed_data.map(lambda x: LabeledPoint(x[0], Vectors.sparse(numFeatures, x[1],x[2])))

#splite data to trainData and testData
(trianData, testData) = labeled_data.randomSplit([0.7, 0.3])

#MLUtils.saveAsLibSVMFile(labeled_data, "/data/mllib/labeled_adult.txt")
for i in trianData.collect():
    print i, type(i),type(i.label),"two"

# ran = RandomForest.trainClassifier(trianData, numClasses=2,
#                                                 categoricalFeaturesInfo={},
#                                                 numTrees=10,featureSubsetStrategy='auto',impurity='gini',
#                                                 maxDepth=4, maxBins=32)


#for i in trianData.collect():
#    print i.features, "two"



#pred = ran.predict(testData.map(lambda x:x.features))
#lp = testData.map(lambda x: x.label).zip(pred)
#err = lp.filter(lambda (x, y): x!=y).count()/float(testData.count())
#print "haha i'm work", err

model = RandomForest.trainClassifier(trianData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=10, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)


# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
print(model.toDebugString())
