# -*- coding: utf-8 -*-
import csv
import re
import nltk.corpus
from collections import defaultdict
import scipy.sparse as sp
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from nltk import SnowballStemmer
import random as rnd 
import logging
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
import operator
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_squared_error
import random
import pandas as pd

dataFolder = "/Users/purav.aggarwal/Documents/Purav/kaggle/avito"
stopwords= frozenset(word.decode('utf-8') for word in nltk.corpus.stopwords.words("russian") if word!="не")    
stemmer = SnowballStemmer('russian')
engChars = [ord(char) for char in u"cCyoOBaAKpPeE"]
rusChars = [ord(char) for char in u"сСуоОВаАКрРеЕ"]
eng_rusTranslateTable = dict(zip(engChars, rusChars))
rus_engTranslateTable = dict(zip(rusChars, engChars))

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)
        
def correctWord (w):
    """ Corrects word by replacing characters with written similarly depending on which language the word. 
        Fraudsters use this technique to avoid detection by anti-fraud algorithms."""

    if len(re.findall(ur"[а-я]",w))>len(re.findall(ur"[a-z]",w)):
        return w.translate(eng_rusTranslateTable)
    else:
        return w.translate(rus_engTranslateTable)

def getItems(fileName, itemsLimit=None):
    """ Reads data file. """
    with open(os.path.join(dataFolder, fileName)) as items_fd:
        logging.info("Sampling...")
        if itemsLimit:
            countReader = csv.DictReader(items_fd, delimiter='\t', quotechar='"')
            numItems = 0
            for row in countReader:
                numItems += 1
            items_fd.seek(0)        
            rnd.seed(0)
            sampleIndexes = set(rnd.sample(range(numItems),itemsLimit))
            
        logging.info("Sampling done. Reading data...")
        itemReader=csv.DictReader(items_fd, delimiter='\t', quotechar = '"')
        itemNum = 0
        for i, item in enumerate(itemReader):
            #item = {featureName:featureValue.decode('utf-8') for featureName,featureValue in item.iteritems()}
            if not itemsLimit or i in sampleIndexes:
                item = {featureName:featureValue.decode('utf-8') for featureName,featureValue in item.iteritems()}
                itemNum += 1
                yield itemNum, item

def readFile(fileName):
    with open(os.path.join(dataFolder, fileName)) as items_fd:
        itemReader=csv.DictReader(items_fd, delimiter='\t', quotechar = '"')
        for i, item in enumerate(itemReader):
            item = {featureName:featureValue.decode('utf-8') for featureName,featureValue in item.iteritems()}
            yield item
            
def getWords(text, stemmRequired = True, correctWordRequired = False):
    """ Splits the text into words, discards stop words and applies stemmer. 
    Parameters
    ----------
    text : str - initial string
    stemmRequired : bool - flag whether stemming required
    correctWordRequired : bool - flag whether correction of words required     
    """

    cleanText = re.sub(u'[^a-zа-я0-9]', ' ', text.lower())
    if correctWordRequired:
        words = [correctWord(w) if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(correctWord(w)) for w in cleanText.split() if len(w)>1 and w not in stopwords]
    else:
        words = [w if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(w) for w in cleanText.split() if len(w)>1 and w not in stopwords]
    
    return words

def processWord(word,index):
    if(index == "category"):
        return "c_"+word
    elif(index == "subcategory"):
        return "s_"+word
    elif(index == "title"):
        return "t_"+word
    elif(index == "attrs"):
        return "a_"+word
    else:
        return word
        
def normList(L, normalizeTo=1):
    vMax = max(L)
    if(vMax <= 0):
        return L
    else:
        return [ x/(vMax*1.0)*normalizeTo for x in L]
    
def processData(fileName, featureIndexes={}, itemsLimit=None,isRegression = False):
    """ Processing data. """
    wordCounts = defaultdict(lambda: 0)
    targets = []
    item_ids = []
    row = []
    col = []
    cur_row = 0
    is_proved = []
    is_blocked = []
    close_hours = []

    for item in readFile(fileName):
        indexes = ["title","description","category","subcategory"]#,"attrs"]
        for index in indexes:
            for word in getWords(item[index], stemmRequired = True, correctWordRequired = False):
                word = processWord(word,index)
                if not featureIndexes:
                    wordCounts[word] += 1
                else:
                    if word in featureIndexes:
                        col.append(featureIndexes[word])
                        row.append(cur_row)
        
        
        #Form a dictonary of Attributes
        import json
        dict_attr = json.loads(re.sub('/\"(?!(,\s"|}))','\\"',item["attrs"]).replace("\t"," ").replace("\n"," ")) if len(item["attrs"])>0 else {}

        '''        
        try:
            dict_attr = json.loads( item['attrs'] )
			# you might need to .encode( 'utf-8' )
        except ValueError:	
            try:
                dict_attr = json.loads( item['attrs'].replace( '/"', r'\"' ))
			# you might need to .encode( 'utf-8' )
            except ValueError:
                print "trying eval()..."
                try: 
                    dict_attr = eval( item['attrs'] ) # no need to encode( 'utf-8' )
                except ValueError:
                    print item['attrs']
        '''                
                         
        for k,v in dict_attr.iteritems():
            word = k+"_"+v
            if not featureIndexes:
                    wordCounts[word] += 1
            else:
                if word in featureIndexes:
                    col.append(featureIndexes[word])
                    row.append(cur_row)
        
        
        if featureIndexes:
            cur_row += 1
            if "is_blocked" in item:
                block = int(item["is_blocked"])
                targets.append(block)
                if isRegression and "is_proved" in item: #Only for TRAINING
                    if block:
                        is_proved.append(int(item["is_proved"]))
                    else:
                        is_proved.append(1)
                    is_blocked.append(block)
                    close_hours.append(float(item["close_hours"]))

            item_ids.append(int(item["itemid"]))
                
    if not featureIndexes:
        index = 0
        for word, count in wordCounts.iteritems():
            if count>=3:
                featureIndexes[word]=index
                index += 1
                
        return featureIndexes
    else:
        features = sp.csr_matrix((np.ones(len(row)),(row,col)), shape=(cur_row, len(featureIndexes)), dtype=np.float64)
        
        regression_targets = []
        if isRegression:
            corr = matthews_corrcoef(is_proved,is_blocked)
            for ele in normList(close_hours):
                regression_targets.append(corr*ele)
            return features, regression_targets,item_ids
        else:
            return features, targets, item_ids
        '''
        if targets:
            return features, targets, item_ids
        else:
            return features, item_ids
        '''
def predictScores(trainFeatures,trainTargets,testFeatures,testItemIds,isRegression = False):
    logging.info("Feature preparation done, fitting model...")
    
    predicted_scores = []
    if isRegression:
        clf = SGDRegressor(     penalty="l2", 
                                alpha=1e-4)
                            
        print("trainFeatures rows::"+str(trainFeatures.shape[0]))
        print("trainTargets rows::"+str(len(trainTargets)))
        clf.fit(trainFeatures,trainTargets)
        logging.info("Predicting...")    
        predicted_scores = clf.predict(testFeatures)
    else:         
        clf = SGDClassifier(    loss="log", 
                                penalty="l2", 
                                alpha=1e-4, 
                                class_weight="auto")
                            
        print("trainFeatures rows::"+str(trainFeatures.shape[0]))
        print("trainTargets rows::"+str(len(trainTargets)))
        clf.fit(trainFeatures,trainTargets)
        logging.info("Predicting...")    
        predicted_scores = clf.predict_proba(testFeatures).T[1]    
    
    logging.info("Write results...")
    output_file = "avito_starter_solution.csv"
    logging.info("Writing submission to %s" % output_file)
    f = open(os.path.join(dataFolder,output_file), "w")
    f.write("id\n")    
    for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse = True):
        f.write("%d\n" % (item_id))
    f.close()

def predictCrossValidatedScore(trainFeatures,trainTargets,trainItemIds,isRegression = False):
    logging.info("Feature preparation done, fitting model...")
                           
    randomPermutation = random.sample(range(trainFeatures.shape[0]), trainFeatures.shape[0])
    numPointsTrain = int(trainFeatures.shape[0]*0.5)
    
    dataTrainFeatures = trainFeatures[randomPermutation[:numPointsTrain]]
    dataValidationFeatures = trainFeatures[randomPermutation[numPointsTrain:]]
    
    dataTrainTargets = [trainTargets[i] for i in randomPermutation[:numPointsTrain]]
    dataValidationTargets = [trainTargets[i] for i in randomPermutation[numPointsTrain:]]

    predicted_scores = []
    if isRegression:
        clf = SGDRegressor(    penalty="l1", 
                                alpha=1e-4)
                            
        print("trainFeatures rows::"+str(trainFeatures.shape[0]))
        print("trainTargets rows::"+str(len(trainTargets)))
        clf.fit(dataTrainFeatures,dataTrainTargets)
        logging.info("Predicting...")    
        predicted_scores = clf.predict(dataValidationFeatures)   
    else:         
        clf = SGDClassifier(    loss="log", 
                                penalty="l2", 
                                alpha=1e-4, 
                                class_weight="auto")
                            
        print("trainFeatures rows::"+str(trainFeatures.shape[0]))
        print("trainTargets rows::"+str(len(trainTargets)))
        clf.fit(dataTrainFeatures,dataTrainTargets)
        logging.info("Predicting...")    
        predicted_scores = clf.predict_proba(dataValidationFeatures).T[1]
            
    error = mean_squared_error(dataValidationTargets,predicted_scores)
    print("% Error:"+ str(error))
    
def main():
    """ Generates features and fits classifier. """
    
    isRegression = True
    featureIndexes = processData(os.path.join(dataFolder,"avito_train.tsv"))
    trainFeatures,trainTargets, trainItemIds=processData(os.path.join(dataFolder,"avito_train.tsv"), featureIndexes,isRegression)
    testFeatures, useless,testItemIds=processData(os.path.join(dataFolder,"avito_test.tsv"), featureIndexes)
    
    predictScores(trainFeatures,trainTargets,testFeatures,testItemIds,isRegression) 
    #predictCrossValidatedScore(trainFeatures,trainTargets,trainItemIds,isRegression)    
    
    logging.info("Done.")
                               
if __name__=="__main__":            
    main()            
    
    
    
'''
itemid 	
category	 - String add - special markers "c_"!
subcategory	 - String add - special markers "s_"!

title	       - String add
description	 - String add

attrs	       - Ignore - a_

price	       - new feature - 150000 - USELESS - too much noise - will have to see if expensice stuff are more likely yo be banned

is_proved	 - Ignore

is_blocked	 - 0 - TARGET

phones_cnt	 - 0
emails_cnt	 - 0
urls_cnt	 - 0


close_hours  - 0.03 - not present in test
'''


