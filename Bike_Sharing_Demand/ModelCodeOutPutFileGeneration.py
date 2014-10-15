# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import random
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def normalize(train, test):
    norm = preprocessing.Normalizer()
    train = norm.fit_transform(train)
    test = norm.transform(test)
    return train, test

def rmsle(predictions,targets):
        return np.sqrt(((np.log(predictions+1) - np.log(targets+1))** 2).mean())
    
    
tr = pd.read_csv('train.csv',sep=',')
te = pd.read_csv('test.csv',sep=',')

#DataTime - 2011-01-02 01:00:00
tr['year'] = tr.datetime.apply(lambda x: float(x.split(' ')[0].split('-')[0]))
tr['hour'] = tr.datetime.apply(lambda x: float(x.split(' ')[1].split(':')[0]))
tr['month'] = tr.datetime.apply(lambda x: float(x.split(' ')[0].split('-')[1]))
tr['day'] = tr.datetime.apply(lambda x: float(x.split(' ')[0].split('-')[2]))
tr.drop('datetime',axis=1,inplace=True)
te['year'] = te.datetime.apply(lambda x: float(x.split(' ')[0].split('-')[0]))
te['hour'] = te.datetime.apply(lambda x: float(x.split(' ')[1].split(':')[0]))
te['month'] = te.datetime.apply(lambda x: float(x.split(' ')[0].split('-')[1]))
te['day'] = te.datetime.apply(lambda x: float(x.split(' ')[0].split('-')[2]))
test_time = te.datetime
dataTimeTest = te.datetime
te.drop('datetime',axis=1,inplace=True)
tr.drop('temp',axis=1,inplace=True)
te.drop('temp',axis=1,inplace=True)

#Handle Working Dat and Holidays
tr['weekend'] = tr.apply(lambda x: 1 if((x['holiday'] == 0) and (x['workingday'] == 0)) else 0,axis = 1)
tr['working'] = tr.apply(lambda x: 1 if((x['holiday'] == 0) and (x['workingday'] == 1)) else 0,axis = 1)
tr['holiday_'] = tr.apply(lambda x: 1 if((x['holiday'] == 1) and (x['workingday'] == 0)) else 0,axis = 1)

te['weekend'] = te.apply(lambda x: 1 if((x['holiday'] == 0) and (x['workingday'] == 0)) else 0,axis = 1)
te['working'] = te.apply(lambda x: 1 if((x['holiday'] == 0) and (x['workingday'] == 1)) else 0,axis = 1)
te['holiday_'] = te.apply(lambda x: 1 if((x['holiday'] == 1) and (x['workingday'] == 0)) else 0,axis = 1)

tr.drop('workingday',axis=1,inplace=True)
tr.drop('holiday',axis=1,inplace=True)
te.drop('workingday',axis=1,inplace=True)
te.drop('holiday',axis=1,inplace=True)
#

#HANDLE SEASON
lb_season = preprocessing.LabelBinarizer()
lb_season.fit(tr.season.as_matrix(columns=None).astype(np.str))
season_dummy = lb_season.transform(tr.season.as_matrix(columns=None).astype(np.str))
#tr.drop('season',axis=1,inplace=True)
index = 0
for ele in lb_season.classes_[0:]:
	tr['season_'+ele] = season_dummy[:,index]
	index += 1

test_season_dummy = lb_season.transform(te.season.as_matrix(columns=None).astype(np.str))
#te.drop('season',axis=1,inplace=True)
index = 0
for ele in lb_season.classes_[0:]:
	te['season_'+ele] = test_season_dummy[:,index]
	index += 1

#HANDLE WEATHER
lb_weather = preprocessing.LabelBinarizer()
lb_weather.fit(tr.weather.as_matrix(columns=None).astype(np.str))
weather_dummy = lb_weather.transform(tr.weather.as_matrix(columns=None).astype(np.str))
tr.drop('weather',axis=1,inplace=True)
index = 0
for ele in lb_weather.classes_[0:]:
	tr['weather_'+ele] = weather_dummy[:,index]
	index += 1

test_weather_dummy = lb_weather.transform(te.weather.as_matrix(columns=None).astype(np.str))
te.drop('weather',axis=1,inplace=True)
index = 0
for ele in lb_weather.classes_[0:]:
	te['weather_'+ele] = test_weather_dummy[:,index]
	index += 1

'''
randomPermutation = random.sample(range(len(tr)), len(tr))
numPointsTrain = int(len(tr)*0.9)
numPointsValidation = len(tr) - numPointsTrain
dataTrain = tr[tr.index.isin(randomPermutation[:numPointsTrain])]
dataValidation = tr[tr.index.isin(randomPermutation[numPointsTrain:])]
tr_norm,te_norm = normalize(dataTrain.drop(['casual','registered','count'],axis=1),dataValidation.drop(['casual','registered','count'],axis=1))
'''
#tr_norm,te_norm = normalize(tr.drop(['casual','registered','count'],axis=1),dataValidation.drop(['casual','registered','count'],axis=1))

def getNextSetOfTrainAndTestData():
    train_itr = 0
    test_itr_first = test_itr_last = 0
    train_len = len(tr)
    test_len = len(te)
    
    while((test_itr_first < test_len) and (train_itr < train_len)):
        test_year = te['year'][test_itr_first]
        test_month = te['month'][test_itr_first]
        while ((test_itr_last < test_len) and (te['year'][test_itr_last] == test_year) and (test_month == te['month'][test_itr_last])):
            test_itr_last += 1
        while ((train_itr < train_len) and (tr['year'][train_itr] <= test_year) and (test_month >= tr['month'][train_itr])):
            train_itr += 1
        yield(tr[0:train_itr],te[test_itr_first:test_itr_last])
        test_itr_first = test_itr_last

'''
        The decision to go for a bicyce or not hugely would depend on the immediate season going around.
        Including all of the last values adds no significant information.
        Let's just include the data from the training set which has the same season instead.
    
'''   

def getRecentSeasonalTrainTestData():
    train_itr_first = 0
    train_itr = 0
    test_itr_first = test_itr_last = 0
    train_len = len(tr)
    test_len = len(te)
    
    while((test_itr_first < test_len) and (train_itr < train_len)):
        test_year = te['year'][test_itr_first]
        test_month = te['month'][test_itr_first]
        test_season = te['season'][test_itr_first]
        
        while ((test_itr_last < test_len) and (te['year'][test_itr_last] == test_year) and (test_month == te['month'][test_itr_last])):
            test_itr_last += 1
        while ((train_itr < train_len) and (tr['year'][train_itr] <= test_year) and (test_month >= tr['month'][train_itr]) and (tr['season'][train_itr] == test_season)):
            train_itr += 1
        yield(tr[train_itr_first:train_itr],te[test_itr_first:test_itr_last])
        if((train_itr < train_len) and (tr['season'][train_itr] != test_season)):
            train_itr_first = train_itr
        test_itr_first = test_itr_last
        
def getExtensiveSeasonalTrainTestData():
    #train_itr_first = 0
    train_itr = 0
    test_itr_first = test_itr_last = 0
    train_len = len(tr)
    test_len = len(te)
    
    while((test_itr_first < test_len) and (train_itr < train_len)):
        test_year = te['year'].iloc[test_itr_first]
        test_month = te['month'].iloc[test_itr_first]
        test_season = te['season'].iloc[test_itr_first]
        trainIndexes = []        
        
        while ((test_itr_last < test_len) and (te['year'].iloc[test_itr_last] == test_year) and (test_month == te['month'].iloc[test_itr_last])):
            test_itr_last += 1
        while (( train_itr < train_len)):
            if (((tr['year'].iloc[train_itr] < test_year)and( (tr['season'].iloc[train_itr] == test_season) or ((abs(test_month - tr['month'].iloc[train_itr]) < 2)))) or (   (tr['year'].iloc[train_itr] == test_year) and ( (test_month >= tr['month'].iloc[train_itr])and ((test_month - tr['month'].iloc[train_itr]) < 3) ))):
              trainIndexes.append(train_itr)
            train_itr += 1
        
        trainingSet = pd.DataFrame((tr.iloc[i] for i in trainIndexes),columns=tr.columns)
        yield(trainingSet,te[test_itr_first:test_itr_last])
        #if((train_itr < train_len) and (tr['season'].iloc[train_itr] != test_season)):
        #    train_itr_first = train_itr
        train_itr = 0
        test_itr_first = test_itr_last


output = []
for tr_,te_ in getExtensiveSeasonalTrainTestData():
    #tr_,te_ = normalize(tr_.drop(['casual','registered','count'],axis=1),te_)
    print("TrainEnd:"+str(tr_['year'].iloc[len(tr_)-1])+":"+str(tr_['month'].iloc[len(tr_)-1])+"TrainStart:"+str(tr_['year'].iloc[0])+":"+str(tr_['month'].iloc[0]))
    print("TestEnd:"+str(te_['year'].iloc[len(te_)-1])+":"+str(te_['month'].iloc[len(te_)-1])+"TestStart:"+str(te_['year'].iloc[0])+":"+str(te_['month'].iloc[0]))
    
    tr_.drop('season',axis=1,inplace=True)
    te_.drop('season',axis=1,inplace=True)


    clf_casual = ExtraTreesRegressor(n_estimators = 100)
    clf_casual.fit(tr_.drop(['casual','registered','count'],axis=1),np.log(tr_.casual+1))  
    output_casual = np.exp(clf_casual.predict(te_))-1

    clf_registered =  ExtraTreesRegressor(n_estimators = 100)
    clf_registered.fit(tr_.drop(['casual','registered','count'],axis=1),np.log(tr_.registered+1))  
    output_registered = np.exp(clf_registered.predict(te_))-1

    clf_count  = ExtraTreesRegressor(n_estimators = 100)
    clf_count.fit(tr_.drop(['casual','registered','count'],axis=1),np.log(tr_['count']+1))  
    output_count = np.exp(clf_count.predict(te_))-1

    out = ((output_casual + output_registered)+output_count)/2
    #out = output_count
    #out = (output_casual + output_registered)
    #print(out.astype(int))
    output.extend(out.astype(int))
    #print(str(out.astype(int).shape[0]))
    #print(output)

#output = sum(output,[])
open_file_object = csv.writer(open("result_ExtraTreesRegressor_100_.csv", "wb"))
open_file_object.writerow(["datetime","count"])
open_file_object.writerows(zip(dataTimeTest, output))  
        
#clf_casual = linear_model.LogisticRegression()
'''
clf_casual = ExtraTreesRegressor(n_estimators = 100)
clf_casual.fit(tr.drop(['casual','registered','count'],axis=1),tr.casual)  
output_casual = clf_casual.predict(te)

#clf_registered = linear_model.LogisticRegression()
clf_registered =  ExtraTreesRegressor(n_estimators = 100)
clf_registered.fit(tr.drop(['casual','registered','count'],axis=1),tr.registered)  
output_registered = clf_registered.predict(te)

#clf_count = linear_model.LogisticRegression()
clf_count  = ExtraTreesRegressor(n_estimators = 100)
clf_count.fit(tr.drop(['casual','registered','count'],axis=1),tr['count'])  
output_count = clf_count.predict(te)

#output = ((output_casual + output_registered)+output_count)/2
output = (output_casual + output_registered)
output = output.astype(int)

open_file_object = csv.writer(open("result_with_RF_reg_plus_casual_100_ExtraTreesRegressorNorm_withno_new_feature.csv", "wb"))
open_file_object.writerow(["datetime","count"])
open_file_object.writerows(zip(dataTimeTest, output))
'''

'''
print("GradientBoostingRegressor")
clf2_casual = GradientBoostingRegressor()
clf2_casual.fit(tr_norm,np.log(dataTrain.casual+1))
output_casual = np.exp(clf2_casual.predict(te_norm))-1
orig_casual = dataValidation.casual.as_matrix()
print("Error in Casual Prediction:"+str(rmsle(output_casual, orig_casual)))
clf2_registered = GradientBoostingRegressor()
clf2_registered.fit(tr_norm,np.log(dataTrain.registered+1))
output_registered = np.exp(clf2_registered.predict(te_norm))-1
orig_registered = dataValidation.registered.as_matrix()
print("Error in Registered Prediction:"+str(rmsle(output_registered, orig_registered)))
clf2_count = GradientBoostingRegressor()
clf2_count.fit(tr_norm,np.log(dataTrain['count']+1))
output_count = np.exp(clf2_count.predict(te_norm))-1
orig_count = dataValidation['count'].as_matrix()
print("Error in Count Prediction:"+str(rmsle(output_count, orig_count)))
print("Error in (Casual+Registered)+Count)/2 Prediction:"+str(rmsle(((output_casual + output_registered)+output_count)/2, orig_count)))
print("Error in Casual + Registered Prediction:"+str(rmsle(output_registered+orig_casual, orig_count)))

print("DecisionTreeRegressor-n_estimators = 100")
clf2_casual = DecisionTreeRegressor()
clf2_casual.fit(tr_norm,dataTrain.casual)  
output_casual = clf2_casual.predict(te_norm)
orig_casual = dataValidation.casual.as_matrix()
print("Error in Casual Prediction:"+str(rmsle(output_casual, orig_casual)))
clf2_registered = DecisionTreeRegressor()
clf2_registered.fit(tr_norm,dataTrain.registered)  
output_registered = clf2_registered.predict(te_norm)
orig_registered = dataValidation.registered.as_matrix()
print("Error in Registered Prediction:"+str(rmsle(output_registered, orig_registered)))
clf2_count = DecisionTreeRegressor()
clf2_count.fit(tr_norm,dataTrain['count'])  
output_count = clf2_count.predict(te_norm)
orig_count = dataValidation['count'].as_matrix()
print("Error in Count Prediction:"+str(rmsle(output_count, orig_count)))
print("Error in (Casual+Registered)+Count)/2 Prediction:"+str(rmsle(((output_casual + output_registered)+output_count)/2, orig_count)))
print("Error in Casual + Registered Prediction:"+str(rmsle(output_registered+orig_casual, orig_count)))

print("ExtraTreesRegressor")
clf2_casual = ExtraTreesRegressor(n_estimators = 100)
clf2_casual.fit(tr_norm,dataTrain.casual)  
output_casual = clf2_casual.predict(te_norm)
orig_casual = dataValidation.casual.as_matrix()
print("Error in Casual Prediction:"+str(rmsle(output_casual, orig_casual)))
clf2_registered = ExtraTreesRegressor(n_estimators = 100)
clf2_registered.fit(tr_norm,dataTrain.registered)  
output_registered = clf2_registered.predict(te_norm)
orig_registered = dataValidation.registered.as_matrix()
print("Error in Registered Prediction:"+str(rmsle(output_registered, orig_registered)))
clf2_count = ExtraTreesRegressor(n_estimators = 100)
clf2_count.fit(tr_norm,dataTrain['count'])  
output_count = clf2_count.predict(te_norm)
orig_count = dataValidation['count'].as_matrix()
print("Error in Count Prediction:"+str(rmsle(output_count, orig_count)))
print("Error in (Casual+Registered)+Count)/2 Prediction:"+str(rmsle(((output_casual + output_registered)+output_count)/2, orig_count)))
print("Error in Casual + Registered Prediction:"+str(rmsle(output_registered+orig_casual, orig_count)))

print("RandomForestRegressor")
clf2_casual = RandomForestRegressor(n_estimators = 100)
clf2_casual.fit(tr_norm,dataTrain.casual)  
output_casual = clf2_casual.predict(te_norm)
orig_casual = dataValidation.casual.as_matrix()
print("Error in Casual Prediction:"+str(rmsle(output_casual, orig_casual)))
clf2_registered = RandomForestRegressor(n_estimators = 100)
clf2_registered.fit(tr_norm,dataTrain.registered)  
output_registered = clf2_registered.predict(te_norm)
orig_registered = dataValidation.registered.as_matrix()
print("Error in Registered Prediction:"+str(rmsle(output_registered, orig_registered)))
clf2_count = RandomForestRegressor(n_estimators = 100)
clf2_count.fit(tr_norm,dataTrain['count'])  
output_count = clf2_count.predict(te_norm)
orig_count = dataValidation['count'].as_matrix()
print("Error in Count Prediction:"+str(rmsle(output_count, orig_count)))
print("Error in (Casual+Registered)+Count)/2 Prediction:"+str(rmsle(((output_casual + output_registered)+output_count)/2, orig_count)))
print("Error in Casual + Registered Prediction:"+str(rmsle(output_registered+orig_casual, orig_count)))
'''

### IMP. Predition should only be done on past data - so we'll have to use a loop ovee the test data and then ...
'''
RESULTS

ADABOOST
purav-aggarwal:Bike_Sharing_Demand purav.aggarwal$ python ModelCode.py
Error in Casual Prediction:1.06572928113
Error in Registered Prediction:1.25349814796
Error in Count Prediction:1.31590733625
Error in (Casual+Registered)+Count)/2 Prediction:1.31590733625
Error in Casual + Registered Prediction:0.962576663373

RF
purav-aggarwal:Bike_Sharing_Demand purav.aggarwal$ python ModelCode.py
Error in Casual Prediction:0.820654479264
Error in Registered Prediction:0.921871106878
Error in Count Prediction:0.886658934556
Error in (Casual+Registered)+Count)/2 Prediction:0.886658934556
Error in Casual + Registered Prediction:0.744995942393

LogLinear
Error in Casual Prediction:0.985908041826
Error in Registered Prediction:1.05240146567
Error in Count Prediction:1.04159807144
Error in (Casual+Registered)+Count)/2 Prediction:1.04159807144
Error in Casual + Registered Prediction:0.928724539092


AFTER NORMALIZATION:
RandomForestRegressor
Error in Casual Prediction:0.731538194102
Error in Registered Prediction:0.735691134274
Error in Count Prediction:0.714514729179
Error in (Casual+Registered)+Count)/2 Prediction:0.714189641828
Error in Casual + Registered Prediction:0.652469275679

DecisionTreeRegressor-n_estimators = 100
Error in Casual Prediction:0.891626838722
Error in Registered Prediction:0.927186891554
Error in Count Prediction:0.85762802278
Error in (Casual+Registered)+Count)/2 Prediction:0.758144825085
Error in Casual + Registered Prediction:0.791830814665
ExtraTreesRegressor
Error in Casual Prediction:0.697683745252
Error in Registered Prediction:0.698459156273
Error in Count Prediction:0.68375660907
Error in (Casual+Registered)+Count)/2 Prediction:0.6791695477
Error in Casual + Registered Prediction:0.62461847895
GradientBoostingRegressor
Error in Casual Prediction:nan
Error in Registered Prediction:nan
Error in Count Prediction:nan
Error in (Casual+Registered)+Count)/2 Prediction:nan
Error in Casual + Registered Prediction:nan
RandomForestRegressor
Error in Casual Prediction:0.718698020005
Error in Registered Prediction:0.715164498262
Error in Count Prediction:0.693661823517
Error in (Casual+Registered)+Count)/2 Prediction:0.692098973878
Error in Casual + Registered Prediction:0.637402335361


With no new feature except date decompo

DecisionTreeRegressor-n_estimators = 100
Error in Casual Prediction:0.70283415373
Error in Registered Prediction:0.45671905962
Error in Count Prediction:0.466454161803
Error in (Casual+Registered)+Count)/2 Prediction:0.399509380079
Error in Casual + Registered Prediction:0.388420384437
ExtraTreesRegressor
Error in Casual Prediction:0.546023044957
Error in Registered Prediction:0.347394674144
Error in Count Prediction:0.35932642218
Error in (Casual+Registered)+Count)/2 Prediction:0.35034177494
Error in Casual + Registered Prediction:0.299616482426
GradientBoostingRegressor
Error in Casual Prediction:0.562168050916
Error in Registered Prediction:0.442984149147
Error in Count Prediction:0.451831725798
Error in (Casual+Registered)+Count)/2 Prediction:0.442894317005
Error in Casual + Registered Prediction:0.378992852607
RandomForestRegressor
Error in Casual Prediction:0.546926822021
Error in Registered Prediction:0.353306554164
Error in Count Prediction:0.379253360593
Error in (Casual+Registered)+Count)/2 Prediction:0.363582936922
Error in Casual + Registered Prediction:0.306750147237

GOTO: http://beyondvalence.blogspot.in/2014/06/predicting-capital-bikeshare-demand-in.html
'''