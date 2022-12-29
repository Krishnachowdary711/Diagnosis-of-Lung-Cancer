#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:39:24 2022

@author: sanvireddy
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro, bartlett, levene, ttest_ind, kruskal
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

#reading csv file

data = pd.read_csv(r'/Users/sanvireddy/Downloads/normbysum.csv')
dataCancerOrControl = pd.read_csv(r'/Users/sanvireddy/Downloads/StudyDesignTable - Study Design-Table 1.csv')



# dropping columns whose samples are Serum or Plasma

def dataCleaning(data,dataCancerOrControl):
    data = data.set_index('norm data-Table 1')
    
    for i in range((data.shape[1]-164)):
        data = data.drop(data.columns[164], axis=1)
           
    # dropping columns whose samples are Serum or Plasma
    rslt_df = data.sort_values(by = 'label', axis = 1) 
    
    for i in range(164,183):
        dataCancerOrControl = dataCancerOrControl.drop(i)
        
    # sorting values to add Diesease state to data
    rslt = dataCancerOrControl.sort_values(by="Samplenumber",axis=0)

    extracted_col = rslt['Disease State']
    ls = list(extracted_col)

    rslt_df.loc[len(rslt_df)]=ls
    
    return rslt_df


# function to classify the data by disease state

def classifyControlOrCancer(data,p,SerumOrPlasma):
    control=list()
    cancer=list()
    for i in range(data.shape[1]):

        if data.iat[-1,i]=="cancer" and data.iat[4,i]==SerumOrPlasma:
            cancer.append(data.iat[p,i])
        elif data.iat[4,i]==SerumOrPlasma and data.iat[-1,i]=="control":
            control.append(data.iat[p,i])        
    cancer=np.array(cancer).astype(float)
    control=np.array(control).astype(float)
    return cancer,control


# functions for different types of states

def ShapiroTest(X):
    stat_Shapiro, p_Shapiro = shapiro(X)
    if(p_Shapiro>0.05):
        return 1
    else:
        return 0

def BarlettTest(cancer,control):
    stat_Bartlett, p_Bartlett = bartlett(cancer,control)
    if(p_Bartlett>0.05):
        return 1
    else:
        return 0

def StudentTtest(cancer,control):
    stat_Student, p_Student=ttest_ind(a=cancer, b=control)
    if(p_Student<0.015):
        return 1
    else:
        return 0

def KruskalTest(cancer,control):
    stat_Kruskal, p_Kruskal = kruskal(cancer,control)
    if(p_Kruskal<0.015):
        return 1
    else:
        return 0
    
def LeveneTest(cancer,control):
    stat_Levene, p_Levene = levene(cancer,control)
    if(p_Levene>0.05):
        return 1
    else:
        return 0
           
      
           
def method(data,X,p,SerumOrPlasma):
    cancer,control = classifyControlOrCancer(data, p,SerumOrPlasma)
    
    if ShapiroTest(X):
        if(BarlettTest(cancer,control)):
            return StudentTtest(cancer, control)
        else:
            return KruskalTest(cancer, control)
    else: 
        
        if LeveneTest(cancer,control):
           return StudentTtest(cancer, control)
        else:
            return KruskalTest(cancer, control)
        
        
rslt_df = dataCleaning(data, dataCancerOrControl)
Metabolites_Serum = list()

Metabolites_Plasma = list()


for i in range(6,164):
    
    if(method(rslt_df,list(rslt_df.iloc[i]),i,"Serum")):
        Metabolites_Serum.append(rslt_df.index.values[i])
    if(method(rslt_df,list(rslt_df.iloc[i]),i,"Plasma")):
        Metabolites_Plasma.append(rslt_df.index.values[i])

 
        
print("Potential Metabolites for Serum: \n",*Metabolites_Serum,sep=" , ")
print("Number of metabolites for serum is: ",len(Metabolites_Serum))
print("\n")
print("Potential Metabolites for Plasma: \n",*Metabolites_Plasma,sep=" , ")
print("Number of metabolites for Plasma is: ",len(Metabolites_Plasma))      

final = rslt_df.transpose()
for ind in final.index:
    if(final[405][ind]=='control'):
        final[405][ind]=1
    else:
        final[405][ind]=0

final = final.astype({405:'int'})

plasma = final.iloc[:82,:]
serum = final.iloc[82:,:]

plasma_X,serum_X = pd.DataFrame(),pd.DataFrame()
plasma_Y,serum_Y=plasma[405],serum[405]
serum_Y = serum_Y.rename('Serum')
plasma_Y = plasma_Y.rename('Plasma')
for i in range(len(Metabolites_Plasma)):
    plasma_X[Metabolites_Plasma[i]]=plasma[Metabolites_Plasma[i]]

for i in range(len(Metabolites_Serum)):
    serum_X[Metabolites_Serum[i]]=serum[Metabolites_Serum[i]]
    
plasma_X=plasma_X.astype('float')
serum_X=serum_X.astype('float')

"""
lr=LogisticRegression(C=1, penalty='l2',max_iter=1000)
rfecv = RFECV(estimator=lr, 
              step=1, 
              cv=StratifiedKFold(10),
              scoring='accuracy')
rfecv.fit(plasma_X, plasma_Y)

plt.figure( figsize=(16, 6))
plt.title('Total features selected versus accuracy')
plt.xlabel('Total features selected')
plt.ylabel('Model accuracy')
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
"""
def RFECV_ranking(X,y):
  rf = RandomForestClassifier(n_estimators= 75, random_state=123)
  rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(10), scoring='accuracy')
  rfecv.fit(X,y)
  print('Optimal number of features: {}'.format(rfecv.n_features_))
  plt.figure(figsize=(16, 9))
  plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
  plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
  plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
  plt.show()

  features = [f for f,s in zip(X.columns, rfecv.support_) if s]
  return features

f = RFECV_ranking(plasma_X, plasma_Y)
print(f)

f1 = RFECV_ranking(serum_X, serum_Y)
    
print(f1) 

# Ridge classifier for plasma samples

plasma_X, plasma_Y = shuffle(plasma_X,plasma_Y)
X_train, X_test , y_train, y_test = train_test_split(plasma_X, plasma_Y, test_size=0.1, random_state=42)

ridgecv = RidgeClassifier()
ridgecv.fit(X_train, y_train)


print("Accuracy score for training data (plasma sample - ridge classifier): %.2f" % (accuracy_score(y_train,ridgecv.predict(X_train))*100))
print("Accuracy score for testing data (plasma sample - ridge classifier): %.2f" % (accuracy_score(y_test,ridgecv.predict(X_test))*100))


# XG Boost for serum samples
serum_X,serum_Y = shuffle(serum_X,serum_Y)
X_train, X_test , y_train, y_test = train_test_split(serum_X, serum_Y, test_size=0.1, random_state=42)


model = XGBClassifier()
model.fit(X_train, y_train)

print("Accuracy score for training data (serum sample - xgboost classifier): %.2f" % (accuracy_score(y_train,model.predict(X_train))*100))
print("Accuracy score for testing data (serum sample - xgboost classifier): %.2f" % (accuracy_score(y_test,model.predict(X_test))*100))


#Boxplots
c=1
plt.figure(figsize=(24,23))
plt.title('Boxplot')
for column in serum_X:
    plt.ylabel('Frequency')
    plt.subplot(6,3,c)
    c+=1
    serum_X.boxplot([column])


c=1
plt.figure(figsize=(35,31))
for column in plasma_X:
    plt.ylabel('Frequency')
    plt.subplot(6,5,c)
    c+=1
    plasma_X.boxplot([column])











   